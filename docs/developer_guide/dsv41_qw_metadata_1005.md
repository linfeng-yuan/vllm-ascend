# V4.1 A5 graph-side metadata and indexer Q/W fusion

Base: `e41bd634c6a753151fe191b25d81489161d3e763` on
`1005_950DT_vllm0300_rebase_main`. Review branch:
`perf/1005-dsv41-qw-overlap`, fork PR #10.

## Scope

Only two independently opt-in features are included:

```json
{
  "multistream_dsv41_metadata": true,
  "enable_dsv41_indexer_qw_fusion": true
}
```

Both default to false. Indexer K/postscatter integration and its graph-bucket
whitelist have been removed. Original K projection and Triton cache writes
remain unchanged. No KV layout, L20 folded-twin ownership, block allocation
or PD transport changes. Ordinary Linear NZ, gating and CompressorV2 are
outside this change. No dependency upgrade is required.

## Graph-side metadata

The previous experiment submitted producers outside the model graph. Its
decode-only trace showed a separate stream but no overlap with main compute;
target/draft MQSFMLA and QLI host calls still took about 665 us per profiled
cycle. That result was not a demonstrated optimization.

The revised MRV2 path defers target MQSFMLA and QLI tiling producers while
preparing attention metadata. `ModelWithContext.forward` submits them inside
FULL graph warmup/capture, using a worker-owned side stream. Input readiness
is recorded on the main stream, followed by per-stage/per-buffer readiness
events. Attention and indexer consumers wait only at first consumption.
The forward finally block joins unused producers too. Producers, waits and
joins are captured; replay does not resubmit producers in Python.

Graph producers consume padded persistent device buffers. Runtime preparation
refreshes their contents, including actual lengths and empty/padded ranks.
MQSFMLA uses the captured token bucket for its length-row shape; QLI derives
actual lengths from device tensors rather than capture-time host values.
The next step is ordered after prior consumers without a global device
synchronization or graph-external event reset.

C2 compressor state preparation remains outside the graph, preserving actual
counts and state-update policy. Its consumer waits for a side-stream event
only when C2 was actually deferred (eager path). QSLI metadata stays with its
candidate-dependent consumer; this PR does not hoist QSLI or all preprocessing.

Eager target execution submits after preparation and joins before retirement.
Draft builders are not switched to deferred mode. The context-local executor
is restored after execution/capture. MRV1's external-event protocol remains
the default executor mode; MRV2 explicitly selects graph-side producers.
Non-A5 FULL metadata capture is rejected rather than silently misbehaving.

### Recipes reference

Source snapshot: `cann-recipes-infer`
`2225cae19d7612c1242d0815271e86e13eea2c95`.

Its `modeling_deepseek.py::forward` invokes `generate_kernel_metadata` inside
the decode callable compiled by `npugraph_ex`. Native input-ready and staged
attention/indexer events belong to this multi-stream graph. Draft has its own
compiled interface. Our revised capture boundary follows that principle
while retaining framework buffer ownership and candidate-length semantics.

## Q/W fusion

The adapter packs only its static Q/W weights after model loading. It requires
Flash MXFP8 WQB `[1280,4096]`, paired E8M0 scales `[20,4096,2]`, BF16 W weights
`[32,5120]` and scale `1/64`. Packed MXFP4 Q, scales and weights go directly
to QLI/QSLI, avoiding duplicate query quantization.

The licensed PythonDSL variant preserves adjacent-pair RoPE and BF16
weight-output rounding. It does not remove the arena wheel or modify general
Linear weights. Microtests at T=1/16/64/384/800 had exact graph replay/scales;
dequantized-Q cosine was approximately 0.999. This accepted experimental
difference is not a substitute for full-model accuracy qualification.

## Validation

Experiment P=141.61.33.22, D=141.61.33.23; proxy
`http://172.27.8.22:8966/v1`, model `deepseek-v41-a5-1005-real-1p1d`.

Both roles: DP8/TP1/EP8, MRV2, async, block128, DSpark 5. P eager/prefix on;
D FULL_DECODE_ONLY, DSpark graph, RecomputeScheduler, prefix off. Existing
shared-expert and Q/KV overlap stay on. Engram, forced EPLB and synthetic
acceptance are off. Dependencies remain torch 2.10.0+cpu, torch_npu
2.10.0.post4 and vLLM 0.30.0 from the validated image.

- 100 related unit tests pass, including deferred C2 handling, graph lifecycle,
  Q/W integration, request counts, existing ACLGraph and preprocessing.
- Six NPU tests pass: synthetic eager/FULL dependencies plus actual MQSFMLA/QLI
  graph replay with changing lengths, compression ratios 1/2 and partially or
  fully empty batches; metadata outputs equal eager reference.
- Ruff checks/format pass. Full `format.sh ci` is not qualified:
  `pre-commit` is absent. NPU tests use ACLGraph-directory `--confcutdir`
  because the outer e2e conftest requires unavailable `modelscope`.
- Original-path 1P1D: direct P/D 2/2 each, proxy arithmetic 10/10; GSM8K
  1254/1319 (95.07199%), 59.69 s, zero API errors/empty, 14 length truncations.
- Combined candidate 1P1D: proxy arithmetic 10/10; GSM8K 1256/1319
  (95.22365%), 83.82 s, zero API errors/empty/retries, 20 length truncations.
  Cold compilation and concurrent smoke requests affect this elapsed time;
  it is not a throughput comparison. The two-question difference does not
  establish an accuracy gain.
- Initial performance pair used a semaphore of 128 but the client's default
  connector limited active connections to 100. Original 8562.88 versus
  candidate 8840.27 output tokens/s, mean TPOT 4.093 versus 3.983 ms.
  This single observation is superseded by the explicit c128 repeat below.
- Explicit c128, unlimited connector, 4096 input / 1024 forced output,
  n256: candidate 9947.67 / 9975.51 / 9922.11 output tokens/s;
  TPOT 4.045 / 4.028 / 4.046 ms. All 768 requests succeeded. Paired original:
  8047.29 / 9698.25 / 9664.34 tokens/s, TPOT 4.057 / 4.140 / 4.144 ms;
  all 768 succeeded. Each pair has identical prompt SHA and fresh prefixes.
  Conservatively discard the slower first original run (cold-start-sensitive).
  The last two pairs average 9681.29 -> 9948.81 tokens/s (+2.76%), and
  4.142 -> 4.037 ms mean TPOT (-2.54%). This modest, workload-specific
  observation is not a statistically established or universal speedup.

### Captured overlap evidence

The decode profile started with 100 running requests and 11186 tokens already
generated. Start/stop both returned 200, 2.001 s apart; all 128 requests
(128-token input / 4096-token output) completed successfully.

Offline JSON analysis of rank zero identifies target model ID 33. Its side
stream 477 contains 180 MQSFMLA/QLI metadata kernels over 60 steps. Their total
duration is 3413.35 us, with 3384.00 us intersecting non-communication compute
on other streams: 99.14% interval overlap in this sample. Matmul, RMSNorm,
dynamic MX quantization, RoPE and compressor epilogue on model 33 appear among
the overlapping peers. This is device interval overlap, not an equal-sized
end-to-end speedup. Q/W fusion executes 480 times in the same target graph.

The trace has no QLI host metadata calls. It still has 120 host MQSFMLA calls
and 60 C2 preparation calls; draft/preparation is not fully captured by this
change. QSLI metadata remains on the model's consumer stream. There are also
60 Gloo all-reduces; reducing these is a separate follow-up, not part of this
candidate performance comparison.

Raw profile under the artifact root:

```text
profiles/qw_metadata_graph/
dp0_pp0_tp0_dcp0_ep0_rank0_1228_20261005191746430_ascend_pt/
ASCEND_PROFILER_OUTPUT/trace_view.json
```

`profile_decode_overlap.json`, `profile_graph_details.json` and
`profile_decode_operators.jsonl` contain counts and interval analysis.
No screenshots or image recognition were used.

GSM8K uses historical answer-format prompting, c200/max4096, temperature 0,
top_p 1, thinking off, custom scoring (not AIS Bench). Performance uses
fixed-seed token prompts separately from accuracy. Prefix-cache conditions
must match: repeated cached prompts cannot be compared to a cold baseline.

Artifacts and literal launch scripts:

```text
/mnt/share/y00882530/dsv4_1/recipes_opt_1005/cluster33/
```

The earlier 133-cluster baseline had HIXL transfer errors and incorrect PD
responses with both switches off. Its direct-D scores/profiles are historical
diagnostics, not passed 1P1D qualification of this revision.

## Remaining gates

1. Extended long-context, DSACP and GPQA before broader deployment.
2. Full repository CI; broader repeated and production-workload performance.

Keep the PR draft until its stated gates have evidence. Do not publish an
unqualified release image.
