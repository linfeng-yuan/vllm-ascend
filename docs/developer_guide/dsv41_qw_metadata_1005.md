# V4.1 A5 metadata overlap and Q/W + K/postscatter fusion experiment

**On hold at the user's request.** Do not merge or deploy this experiment.
The Q/W + K model retest was stopped before readiness; the original editable
baseline and original operator paths are being restored on the experiment D.
There is no complete Q/W+K GSM8K score or end-to-end performance qualification.
The bucket whitelist is an experimental workaround, not the proposed upstream
API. A future design needs a shape-polymorphic or bounded, prewarmed operator
contract that works with normal framework scheduling without model-specific
graph-bucket coupling.

The runtime compilation observed here was in a direct-D test with local
prefill/mixed batches, not a pure P-to-D disaggregated decode measurement.
`FULL_DECODE_ONLY` does not capture those eager execution paths. In the recipes
snapshot, decode inputs are padded to `batch_size_per_dp_rank` (execution
engine lines 372-393, 681), and warmup uses the fixed target width `next_n+1`
(line 830). The model worker separately selects compiled decode versus eager
prefill (line 446). That bounds the usual decode shapes but is **not** proof
that recipes never cold-compiles on other inputs or deployment settings.

## Scope and switches

Base: `e41bd634c6a753151fe191b25d81489161d3e763`.
All three features are opt-in and independently testable in `additional_config`:

```json
{
  "multistream_dsv41_metadata": false,
  "enable_dsv41_indexer_qw_fusion": true,
  "enable_dsv41_indexer_k_fusion": true
}
```

Keep async scheduling, MRV2, block size 128, target FULL_DECODE_ONLY and
DSpark full graphs enabled. Do not enable force-EPLB or synthetic acceptance
when validating correctness or reporting real-generation performance.

This experiment does not change KV allocation, page layout, split/folded
index cache ownership, L20's folded twin, or PD transfer descriptors.
General Linear NZ, gate fusion and CompressorV2 are out of scope.

## Metadata scheduling

MRV2 previously built V4.1 device metadata inline. Attention Q/KV overlap
(`multistream_dsv4_dsa_overlap`) is a different feature.

The new target-state coordinator temporarily defers the target builders,
then submits their compressor/attention/indexer metadata tasks on a separate
NPU stream. Consumers wait at their first use. Full graphs use external
events keyed by padded token count; capture and replay must use the same key.
Replay must not issue a second host wait/reset on events already reset by
the captured graph. Otherwise it deadlocks waiting for the next producer.

At the end of forward, unused producers are also joined before input buffers
can be reused by the next asynchronous scheduling step. The coordinator is
scoped to target execution, not `sample_tokens` or DSpark proposal. Builders
return to their prior mode after each build. Profiling's temporary model state
receives a fresh coordinator when KV caches are initialized again.

QSLI metadata that currently depends on candidate lengths is **not** moved
ahead of its producer in this patch. Recipes' position-derived lengths must
be proved equivalent for speculative/padded inputs before changing that
dependency. Therefore this patch does not claim to hide all metadata cost.

## Q/W integration

The packaged Q/W implementation rotates halves; the current framework uses
adjacent-pair RoPE. A direct substitution gave Q cosine approximately
0.78–0.82. The vendored DSL variant instead rotates adjacent pairs and
preserves BF16 intermediate rounding. Its distinct kernel/export name avoids
reusing the original packaged native binary. Vendor tuning environment
overrides are frozen to the tested defaults; the service switch is centralized
in AscendConfig.

After quantization weight postprocessing, the model packs only this operator's
two static weights into NZ. The original Linear weights remain unchanged.
The adapter accepts Flash geometry (5120/1280/32/128/64) and paired E8M0 scales,
uses the existing activation quantization scale algorithm, and produces
packed Q directly. No dummy BF16 Q allocation or second query quantization is
needed. Packed Q supplies QSLI's head count and logical head width.

The remaining approximately 0.001 cosine discrepancy is accepted for further
experimentation, **not** a substitute for end-to-end accuracy qualification.

## Evidence collected so far

- Metadata lifecycle on 950DT: 64 changing-input full-graph replays and 16
  eager steps, exact results, including an unused producer and buffer reuse.
- Metadata plus Q/W integration-related CPU unit tests: 73 passed, including
  producer failure cleanup, graph replay without double waits, and adapter
  weight/scale checks. Ruff check/format and whitespace checks passed.
  `bash format.sh ci` was attempted but blocked by missing `pre-commit` in
  the test environment; this is not a completed full CI run.
- Metadata-only actual model: GSM8K 1261/1319 (95.60%), API errors 0,
  length truncations 0, elapsed 93.38 s. Historical answer-format prompt,
  concurrency 200, max_tokens 4096, temperature/top_p 1, thinking false.
  This is the custom script evaluator, not official AIS Bench scoring.
- Q/W adapter NPU tests at T=1/16/64/384/800: graph replay exact; scales exact;
  dequantized Q cosine 0.99897–0.99904. This is not full model qualification.
- Q/W plus metadata GSM8K at temperature 1: 1250/1319 (94.77%), errors and
  length truncations both zero. The metadata-only run above scored 1261.
  This one stochastic comparison does not establish a regression or a gain.
- Temperature-zero comparison, same 1319 questions and request limits:
  both switches off 1248/1319 (94.62%), both on 1256/1319 (95.22%). API errors
  were zero; length truncations were 24 and 22 respectively. Eight previously
  correct questions became incorrect, 16 previously incorrect became correct,
  and 1285 final answers matched. Temperature zero does not guarantee bitwise
  determinism across dynamic batches. No GPQA qualification in this experiment.
- Metadata-only direct-D performance: fixed 4096 input / 1024 output tokens,
  128 concurrency, 256 requests, all successful; 4690.25 output tokens/s,
  mean request TPOT 15.66 ms. Q/W plus metadata: 4792.18 output tokens/s,
  mean request TPOT 15.52 ms, 256/256 successful. This approximately 2.17%
  difference is one run, not a qualified speedup or a PD throughput result.
  Output length is forced for this separate performance test, not accuracy.
  The original path (both switches off) measured 4786.92 output tokens/s and
  15.32 ms mean TPOT. Thus the combined candidate is effectively throughput
  neutral against the original path in this mixed workload, not 2.17% faster
  than the original. Its mean TPOT was slightly worse in this one comparison.
  After restarting the identical candidate, a repeat measured 4653.13 output
  tokens/s and 15.91 ms mean TPOT (256/256, no errors). No repeatable throughput
  improvement has been established; do not promote this as a speedup patch.
- A two-second service trace contains the new interleaved Q/W kernel, proving
  actual use rather than only configured use. The first window still includes
  prefill; it is not proof of decode metadata overlap. A decode-only window
  and critical-path analysis remain required before making an overlap claim.

### Decode profile finding: separate stream is not sufficient

A second capture used 128-token prompts and 4096-token outputs, starting after
404930 generated tokens with 78 requests still running. Start/stop returned
200, separated by 2.0006 seconds; all 128 requests succeeded. Rank zero's
trace contains real FULL graph model IDs and 69 C2 metadata cycles.

In that sampled trace, metadata stream 50's 276 kernels took 3641.14 us total,
approximately 52.77 us per cycle. Their time intervals did **not** intersect
AI-core/vector compute on the other streams (communication, waits and copies
excluded). The target Q/W fusion kernel is present, median 13.36 us.

The CPU-side MQSFMLA metadata calls numbered 207 with 24325.68 us total; QLI
calls numbered 138 with 21540.97 us total. Dividing by the 69 cycles gives
approximately 664.73 us per cycle across those calls, including target and
draft metadata preparation. These are profiled host call durations, not pure
CPU compute time or an unprofiled latency claim. Moving their device work to
another stream has not removed the per-step host dispatch cost.

**The requested main-graph metadata overlap and performance gain are not yet
achieved.** This experimental implementation establishes lifecycle correctness
but is not the completed optimization. Both new flags are rolled back to off
in the experimental service; the existing attention/shared-expert multistream
features remain enabled. The next design must capture compatible metadata
work in the graph's side branch, rather than artificially delaying already
finished metadata just to manufacture timeline overlap. Padded/empty ranks,
actual request/token counts and speculative rollback must remain correct;
capture-time Python constants cannot silently replace dynamic inputs.

Raw rank-zero profile:

```text
/mnt/share/y00882530/dsv4_1/recipes_opt_1005/profiles/qw_metadata/
dp0_pp0_tp0_dcp0_ep0_rank0_54460_20261005172352761_ascend_pt/
ASCEND_PROFILER_OUTPUT/trace_view.json
```

`profile_decode_overlap.json` and `profile_decode_operators.jsonl` in the stage
directory contain the interval/count analysis. No image recognition was used.

The experimental PD baseline was already producing incorrect responses before
these changes. Direct-node results must not be presented as a passed 1P1D
regression. PD routing/transfer diagnosis remains a separate prerequisite.

## K/postscatter integration

The installed `ops.indexer_prologue_k` uses BF16 matmul followed by fused
RMSNorm, adjacent-pair RoPE, MXFP4 packing and scatter. It accepts explicit
cache strides and INT64 flattened slots, and skips slot -1.

Tests used 64/128-row storage pages, padded physical page strides, nonzero
storage offsets, cross-page writes and invalid slots, at T=1/16/64/384. Split
K/scale bytes and the existing Triton-folded twin matched exactly; untouched
bytes retained their sentinel. For example, page128/T64 measured approximately
14.50 us reference versus 8.91 us fused in repeated graph microbenchmarks.

The opt-in adapter writes the existing split planes, then retains the
existing fold for L20. No layout changes are needed for this route. Static
WK NZ packing and norm-weight conversion happen after weight loading, before
KV memory sizing. Non-source layers do not prepare or invoke this adapter.
The prepared compression-aware INT64 flat slots are reused, not reconstructed
using the logical block size. Empty batches bypass the kernel. The installed
arena implementation is required; no wheel is removed or dependency upgraded.
The large-offset test also matched the reference
through physical byte offset 2147484544, while page-zero sentinels remained
unchanged (no 32-bit wrap). Real-model accuracy and end-to-end performance
still require validation before promotion.

Integration follow-up: 78 related CPU unit tests pass. Eight A5 NPU tests
also pass with three changing-input graph replays per case (page64/128,
T=1/16/64/384), comparing the entire backing allocation and the folded twin
byte for byte. The NPU test was executed standalone outside the repository's
shared e2e conftest; this is not a full e2e CI run. The adapter microbenchmark
measured page128 T1/T16/T64/T384 reference 7.15/10.19/14.59/22.99 us versus
fused 4.64/4.64/6.62/14.75 us. Differences from earlier timings include the
measurement run and current wrapper; these are local kernel-chain numbers.

### Avoid request-time K compilation

The first unrestricted K integration exposed a cold-start regression in real
mixed batching: `py-spy` showed workers compiling
`ops.indexer_prologue_k._get_fused_pipeline_callable` through bisheng in the
request path as token counts changed. That GSM8K attempt was intentionally
stopped after 109 records; it is a partial diagnostic, **not an accuracy
score**. Logs are preserved as `gsm1319_qw_k_cold_partial.log` and
`d_qw_k_cold_initial.log` in the stage directory.

The model adapter now admits only token counts from
`compilation_config.cudagraph_capture_sizes`. All other shapes use the
original K projection/RMSNorm/RoPE/Triton writer, without a new DSL compile.
This retains K fusion in the warmed full-decode graph buckets and keeps
arbitrary prefill/mixed shapes on the proven path. No extra synchronization,
tensor-value host reads or cache layout changes are introduced. With no graph
buckets configured, this integration leaves the original writer active.
80 related unit tests pass, including unbucketed-shape fallback and empty
bucket behavior. First-time graph-bucket compilation still adds startup time;
it must not be advertised as eliminated or counted as steady-state latency.

## How recipes captures metadata

Source snapshot: `cann-recipes-infer` commit
`2225cae19d7612c1242d0815271e86e13eea2c95`.
This is a source-level explanation of its configured decode path, not a claim
that every event in a separately supplied trace has been matched to this SHA.

- `executor/core/model_worker/model_worker.py:46`: `main_decode` calls
  `self.forward`; `compile_model` at line 483 selects that interface.
- `executor/utils/graph_utils.py:66`: `npugraph_ex` compiles that callable via
  `cache_compile` or `torch.compile(..., fullgraph=True, backend="npugraph_ex")`.
- `models/deepseek_v4_1/models/modeling_deepseek.py:3149`: **inside forward**,
  `generate_kernel_metadata` runs before the model layers. It is not in the
  host-only `preprocess_model_inputs` block at line 3124.
- The generator at line 2994 records input readiness, switches to the native
  metadata stream, waits for inputs and produces MQSFMLA metadata. Event 1
  makes that result available to the first attention consumer while indexer
  metadata continues. Event 2 releases QLI/QSLI consumers.
- `executor/utils/stream_utils.py:50` uses native `torch.npu.stream` for this
  mode, with native event record/wait. These operations belong to the compiled
  forward's multi-stream graph, rather than a separate host launch per decode
  iteration. Eager execution uses the same stream organization without replay.
- Draft decode has a separate compiled interface:
  `executor/core/model_worker/dspark_worker.py:129` compiles
  `forward_spec_decode_graph`; target and draft are not one undifferentiated
  graph.

Therefore the answer is **both an auxiliary stream and graph-side capture /
replay**, when multi-stream + npugraph_ex decode is enabled. Tensor metadata
generation and dependencies are captured; this does not mean the scheduler,
Python input preparation or every host action is captured. Our current
experimental external producer does not replicate that capture boundary.
Moving the producer inside requires fixed-address device inputs and correct
dynamic actual lengths, padding, async buffer lifetime and draft rollback.
Recipes' position-derived QSLI candidate lengths must first be proved
equivalent to the framework candidate lengths, not copied unconditionally.

## Review and reproduction

Implementation branch: `perf/1005-dsv41-qw-overlap`, based on the unchanged
`1005_950DT_vllm0300_rebase_main` fork branch. This is an experimental review,
not a release image. The context-local executor is scoped and restored using
`ContextVar`; it must not leak into a subsequent draft proposal. Review that
lifecycle together with the graph capture/replay and exception paths.

NPU tests ran on 133.107 in `ylf_dsv41_real_1005_133107`, using an isolated
source copy, preserving the original editable installation and dependencies.
Artifacts and literal launch scripts are in:

```text
/mnt/share/y00882530/dsv4_1/recipes_opt_1005/
```

The raw correctness/performance logs and profiler control records are kept
there. Launch scripts preserve async scheduling, real routing/acceptance,
MRV2, DSpark graph mode, RecomputeScheduler and block128. Engram is disabled.
Use `run_d_optimization_baseline.sh` for both switches off, or
`run_d_qw_metadata_candidate.sh` for both switches on. Never run both services
on the same cards simultaneously. Performance parameters and prompt hash are
recorded by `benchmark_optimization_ab.py`; scoring uses `smoke_gsm8k.py`.

Unit-test command in the qualified dependency environment:

```bash
pytest -q \
  tests/ut/attention/test_dsa_v41.py \
  tests/ut/models/test_deepseek_v41_preprocess.py \
  tests/ut/worker/test_device_metadata.py \
  tests/ut/worker/v2/test_device_metadata.py \
  tests/ut/ops/test_dsv41_indexer_qw.py \
  tests/ut/ops/test_dsv41_indexer_k.py
```

K integration can be tested independently with
`tests/e2e/pull_request/one_card/test_dsv41_indexer_k_fusion.py` on the qualified
A5 arena/DSL stack. `run_d_qw_k_candidate.sh` enables Q/W and K with metadata
off, so the operator comparison is not confounded by the external producer.

## Remaining acceptance gates

1. Full-model Q/W + metadata accuracy and identical-request performance A/B.
2. Service profiling proving producer/main-graph overlap, including async
   scheduling; distinguish total operator time from exposed critical-path time.
3. Long-context, shape transitions, empty DP ranks and DSACP coverage.
4. Restore correct PD baseline and repeat 1P1D accuracy/performance.
5. Submit a reviewable fork PR; do not publish an unqualified release image.
