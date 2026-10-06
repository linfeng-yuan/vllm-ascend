# DeepSeek V4.1 Engram MRV2 integration review

## Scope and provenance

This branch now includes `1005_950DT_vllm0300_rebase_main` at `32284fe0c`
(PR #10 and #11 merged) and the author's refreshed PR #9 head `5b1936cc0`.
The original cherry-picks `905a11f32` and `9b3f169ef` retain their authors and
`-x` provenance. Merge `b4764709d` brings in the author's latest commits without
rewriting the published PR #14 branch, retaining the wrapper integration fix.
PR #13 is not included.
No KV-cache layout, native operator, dependency version, or Indexer K changes
are introduced by the integration fix.

## Findings and fixes

1. The actual registry selects `AscendDeepseekV41ForCausalLM` in `vl_model.py`,
   including for text-only requests. PR #9 added the MRV2 model-state hook only
   to the inner language model. The runner therefore selected the default
   state instead of `AscendDeepseekV41ModelState`. Forward `get_model_state_cls`
   through the registered wrapper and test the real registry-to-factory path.
2. The wrapper also omitted `prime_engram_v2_graph_inputs` and the preparation
   keywords `cg_mode` and `force_dummy`. Forward these unchanged so graph
   buckets, dummy preparation, runtime graph mode and event handling reach
   the inner implementation.
3. This was not merely a performance issue. MRV2 disables the historical slot
   cache; the dedicated state gathers request-specific lookback tokens. With
   the default state this history is absent, and missing history is padded.
   Cross-chunk Engram hashes may therefore differ. A plausible short answer or
   aggregate GSM8K score does not validate this history path.
4. On the newer 1005 base, the default state already prepares Engram inputs.
   Directly porting PR #9's subclass causes duplicate preparation via `super`.
   Introduce single overridable real/dummy preparation hooks: preserve default
   behavior for other models and gather history exactly once for V4.1.
5. Resolve the runner cherry-pick conflict by retaining both PR #11's scoped
   DP-coordination bypass and PR #9's `try/finally` lookup retirement. Neither
   optimization replaces the other's lifecycle handling. The refreshed merge
   also preserves PR #10's metadata activation, with lookup retirement nested
   inside both contexts as in the author's latest runner implementation.

## Regression coverage

- Real registry wrapper selection, model-state factory, forwarding of both
  graph hooks, graph mode, dummy mode, history, graph buckets and retirement.
- Single-call preparation; updated test fixtures for current MRV2 interfaces.
- NPU lookback gather with permuted requests, newest-first history, short and
  empty prefixes, inactive rows and buffer reuse by a shorter subsequent batch.
- Existing one-card graph replay and two-card DP-shared-table graph tests.

## Validation status

Refreshed runtime code: `b4764709d`, based on latest 1005 `32284fe0c`
plus author's PR #9 `5b1936cc0`. Both P/D use identical repaired source.

- 140 targeted unit tests pass, including actual registry/state selection,
  single preparation, metadata activation and scoped DP coordination.
- NPU lookback gather, one-card UVA replay and two-card DP shared-table graph
  replay pass (3 tests). Small-table graph tests use a stub hash.
- Local Ruff and git diff whitespace checks pass. Full format CI remains
  unavailable because the container lacks pre-commit; dependencies unchanged.

### Full-weight 1P1D accuracy

P=133.108, D=133.110, DP8/TP1/EP8 each. Both have Engram CPU offload and DP
shared memory enabled, MRV2, async scheduling, block128 and DSpark5.
P remains eager with Engram overlap OFF and prefix caching ON. D uses
FULL_DECODE_ONLY with graph-enabled DSpark, RecomputeScheduler and prefix
caching OFF. Only D's `multistream_engram_overlap` differs between A/B.
Merged #10 metadata/QW and #11 coordination optimizations are present.
No synthetic acceptance, forced EPLB, dependency or KV-layout changes.

Both variants pass all 14 direct-P/direct-D/proxy/concurrent smoke requests.

| D overlap | GSM8K correct/1319 | Score | Duration | API/empty/retries | Length |
| --- | --- | --- | --- | --- | --- |
| OFF | 1281/1319 | 97.119% | 26.28 s | 0/0/0 | 0 |
| ON | 1278/1319 | 96.892% | 37.49 s | 0/0/0 | 2 |
| ON repeat | 1281/1319 | 97.119% | 35.38 s | 0/0/0 | 2 |

Custom-script evaluation, not AIS Bench: concurrency200, max_tokens4096,
temperature0, top_p1, thinking=false, historical answer:$ANSWER prompt.
A repeat reaches the OFF score, so the first three-question difference is
not a demonstrated stable regression. This is not proof of numerical
equivalence: GSM truncations remain, and long-context/multimodal coverage
is not included. GPQA follow-up is recorded below. GSM duration is not a controlled performance measurement
because output lengths and cache state differ.

### GPQA Diamond follow-up and output-quality audit

Same running 1P1D (`b4764709d`), Engram ON, D overlap ON. No service restart,
code change or profiler capture occurred during the run. AIS Bench
`ais-bench-benchmark==3.1.20260630` (source `1e7cb37e3`) ran on 141.62 in
CPU-only container `ylf_gpqa_pr14_1006`; dependencies were not upgraded.

Historical GPQA Diamond prompt/evaluator, non-streaming, thinking=true,
temperature=1, top_p=1, max_out_len=128000, retry=2, no warmups. Configured
concurrency200; the dataset contains198 requests, so actual concurrency
cannot reach200. This is not the GSM8K no-thinking configuration.

- AIS Bench: **180/198 = 90.91%**, extraction100%, all198 unique records
  successful, inference406.33s.
- All198 outputs contain a nonempty final answer and end with `Answer: A/B/C/D`.
  No U+FFFD replacement characters or unexpected control characters.
- Heuristic full-output scan finds no long consecutive token loop (period
  1..64 lexical tokens, at least3 copies and96 tokens in the repeated span),
  no character run of16 or more, and no long line repeated4 or more times.
  Final-answer sections have no16-token phrase repeated4 or more times.
- A looser repeated16-token-phrase check flags52 reasoning sections, commonly
  repeated question excerpts, chemical names or formulas. This is a review
  candidate count, **not52 corrupt outputs**. Long reasoning does revisit
  hypotheses and can be redundant; do not describe all outputs as concise
  or entirely repetition-free.
- Manually reviewed all18 wrong final answers with their questions and
  beginning/middle/end samples from the10 longest outputs by character count.
  Reviewed answers remain on-topic and readable; wrong options are not caused
  by empty/garbled/unparseable output. This is not a domain-expert validation
  of every reasoning step or proof of no numerical/semantic regression.
- Offline retokenization with the served model's tokenizer: raw output median
  1780.5 tokens, maximum44869 (id147), none reaches128000. All final answers
  are complete. AIS accuracy artifacts do **not** retain API finish_reason,
  so do not claim a measured zero `finish_reason=length` count.

No controlled same-config GPQA overlap-OFF pair was run. Earlier169/198 data
used Engram OFF and is not an apples-to-apples comparison. This run supports
correct serving and no obvious output collapse on this dataset, not a blanket
"no degradation" claim. Preserve it as the pre-graph-producer baseline;
repeat scoring and output-quality checks after the scheduling redesign.

Artifacts on 141.62:
`/mnt/share/y00882530/dsv4_1/pr14_refreshed_1005/gpqa_c200/outputs/20261006_003926/`.
Local copy: `deployment/pr14_refreshed_1005/gpqa_c200_results/` in the parent
workspace, including original predictions/results, `quality_audit.json`,
`retokenized_lengths.json` and human-review text exports. The audit scripts
are in the parent workspace's `deployment/pr14_refreshed_1005/` directory.

### Timed A/B, profiler disabled

Fixed-seed random valid-text-token prompts; 4096 input, exactly1024 output,
ignore_eos, concurrency128, 256 timed requests after16 warmups per round.
Each pair uses identical prompt hashes and independent cache_salt.
All four rounds complete256/256, zero errors, zero reported cached tokens.
Acceptance and routing are real; this fixed-output microbenchmark does not
represent arbitrary user workloads.

| Round | OFF output tokens/s | ON output tokens/s | Change | OFF mean TPOT | ON mean TPOT |
| --- | --- | --- | --- | --- | --- |
| 1 | 9112.01 | 10217.42 | +12.13% | 3.879 ms | 4.022 ms |
| 2 | 10260.69 | 10211.59 | -0.48% | 3.936 ms | 3.999 ms |

These results do not establish a stable speedup. OFF round1 has larger TTFT
(8.264 s versus6.594 s in round2), while ON rounds are6.579/6.563 s.
The ordering/warmup variation is a confound, not evidence of an Engram
benefit. Mean TPOT worsens by3.70%/1.62% in the corresponding ON rounds.
Do not advertise the aggregate throughput increase as an optimization win.

### Profile: auxiliary stream works, overlap does not

Each variant has a separate2-second capture, started at100 running requests,
then all128 profile requests drain successfully. Time-axis JSON was analyzed;
profiler-on throughput is not used above. Evidence is DP0 for this workload.

- OFF:62 hash calls and124 UVA lookups, stream61; compute overlap0%.
- ON:61 hash calls and122 UVA lookups, auxiliary stream44; compute overlap0%.
- Neither sample contains Gloo allreduce events.
- ON producers have model id4294967295 (outside the graph), while the target
  replay has model id32. In all61 paired samples both lookups finish before
  MODEL_EXECUTE; the gap from last lookup completion to graph launch is
  at least176.75 us, median206.5 us. Hash start to graph launch median623 us.

The wrapper fix makes the PR #9 auxiliary producer reachable, but it still
submits hash/lookup from Python before model replay. Moving work to another
stream alone does not ensure overlap: in this observed execution the producer
is already finished before target compute starts. Profiler adds host overhead,
so these gap sizes are not unprofiled latency estimates.

A future optimization needs a separately validated producer/capture scheduling
design (for example graph-side production with stable device inputs and
consumer-local dependencies), retaining request lookback, dummy steps and
async buffer-reuse safety. That redesign was not implemented atb4764709d;
the graph-side follow-up below supersedes this implementation status.
Keep PR #14 Draft; do not enable Engram overlap by default or call it a
performance-qualified release based on this test.

### Artifacts

Remote stage:
`/mnt/share/y00882530/dsv4_1/pr14_refreshed_1005`

- `run_p.sh`, `run_d_off.sh`, `run_d_on.sh`, `run_tests.sh`.
- `logs/ut.log`, `logs/smoke_off.log`, `logs/smoke_on.log`.
- `logs/gsm8k_off.log`, `logs/gsm8k_on.log`, `logs/gsm8k_on_repeat.log`.
- `perf_{off,on}_warm{1,2}/summary.json`, request records and metrics.
- `profile_off_analysis.json`, `profile_on_analysis.json`,
  `on_graph_gap.json`, `profiles/{off,on}/dp0_*_ascend_pt`.

## Graph-side producer experiment, October 6

This follow-up replaces graph-external production only for MRV2 FULL replay,
slotless hashing, PP1, no sequence parallelism, and local/DP-shared tables
with table TP/DP size1. The existing overlap setting still controls activation.
Other layouts and eager execution retain their previous preparation path.
KV layout, operator packages and dependencies are unchanged.

Fixed-address query coordinates, history and a device valid-token count are
staged before replay. Hashing and UVA lookup run on an auxiliary stream inside
the same capture as the model. Main-stream waits occur at the first mask
consumer and each lookup consumer, followed by a graph-local stream join.
The device count handles idle DP steps and padded buckets without baking
capture-time dummy values into replay. The coordinate buffer is separate
from FIA's extra padding row. No graph-external producer or ExternalEvent
submission is needed on this guarded route.

Validation uses the same 133.108=P / 133.110=D 1P1D configuration above:
Engram ON, D overlap ON, P eager/overlap OFF, async scheduling, MRV2,
block128, RecomputeScheduler and DSpark graph decoding, real acceptance and
routing. Both nodes have byte-identical production Python files from
`/vllm-workspace/vllm-ascend-pr14-graphproducer-1006`.

### Functional and quality results

- 153 focused unit tests pass. Three NPU tests pass, including actual UVA
  lookup across graph buckets96/192, changed active lengths, repeated idle
  steps and buffer reuse. The producer scheduling test stubs hashing;
  full-model runs below exercise the real hash implementation.
- Exact-answer direct P/D/proxy smoke:14/14.
- GSM8K:1282/1319 (97.19%),40.96s, zero API errors/empty outputs/retries,
  two length-truncated outputs at4096 tokens. This is the custom-script
  evaluation, not AIS Bench's official GSM8K evaluator.
- AIS Bench GPQA Diamond:180/198 (90.91%),198 unique successes,100% answer
  extraction,558.33s. Identical historical prompt/settings: concurrency200,
  thinking=true, temperature1, top_p1, maximum128000 output tokens.
  The complete isolated D counter delta is198 stop, zero length/error/abort/
  repetition. No other requests or profiling overlapped this evaluation.
- Full output scan finds no replacement characters, unexpected controls,
  or long consecutive lexical loops. All198 final answers are nonempty and
  terminate in the requested answer format. Two new strict heuristic flags
  are question-option repetition in reasoning (id7) and table-alignment
  spaces (id186), not corrupt bytes or endless repeated output.
- Reviewed all18 wrong final answers with questions, and beginning/middle/
  ending excerpts of the10 longest outputs. Reasoning can revisit hypotheses
  excessively. In particular id147 produces69380 tokens and eventually
  ignores the fluorine-compound constraint, yielding a wrong answer; its
  baseline answer was correct at44869 tokens. Do not call this flawless
  reasoning or infer absence of cognitive degradation from the total score.
- Versus the matched baseline, six answers change correct-to-wrong and six
  wrong-to-correct. Median output length decreases1780.5 to1597 tokens but
  maximum increases44869 to69380. Temperature1 and a single paired run do
  not establish numerical equivalence or attribute these changes to a race.

GPQA artifacts on141.62:
`/mnt/share/y00882530/dsv4_1/pr14_refreshed_1005/gpqa_c200/graphproducer_outputs/20261006_012509/`.
The parent workspace has `deployment/pr14_refreshed_1005/gpqa_graphproducer_results/`,
including predictions, audit, token lengths, matched comparison and review
exports. `graphproducer_gpqa_metrics.json` retains finish-counter deltas.

### Fixed-length timing, separate from profiler

Same prompts/hashes and workload as the previous A/B, zero cached tokens,
256/256 successes per round. These are historical paired comparisons,
not a randomized interleaved experiment.

| Round | Old ON output tokens/s | Graph producer output tokens/s | Old ON mean TPOT | Graph producer mean TPOT |
| --- | --- | --- | --- | --- |
| 1 | 10217.42 | 9504.48 | 4.022 ms | 3.806 ms |
| 2 | 10211.59 | 10388.15 | 3.999 ms | 3.820 ms |

Mean TPOT improves5.37%/4.49%, but aggregate output throughput changes
-6.98%/+1.73%. Candidate TTFT is7.748/6.587s; the first round is again slower
in prefill/startup. Do not claim a stable end-to-end throughput improvement.
Compared with the warmed OFF round2, candidate round2 TPOT improves2.94%.

### Time-axis evidence: real overlap, not a99% network speedup

A separate2-second capture starts at100 running requests and10367 generated
tokens; all128 profiling requests complete without errors. Both profile
control calls return200. The same JSON interval analysis used for OFF/ON
excludes communication, metadata, waits and copies from peer compute.

DP0 sample:

| Producer | Calls | Stream | Model id | Median duration | Time overlapping other compute |
| --- | --- | --- | --- | --- | --- |
| `_hash_ids_kernel` | 63 | 569 | 32 | 26.477 us | 99.24% |
| `_engram_host_uva_gather_dequant_kernel` | 126 | 569 | 32 | 42.749 us | 99.82% |

Peer operations include `aclnnQuantMatmulV5_QuantBatchMatmulV3_QuantBatchMatmulV3`
and `InplacePartialRotaryMul` on stream571, also model32. Unlike the old ON
producer's model id4294967295 and zero overlap, these operations execute
inside the target graph and overlap real model compute. No Gloo events were
found in this sample. The percentages describe observed producer duration
overlap on DP0, not all ranks/workloads, eliminated critical-path latency,
or an end-to-end99% speedup. The timing table is the performance evidence.

Evidence under the remote stage:

- `perf_graphproducer_warm{1,2}/summary.json`, full request and metric records.
- `profile_graphproducer_control.json`, `profile_graphproducer_analysis.json`.
- `profiles/graphproducer/dp0_pp0_tp0_dcp0_ep0_rank0_1236_20261006013614317_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`.

Keep the PR in Draft pending review of the guarded graph scheduling change
and output-quality caveats. Do not advertise stable whole-service throughput
gain or blanket precision equivalence from these runs.

### Startup incident

The first D launch failed before DP7 capture: Mooncake/ADXL could not read
the device EID (`dcmiv2_get_eid_list_by_urma_dev_index`, return-8005), leaving
other ranks waiting. Preserve
`logs/d_graphproducer_attempt1_adxl_eid_failure.log`. Restarting only the
isolated D container with the same configuration succeeded. The underlying
ADXL failure was not diagnosed or claimed fixed; no driver changes were made.
