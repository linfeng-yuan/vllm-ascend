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
async buffer-reuse safety. That redesign is not implemented in this refresh.
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
