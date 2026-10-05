# MRV2 scoped DP coordination bypass

## Purpose and boundaries

MRV1 already consults `should_skip_allreduce_across_dp_group` before its
CPU/Gloo token-count coordination. MRV2 inherited the upstream dispatcher
without entering its `skip_dp_coordination` context, so eligible decode
workers still performed one CPU collective per target step.

This change wraps the inherited `execute_model` call with that upstream
context, using the existing Ascend MRV1 eligibility predicate. The inherited
`_dummy_run` delegates to this entry point and is covered too. Normal DSpark
proposal reuses the target's `DPSyncState`; it is not separately patched.

The upstream local graph selection, bucket padding, and DP-sync state creation
remain intact. No global monkey patch, fake DP world size, new configuration
switch, dependency upgrade, or KV-cache layout change is introduced.

The predicate remains authoritative: dense models and eligible dynamic-MC2
KV consumers can skip; unsupported communication configurations retain their
collective. DBO/ubatching explicitly retains coordination even when that
predicate would otherwise allow skipping. This is not removal of all Gloo,
HCCL, scheduler, or expert-parallel communication.

This follow-up is stacked on the metadata/QW branch from PR #10. It contains
no Indexer K fusion. Keep those changes and their qualification separate.

## Validation configuration

2026-10-05, actual 1P1D on 33.22/33.23, each DP8/TP1/EP8. vLLM 0.30.0,
MRV2, async scheduling, block size 128, real DSpark acceptance, Engram off,
force-EPLB off. P eager/prefix on; D FULL_DECODE_ONLY/DSpark graph,
RecomputeScheduler on/prefix off. Metadata graph-side overlap and Q/W fusion
are enabled in both the comparison and candidate builds.

Only this task's isolated containers were restarted. The editable baseline
tree and other deployments were not modified.

### Unit and functional coverage

```bash
python -m pytest -q \
  tests/ut/worker/v2/test_dp_coordination.py \
  tests/ut/ops/test_token_dispatcher.py
```

36 passed, 11 skipped. New tests cover exception-safe context restoration,
DBO protection, ineligible configurations retaining Gloo, zero/nonzero local
tokens, DP ranks 0/7, and FULL graph bucket padding without a collective.
Ruff checks, formatting, and `git diff --check` pass. Full repository CI is
not qualified; the environment lacks `pre-commit` for `format.sh ci`.

Both direct endpoints answered 2/2 arithmetic cases correctly. Actual proxy
PD answered 10/10, including concurrent requests, with zero API errors.

GSM8K is a custom-script evaluation, not AIS Bench official scoring:
c200, max output 4096, temperature 0, top_p 1, thinking off, historical
answer-format prompt. First candidate run: 1252/1319 (94.92%), 45.18 s,
zero API errors/retries and 20 length truncations. A second complete run also
scored 1252/1319, in 39.44 s, with zero API errors/retries and 21 length
truncations. The preceding metadata/QW run was 1256/1319. These samples do
not establish accuracy equivalence; do not hide the four-question difference
or label it harmless variance without further paired testing.

### Performance samples

Two matched cold-prefix request sets: 4096 input / 1024 output tokens,
128 concurrency, 256 timed requests after 16 warmups, unlimited HTTP
connector, real acceptance. Prompt SHA and zero cached prompt tokens were
verified; all 512 requests succeeded in each variant across the two runs.

| Sample | Metadata/QW output tok/s | + DP bypass output tok/s | Metadata/QW TPOT ms | + DP bypass TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| 2 | 9975.51 | 9246.59 | 4.0278 | 3.8004 |
| 3 | 9922.11 | 9986.55 | 4.0459 | 3.8475 |
| Mean | 9948.81 | 9616.57 | 4.0368 | 3.8239 |

Measured request TPOT improved about 5.3%, while end-to-end throughput was
mixed and its two-sample mean fell about 3.3%. Do not claim a stable overall
throughput gain or discard the slower sample. These are small 1P1D workload
samples, not statistical qualification or DP32 results.

### Profile evidence

The candidate profile started with 100 running requests and 10727 tokens
already generated. Start/stop returned HTTP 200, 2.0008 s apart; all 128 load
requests completed without API errors. Offline JSON analysis of rank zero
contains 63 decode steps and zero `c10d::allreduce_` / `gloo:all_reduce` host
events. The earlier metadata/QW profile contained 60 of each over 60 steps.
This establishes removal of the targeted per-step collective in this sample,
not absence of every distributed communication operation or startup barrier.

Target model 32 still captures metadata on side stream 473: 189 MQSFMLA/QLI
kernels, 3419.35 us total, of which 3387.50 us intersects non-communication
compute on other streams (99.07%). QSLI stays on consumer stream 474. The
profile still contains host C2 preparation and draft MQ metadata calls.

Raw candidate trace:

```text
profiles/mrv2_dp_sync/
dp0_pp0_tp0_dcp0_ep0_rank0_1228_20261005194100235_ascend_pt/
ASCEND_PROFILER_OUTPUT/trace_view.json
```

## Artifacts and remaining qualification

Artifact root:

```text
/mnt/share/y00882530/dsv4_1/recipes_opt_1005/cluster33/
```

`logs/unit_tests_dp_sync_final.log`, `logs/gsm1319_dp_sync_c200_t4096_t0.log`,
`perf_dp_sync_pair_validation.json`, and `run_d_dp_sync.sh` record the
tests and literal candidate configuration. The second accuracy log adds
`_round2` before `.log`. `profile_dp_sync_control.json` and
`dp_sync_profile_analysis/` record the profile control and JSON analysis.

Before a general release: extended GPQA/long-context coverage, DP32,
additional communication configurations, full repository CI, and repeated
steady-state performance; explain the small observed GSM8K score delta with
additional paired testing. Disabled-policy and DBO behavior have unit
coverage, not end-to-end qualification in this experiment.
