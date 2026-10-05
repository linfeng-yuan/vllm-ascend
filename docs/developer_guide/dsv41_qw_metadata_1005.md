# V4.1 A5 metadata overlap and Q/W fusion experiment

## Scope and switches

Base: `e41bd634c6a753151fe191b25d81489161d3e763`.
Both features are opt-in and independently testable in `additional_config`:

```json
{
  "multistream_dsv41_metadata": true,
  "enable_dsv41_indexer_qw_fusion": true
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
- A two-second service trace contains the new interleaved Q/W kernel, proving
  actual use rather than only configured use. The first window still includes
  prefill; it is not proof of decode metadata overlap. A decode-only window
  and critical-path analysis remain required before making an overlap claim.

The experimental PD baseline was already producing incorrect responses before
these changes. Direct-node results must not be presented as a passed 1P1D
regression. PD routing/transfer diagnosis remains a separate prerequisite.

## K/postscatter feasibility

The installed `ops.indexer_prologue_k` uses BF16 matmul followed by fused
RMSNorm, adjacent-pair RoPE, MXFP4 packing and scatter. It accepts explicit
cache strides and INT64 flattened slots, and skips slot -1.

Tests used 64/128-row storage pages, padded physical page strides, nonzero
storage offsets, cross-page writes and invalid slots, at T=1/16/64/384. Split
K/scale bytes and the existing Triton-folded twin matched exactly; untouched
bytes retained their sentinel. For example, page128/T64 measured approximately
14.50 us reference versus 8.91 us fused in repeated graph microbenchmarks.

Candidate integration: write the existing split planes, then retain the
existing fold for L20. No layout changes are needed for this route. It is not
enabled in the model yet. The large-offset test also matched the reference
through physical byte offset 2147484544, while page-zero sentinels remained
unchanged (no 32-bit wrap). Real-model accuracy and end-to-end performance
still require validation before promotion.

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
  tests/ut/ops/test_dsv41_indexer_qw.py
```

## Remaining acceptance gates

1. Full-model Q/W + metadata accuracy and identical-request performance A/B.
2. Service profiling proving producer/main-graph overlap, including async
   scheduling; distinguish total operator time from exposed critical-path time.
3. Long-context, shape transitions, empty DP ranks and DSACP coverage.
4. Restore correct PD baseline and repeat 1P1D accuracy/performance.
5. Submit a reviewable fork PR; do not publish an unqualified release image.
