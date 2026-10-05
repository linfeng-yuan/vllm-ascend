# DeepSeek V4.1 Engram MRV2 integration review

## Scope and provenance

This branch is based on `1005_950DT_vllm0300_rebase_main` at `ec5b8479f`
(PR #11 already merged). It cherry-picks PR #9 commits `905a11f32` and
`9b3f169ef` with their original authors and `-x` provenance, then adds the
registered-wrapper integration fix. It does not include PR #10 or PR #13.
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
   optimization replaces the other's lifecycle handling.

## Regression coverage

- Real registry wrapper selection, model-state factory, forwarding of both
  graph hooks, graph mode, dummy mode, history, graph buckets and retirement.
- Single-call preparation; updated test fixtures for current MRV2 interfaces.
- NPU lookback gather with permuted requests, newest-first history, short and
  empty prefixes, inactive rows and buffer reuse by a shorter subsequent batch.
- Existing one-card graph replay and two-card DP-shared-table graph tests.

## Validation status

The four missing wrapper contracts fail before the fix. After the fix, 91
targeted unit tests pass across `test_engram_multistream.py`,
`test_engram_v2_model_state.py`, `test_engram_registered_v2_contract.py` and
`test_model_runner_v2.py`. Ruff and `git diff --check` pass. Full `format.sh ci`
could not run because the test container does not have `pre-commit`; no runtime
dependencies were upgraded to bypass that limitation.

On Ascend hardware, the lookback gather and one-card UVA graph replay tests
pass (2 tests), as does the two-card DP-shared-table graph replay test (1 test).
These hardware tests ran with the existing PR #10 stack plus the same wrapper
fix; hash/lookup graph tests use a small synthetic table and a stub hash. They
do not substitute for full-weight end-to-end accuracy or overlap measurements.
The standalone lookback test imports the upstream speculator before model
state modules, matching the runner startup import order.

Full-model accuracy and performance qualification is tracked in the PR. The
previous PR #9 switch-on/off profile was collected before this fix: hash and
UVA lookup stayed on the same stream, with zero measured compute overlap.
Those results must not be presented as evidence that Engram overlap works.
Likewise, previous GSM8K scores do not qualify the repaired history path.

For an overlap A/B comparison, both sides must contain this correctness fix;
only `multistream_engram_overlap` should differ. Keep CPU offload,
`dp_shared_memory`, MRV2, async scheduling, DSpark graph mode and block size 128
identical. Use real routing and acceptance, not synthetic acceptance or forced
EPLB. If testing with PR #10's metadata/QW stack, report that explicitly rather
than describing it as the clean PR branch.
