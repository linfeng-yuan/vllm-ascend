# DS41 reproducible native wheel release

This directory vendors the DS41 samples and native package builder from the exact commit recorded in `UPSTREAM.json`. It is part of cann-recipes-infer so the fixes and build input travel together. The source license is preserved in LICENSE and each sample.

## Included fixes

* Apply cann/cannbot-dsl PR 181 (commit recorded in UPSTREAM.json) to QLI, QSLI and both AICPU metadata sources. Read Cube count from the input NPU; size workspaces and launch counts accordingly. Keep the fixed LI/LD metadata boundary at 36.
* Default QLI and QSLI exporters cover both has_seqused_q=False and True. After mask/ratio deduplication each exports 36 contracts (18 per flag). No external CANNBOTDSL_DS41_PROFILES override is needed for these defaults. Other operators keep upstream profiles.
* Build precompiled AICPU metadata into the wheel using the upstream DS41 builder. During packaging, detect the installed CANN HCC C++ include version rather than relying on DSL 0.7.0's hardcoded 7.3.0 paths (this image contains 14.3.0). The serving environment does not need this build-time include workaround for packaged metadata.

The distribution is `cannbot-arena-net==0.3.0+ds41fix20261021`, imported as `ops`. This follows the public builder's 0.3.0 distribution, rather than relabeling it as the old 0.1.0 cannbot-arena-net-ops. It pins the official `cannbot-dsl==0.7.0` compiler/runtime. Do not mix packages that both provide `ops` or `cannbotdsl`.

## Build

Use a matching aarch64 Python interpreter, matching official cp311/cp312 cannbot-dsl 0.7.0 wheel, and CANN environment. Required Python build dependencies include torch, torch_npu, pytest, setuptools and wheel. The full group needs NPU device visibility for QW/K host-side device-property queries; packaging does not execute the tensor kernels.

```bash
export PYTHON=/path/to/python3.11 # or python3.12
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
export CANNBOTDSL_WHEEL=/path/to/cannbot_dsl-0.7.0-cp311-cp311-manylinux_2_28_aarch64.whl
unset CANNBOTDSL_DS41_PROFILES
./run-build.sh --group ds41 --work-root /tmp/ds41-work-cp311 --output-root /tmp/ds41-output-cp311
```

Run from a local filesystem if the shared mount does not support the builder's atomic directory publication. Keep separate work/output roots for each ABI. The default build exports the complete selected DS41 group, not just the previously missing three variants.

## Validation

```bash
python tests/test_release_contracts.py
# After installing the generated wheel and matching DSL in a clean environment:
python tests/verify_vllm_static.py --report /tmp/ds41-native.json
python tests/verify_all_static.py --report /tmp/ds41-all-native.json
python tests/run_kernel_precision.py --cores 32 --report /tmp/ds41-32.json
```

On a physical 28-core NPU use --cores 28. On a 32-core NPU --cores 28 explicitly tests a real 28-worker NPU launch by overriding only the properties value consumed by the operators; it is not a physical 28-core SKU validation. Reports record the distinction. Kernel checks use the upstream independent MXFP4/TopK reference and require installed static binaries. They are not a full model or DeepSWE precision assessment.

B/T/query and KV lengths remain dynamic within exported contracts. Other static combinations, changed source, compiler/runtime or backend pipeline need their own coverage and key verification.
