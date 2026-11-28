# DeepSeek V4.1 Flash on Ascend 950 — r148 DP16/EP16 Decode

This reproduces the BS80 Decode measurement. One 8-card Prefill node fills the KV cache. Two 8-card Decode nodes form a single DP16/EP16 group. The benchmark measures steady Decode after Prefill and KV transfer finish.

## 1. Obtain the matching inputs

- Three ARM64 hosts with eight Ascend 950 cards each, compatible driver and firmware, and a working HCCL/Mooncake data network. The launch script uses Docker in privileged mode with host networking and needs access to the Ascend device nodes on each host.
- The A5 Ubuntu ARM64 r148 base image with CANN 9.3.0. Check that your build host can pull the SWR image in section 2. The expected image ID is `sha256:9722b0304456438355302be279e95e1f2a5e0aa08401d16d19ba35a6a4821a3d`.
- [Download r148-dp16ep16-customer-inputs.zip](https://media.githubusercontent.com/media/linfeng-yuan/vllm-ascend/94a37149d29a38acaf67aebfe8fe72bd0da8d4eb/examples/dsv41_r148_dp16ep16/inputs/r148-dp16ep16-customer-inputs.zip) from the [`inputs/` directory](inputs/). Its `runtime/` directory contains `runtime.tar.gz` and `extras.tar.gz`, checked against [runtime.sha256](runtime.sha256). They provide the integrated Ascend source at `60ece1a4bc09480e014a28a58e6ca4ea7bd22abd`, the compiled native extension, Engram/HIXL, DeepGEMM, Fastokens, the installed `cannbot_dsl 0.8.2` and `cannbot_arena_net 0.3.0` with attention epilogue AOT artifacts, and the installed MegaMHC vendor. The base vLLM revision is `4ba787ec9959e566d3b587a796dff07673324da5`.
- The full DeepSeek V4.1 Flash checkpoint. Mount its directory as `MODEL_PATH`. [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)
- The `prompt-128k.jsonl` fixture from the same ZIP. Its only line contains the original benchmark's `input` messages and has SHA-256 `47b8e092d5022551477cff7351ee150e2d086e35d2f49a3d06050356d9e77246`. The serving tokenizer and chat template produce 131,102 tokens, including 131,072 cacheable tokens.

### Download and extract the ZIP

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch repro/dsv41-r148-dp16ep16 https://github.com/linfeng-yuan/vllm-ascend.git
cd vllm-ascend
curl -fL -o r148-dp16ep16-customer-inputs.zip \
  'https://media.githubusercontent.com/media/linfeng-yuan/vllm-ascend/94a37149d29a38acaf67aebfe8fe72bd0da8d4eb/examples/dsv41_r148_dp16ep16/inputs/r148-dp16ep16-customer-inputs.zip'
echo '1226777218db4af3dd605b84a06378d707313b3779cb383657b7f13f252440cf  r148-dp16ep16-customer-inputs.zip' | sha256sum -c -
export BUNDLE_DIR="$PWD/r148-inputs"
unzip r148-dp16ep16-customer-inputs.zip -d "$BUNDLE_DIR"
```

From the repository root, Git LFS can also download the ZIP with `git lfs pull --include="examples/dsv41_r148_dp16ep16/inputs/r148-dp16ep16-customer-inputs.zip" origin`. Verify the same SHA-256 before extracting it.

### Standalone CANNBot wheels

The ZIP's runtime archives already include both installed packages. For a standalone installation, get the original ARM64 Python 3.12 wheels from the [wheel directory](wheels/) or use the downloads below.

| Wheel | SHA-256 |
| --- | --- |
| [CANNBot DSL 0.8.2](https://media.githubusercontent.com/media/linfeng-yuan/vllm-ascend/7606dca39815b8ea728d15760c5a7d09cbe0da4d/examples/dsv41_r148_dp16ep16/wheels/cannbot_dsl-0.8.2-cp312-cp312-manylinux_2_17_aarch64.whl) | `d5c8e20185f0697e5e75b76a8f9617496611548995800f3f3d72cc0a613064df` |
| [CANNBot Arena Net 0.3.0 with attention epilogue](https://media.githubusercontent.com/media/linfeng-yuan/vllm-ascend/7606dca39815b8ea728d15760c5a7d09cbe0da4d/examples/dsv41_r148_dp16ep16/wheels/cannbot_arena_net-0.3.0-cp312-cp312-linux_aarch64.whl) | `78f7dac328a6539ce3f403260731b9996a4802b2bd4ba79ee1cbe02636789589` |

```bash
git lfs pull --include="examples/dsv41_r148_dp16ep16/wheels/*.whl" origin
```

## 2. Prepare the r148 container image

Run this once on an ARM64 host. After loading the base image, the build assembles both archives offline, validates the installed operators, and uses the measured attention epilogue threshold of 384.

```bash
cd examples/dsv41_r148_dp16ep16
export BASE_IMAGE=swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:nightly-ci-main-a5-cann9.3.0_20261113161300-20261123-r148
docker pull --platform linux/arm64 "$BASE_IMAGE"
docker image inspect --format '{{.Id}}' "$BASE_IMAGE"
export IMAGE=dsv41-r148:dp16ep16-bs80
bash build-image.sh "$BUNDLE_DIR/runtime" "$IMAGE"
docker run --rm --network none --entrypoint bash "$IMAGE" /usr/local/bin/dsv41-check-runtime
```

If your build host cannot pull from SWR, obtain a `docker save` archive of the same ARM64 base image through your delivery contact. Load it with `docker load -i /path/to/r148-base.tar`, set `BASE_IMAGE` to the loaded tag, and continue from the image ID check.

Distribute the resulting image to the other nodes through your registry, or with `docker save` and `docker load`. Pin the same image ID on all three nodes. If you were supplied the original prebuilt r148 benchmark image, its epilogue-384 image ID is `sha256:94f03289b6b435dac055de025c4c78747754bac05924cc63436c48c14a184da9`. It already carries the MegaMHC vendor package.

## 3. Build MegaMHC from source, if required

The packaged runtime already provides the historical vendor build. To build it from source, get the pinned source inputs listed in [mega-mhc-sources.sha256](mega-mhc-sources.sha256). The operator is `mega_mhc` from CANN ops-transformer at `224e4b1362c3fd0585b4bc7f4ef38bc1f8b67484`, with the recorded build adjustments and dependency archives. Keep the r148 CANN compiler and Python environment for this build.

Run the script inside the **base r148 image** with the source inputs mounted read-only. Pass a new build directory on a writable filesystem. The script builds for `ascend950` and extracts the vendor payload to the build directory.

```bash
MHC_INPUT="$BUNDLE_DIR/mega-mhc-sources"
MHC_OUT=/path/to/shared/mega-mhc-builds
mkdir -p "$MHC_OUT"
docker run --rm --network none -v "$PWD:/repro:ro" -v "$MHC_INPUT:/sources:ro" \
  -v "$MHC_OUT:/out" --entrypoint bash "$BASE_IMAGE" \
  /repro/rebuild-mega-mhc.sh /sources /out/ascend950
export MHC_VENDOR="$MHC_OUT/ascend950/package/packages/vendors/dsv41_mhc20_transformer"
```

Expose that directory on every serving host before creating containers. With `MHC_VENDOR` set, `create-container.sh` mounts the rebuilt vendor at the original vendor path. Run the packaged vendor baseline first with `MHC_VENDOR` unset, then repeat with the rebuilt vendor to check performance parity.

## 4. Start P, D0 and D1

On each host, use the same checkout and image, set the three actual **data-network** IPv4 addresses and the local HCCL interface, and mount the required model and run paths. Use a new shared run directory per attempt. The three nodes must reach one another on ports 8000–8007, the DP rendezvous at 12321, and the KV-transfer ports starting at 36010.

```bash
export P_IP=192.0.2.10 D0_IP=192.0.2.11 D1_IP=192.0.2.12
export NPU_NIC_NAME='<local data interface>'
export IMAGE=dsv41-r148:dp16ep16-bs80  # Set the tag loaded on this host.
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export RUN_DIR=/shared/dsv41-r148/run-01
export CONTAINER=dsv41-r148-bs80
bash create-container.sh
```

Set `CONTAINER_IPC=private` on D1 before `create-container.sh` to match the D1 container used for the report. Use the same absolute `RUN_DIR` on all hosts.

Run each role on its designated host. Use separate terminals and start the two D nodes in quick succession.

```bash
docker exec "$CONTAINER" bash /repro/run.sh p      # P: DP8
docker exec "$CONTAINER" bash /repro/run.sh d0     # D0: global DP ranks 0–7
docker exec "$CONTAINER" bash /repro/run.sh d1     # D1: global DP ranks 8–15
docker exec "$CONTAINER" bash /repro/run.sh proxy  # D0: port 1999
```

D uses TP1 for the model, LM-head TP8, expert parallelism, DeepGEMM MegaMoE with `enable_fused_mc2=3`, MegaMHC, indexer QW, graph capture limit 480, and DSpark with five draft tokens and synthetic acceptance target 3.51. D sets `max_num_seqs=80`, `max_num_batched_tokens=1024`, and GPU memory utilization 0.92. The historical BS80 run used automatic graph sizes with static kernel disabled. P uses eager execution and prefix caching.

For model accuracy evaluation, set D's `enable_force_eplb` to `false` and remove the synthetic rejection-sampling settings so DSpark uses standard rejection sampling.

## 5. Run the benchmark and check the result

Copy `$BUNDLE_DIR/prompt-128k.jsonl` from the extraction host to `$RUN_DIR/prompt-128k.jsonl` in the shared run directory visible to D0, then run the client on D0. The script waits for all 24 ranks and the proxy, warms P twice, then sends 1,280 requests with 2,048 output tokens each. It waits for a six-second window with 80 active requests on every D rank, zero P work in flight, and zero ongoing D KV fetches.

```bash
docker exec "$CONTAINER" bash /repro/run.sh bench "$RUN_DIR/prompt-128k.jsonl"
```

Check `$RUN_DIR/pure-d-128k-2k/summary.json` for 1,280 successful requests and verified prefix reuse. Use `steady-throughput.json` for the measured throughput. The historical run measured **77,200 output tok/s** across 16 D ranks, or **4,825 tok/s per card**.

For the trace-based TPOT, run the same workload with profiling in a separate pass after the benchmark:

```bash
docker exec "$CONTAINER" bash /repro/run.sh bench "$RUN_DIR/prompt-128k.jsonl" --profile
```

The client captures 10 steps on D rank 0 after the steady window and writes its own results under `$RUN_DIR/profile-bench`. The trace is under `$RUN_DIR/profile`. Take the interval between successive main graph replays and divide by the measured acceptance length including the bonus token. The historical trace gave approximately 56 ms per step, or **16.0 ms/token** at 3.50 tokens per step. Report trace TPOT and measured throughput as separate metrics.

After the benchmark, run `docker stop "$CONTAINER"` on each host to release the cards. The output files are in the mounted `RUN_DIR`.

## 6. GPQA Diamond accuracy

The r148 stack scored **183/198 (92.42%)** in a full GPQA Diamond run. All 198 requests returned. Of these, 197 completed normally and one reached the output limit before producing a final choice. The truncated answer counted as incorrect. There were no transport errors.

The accuracy run used one 8-card Prefill node and one 8-card Decode node, each with DP8/EP8. The proxy routed all requests through P rank 0 and D rank 0, with 16 concurrent requests. D used standard DSpark rejection sampling with `enable_force_eplb=false`. Requests used `temperature=1.0`, `top_p=1.0`, and a limit of 131,072 generated tokens per answer, with thinking enabled and `reasoning_effort=100`.
