"""Small NPU precision checks using the upstream independent MXFP4 reference.

--cores=28 on a 32-core device validates a 28-worker launch, not a physical
28-core SKU. The report records this distinction explicitly.
"""
import argparse, json, os, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--cores", type=int, required=True)
parser.add_argument("--report", required=True)
args = parser.parse_args()
os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"] = "require"
os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"] = args.report + ".native.json"
os.environ.pop("CANNBOTDSL_CACHE_DIR", None)
import torch, torch_npu
import reference_topk as ref
from ops.quant_lightning_indexer_dsl import quant_lightning_indexer as qli
from ops.quant_lightning_indexer_metadata_dsl import quant_lightning_indexer_metadata as qli_metadata
from ops.quant_sparse_lightning_indexer_dsl import quant_sparse_lightning_indexer as qsli
from ops.quant_sparse_lightning_indexer_metadata_dsl import quant_sparse_lightning_indexer_metadata as qsli_metadata

# A missing packaged AICPU metadata binary must fail, rather than compiling
# silently and making the release look self-contained.
import cannbotdsl.aicpu.toolchain as aicpu_toolchain
def reject_aicpu_compile(*args,**kwargs):
    raise AssertionError("Unexpected runtime AICPU metadata compilation")
aicpu_toolchain.compile_aicpu_kernel=reject_aicpu_compile

torch.set_num_threads(4)
device = ref._npu_device()
original_properties = torch.npu.get_device_properties
actual = original_properties(device)
physical = int(actual.cube_core_num)
assert 1 <= args.cores <= physical, (args.cores, physical)
class Properties:
    def __init__(self, real): self.real = real
    def __getattr__(self, name):
        return args.cores if name == "cube_core_num" else getattr(self.real, name)
# Alter only the reported worker count consumed by QLI/QSLI and metadata.
# Device, tensor memory, kernels and synchronization remain actual NPU calls.
torch.npu.get_device_properties = lambda device: Properties(original_properties(device))
cases = []
for sparse,candidate,page in [(False,False,64),(False,True,128),(True,True,128)]:
    for has_used_q,query_lengths,key_lengths in [(True,(3,4),(513,777)),(False,(3,4),(513,777)),(False,(1,2),(129,513))]:
        start = time.time()
        data = ref._make_inputs("PA_BBND", page, 3, sparse=sparse, candidates=candidate, query_lengths=query_lengths,key_lengths=key_lengths)
        if not has_used_q: data["sequence"].pop("seqused_q")
        metadata_args = (data["candidate_lengths"],) if sparse else ()
        metadata = (qsli_metadata if sparse else qli_metadata)(*metadata_args, **data["sequence"], **data["metadata_attributes"])
        kernel_args = [data[key] for key in ("q","k","w","descale_q")]
        if sparse: kernel_args += [data["candidate_ids"], data["candidate_lengths"]]
        else: kernel_args += [data["descale_k"]]
        kwargs = dict(data["sequence"], **data["attributes"], quant_mode=1, block_table=data["block_table"], metadata=metadata, return_value=True)
        if sparse: kwargs["descale_k"] = data["descale_k"]
        out = (qsli if sparse else qli)(*kernel_args, **kwargs)
        torch.npu.synchronize()
        indices, values = [value.cpu() for value in out[:2]]
        assert tuple(indices.shape) == (sum(query_lengths),1,512)
        for row,(scores,allowed) in enumerate(zip(data["reference"],data["allowed"])):
            ref._check_topk(indices[row,0], values[row,0], scores, allowed)
        cases.append({"operator":"qsli" if sparse else "qli", "candidate":candidate, "page":page, "has_seqused_q":has_used_q, "query_lengths":query_lengths, "key_lengths":key_lengths, "status":"PASS", "seconds":round(time.time()-start,3)})
        print(cases[-1], flush=True)
report = {"status":"PASS", "physical_cube_cores":physical, "launch_cube_workers":args.cores, "physical_28_core_device":physical==28, "device":actual.name, "cases":cases}
Path(args.report).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
