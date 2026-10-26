"""Reload every default DS41 export using installed AOT binaries only."""
import argparse, importlib, json, os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--report",required=True);p.add_argument("--cores",type=int);p.add_argument("--only",nargs="+");a=p.parse_args()
os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"]="require"
os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"]=a.report
os.environ.pop("CANNBOTDSL_CACHE_DIR",None)
os.environ.pop("CANNBOTDSL_DS41_PROFILES",None)
import ops
if a.cores:
    import torch, torch_npu, cannbotdsl
    original_torch_properties=torch.npu.get_device_properties
    original_platform_info=cannbotdsl.get_platform_info
    class Properties:
        def __init__(self,real):self.real=real
        def __getattr__(self,name):
            if name=="cube_core_num":return a.cores
            if name=="vector_core_num":return 2*a.cores
            return getattr(self.real,name)
    torch.npu.get_device_properties=lambda *args,**kwargs:Properties(original_torch_properties(*args,**kwargs))
    cannbotdsl.get_platform_info=lambda *args,**kwargs:Properties(original_platform_info(*args,**kwargs))
exports=[("mixed_quant_sparse_flash_mla","export_mqsmla"),("quant_lightning_indexer_dsl","export_qli"),("quant_sparse_lightning_indexer_dsl","export_qsli"),("indexer_prologue_qw","export_indexer_prologue_qw"),("indexer_prologue_k","export_indexer_prologue_k"),("engram_gate","export_engram_gate"),("attn_epilogue","export_attn_epilogue")]
results=[]
previous=0
for module,function in exports:
    if a.only and module not in a.only:continue
    getattr(importlib.import_module("ops."+module),function)()
    data=json.loads(Path(a.report).read_text())
    counts=data["counts"]
    assert not counts.get("installed_miss",0),data
    assert not counts.get("backend_compile",0),data
    hits=counts.get("installed_hit",0)
    assert hits>previous,(module,counts)
    results.append({"module":module,"installed_hits":hits-previous,"status":"PASS"})
    previous=hits
    print(results[-1],flush=True)
Path(a.report+".validation.json").write_text(json.dumps({"status":"PASS","operators":results,"counts":counts},indent=2))
