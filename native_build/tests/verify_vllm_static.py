"""Run the actual three vLLM contracts in a fresh process, requiring AOT hits."""
from pathlib import Path
import os, json, argparse, importlib.metadata as md

parser = argparse.ArgumentParser()
parser.add_argument("--report", required=True)
args = parser.parse_args()
report_path = Path(args.report)
os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"] = "require"
os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"] = str(report_path)
os.environ.pop("CANNBOTDSL_CACHE_DIR", None)
import ops
import ops.quant_lightning_indexer_dsl as q
import ops.quant_sparse_lightning_indexer_dsl as s
cases = [
    ("qli_no_candidate", q._build_compiled_fused_runner, (512,False,1,3,2,True,False,True,True,64,8,32,6)),
    ("qli_candidate", q._build_compiled_fused_runner, (512,True,2048,3,1,True,False,True,True,128,8,32,6)),
    ("qsli", s._build_compiled_fused_runner, (16384,512,2048,3,1,True,False,True,True,False,128,8)),
]
for name,function,config in cases:
    runner = function(*config)
    runner.close()
    print(name, "STATIC LOADED", flush=True)
data = json.loads(report_path.read_text())
assert data["counts"].get("installed_hit") == 3, data
assert not data["counts"].get("installed_miss", 0), data
assert not data["counts"].get("backend_compile", 0), data
result = {"status":"PASS", "python":__import__("sys").version, "ops_path":ops.__file__, "ops_version":md.version("cannbot-arena-net"), "dsl_version":md.version("cannbot-dsl"), "native_report":data}
report_path.with_name(report_path.stem+"_validation.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
