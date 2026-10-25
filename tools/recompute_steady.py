import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('case')
a = parser.parse_args()
case = pathlib.Path(a.case)
assert json.loads((case / 'exit.json').read_text())['returncode'] == 0
originals = list((case / 'output').iterdir())
assert len(originals) == 1, originals
stamp = originals[0].name
destination = case / 'stable-wave01'
destination.mkdir(exist_ok=False)
# Keep a separate result tree; AISBench reuses raw timestamps and rewrites its
# CSV/JSON summaries. The original full-window results must remain untouched.
shutil.copytree(originals[0], destination / 'output' / stamp)
for name in ['dataset.json', 'monitor-summary.json']:
    shutil.copy2(case / name, destination / name)
config = (case / 'config.py').read_text()
assert "summarizer = dict(attr='performance')" in config
config = config.replace("summarizer = dict(attr='performance')", '''from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import StablePerfMetricCalculator
summarizer = dict(attr='performance', type=DefaultPerfSummarizer,
    calculator=dict(type=StablePerfMetricCalculator,
        stats_list=['Average','Min','Max','Median','P75','P90','P99']))''')
(destination / 'config.py').write_text(config)
source = pathlib.Path('/vllm-workspace/vllm-ascend/benchmark/ais_bench/benchmark/calculators/stable_perf_metric_calculator.py')
command = [sys.executable, '/workspace/experiment/scripts/aisbench_wave01.py', str(destination / 'config.py'),
           '--mode', 'perf', '--reuse', stamp, '--work-dir', str(destination / 'output'), '--debug']
provenance = {'source_case': str(case), 'wave_offset': 0.1, 'original_source': str(source),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'command': command, 'inference_skipped_by_reuse': True, 'started': time.time()}
(destination / 'provenance.json').write_text(json.dumps(provenance, indent=2))
with (destination / 'aisbench-steady.log').open('w') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', NO_PROXY='*',
                 YLF_STEADY_WINDOW_JSON=str(destination / 'steady-window.json')))
provenance.update(returncode=result.returncode, ended=time.time())
(destination / 'provenance.json').write_text(json.dumps(provenance, indent=2))
assert result.returncode == 0, result.returncode
from parse_result import parse
summary = parse(destination)
summary.update(statistic_scope='stable', wave_offset=0.1, source_case=case.name)
window = json.loads((destination / 'steady-window.json').read_text())
summary['stable_window'] = {k: v for k, v in window.items() if k != 'selected_request_indices'}
summary['selected_requests'] = len(window['selected_request_indices'])
(destination / 'result-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
