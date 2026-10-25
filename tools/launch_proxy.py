import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path('/workspace/experiment')
SOURCE = pathlib.Path('/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py')
parser = argparse.ArgumentParser()
parser.add_argument('--topology', choices=['dp32', 'pcp8', 'pcp8-1p', 'pcp8-3p-remap', 'pcp8-2p-remap'], required=True)
parser.add_argument('--tag', required=True)
a = parser.parse_args()
p_nodes = [21, 23] if a.topology == 'dp32' else [21, 23, 25]
d_endpoints = [(n, 18200 + r) for n in ([25, 26, 27, 28] if a.topology == 'dp32' else [27, 28]) for r in (range(8) if a.topology == 'dp32' else [0])]
if a.topology == 'pcp8-1p':
    p_nodes = [21]
    d_endpoints = [(23, 18200), (24, 18200)]
if a.topology in ['pcp8-3p-remap','pcp8-2p-remap']:
    p_nodes = [21,25] if a.topology == 'pcp8-2p-remap' else [21,25,27]
    d_endpoints = [(23, 18200), (24, 18200)]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for node, port in [(n, 18100) for n in p_nodes] + d_endpoints:
    with opener.open('http://197.33.%d.1:%d/health' % (node + 24, port), timeout=10) as response:
        assert response.status == 200
command = [sys.executable, str(SOURCE), '--host', '197.33.45.1', '--port', '18300', '--workers', '4', '--prefiller-hosts'] + ['197.33.%d.1' % (n + 24) for n in p_nodes] + ['--prefiller-ports'] + ['18100'] * len(p_nodes) + ['--decoder-hosts'] + ['197.33.%d.1' % (n + 24) for n, port in d_endpoints] + ['--decoder-ports'] + [str(port) for n, port in d_endpoints]
logdir = ROOT / 'logs' / a.tag
logdir.mkdir(parents=True,exist_ok=True)
record = {'started': time.time(), 'source': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'command': command, 'p_instances': len(p_nodes), 'd_instances': len(d_endpoints)}
(logdir / 'proxy.args.json').write_text(json.dumps(record, indent=2))
with (logdir / 'proxy.log').open('x') as stream:
    proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    record['pid']=proc.pid
    (logdir / 'proxy.args.json').write_text(json.dumps(record, indent=2))
    proc.wait()
    result=proc
(logdir / 'proxy.exit.json').write_text(json.dumps({'returncode': result.returncode, 'ended': time.time()}))
raise SystemExit(result.returncode)
