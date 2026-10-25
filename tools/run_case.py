import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import pathlib
import random
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path('/workspace/experiment')
MODEL = '/models/DeepSeek-V4-Pro-0813'
MODEL_NAME = 'dsv4-pro-0813'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
FIELDS = ['num_requests_running', 'num_requests_waiting', 'request_success_total', 'generation_tokens_total', 'prefix_cache_queries_total', 'prefix_cache_hits_total', 'external_prefix_cache_queries_total', 'external_prefix_cache_hits_total', 'time_to_first_token_seconds_sum', 'time_to_first_token_seconds_count', 'request_queue_time_seconds_sum', 'request_queue_time_seconds_count', 'e2e_request_latency_seconds_sum', 'e2e_request_latency_seconds_count', 'kv_cache_usage_perc', 'num_preemptions_total', 'request_num_preemptions_sum']


def http(url, data=None, timeout=20):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as response:
        return response.read()


def endpoints(topology):
    if topology == 'pcp8-3p-remap':
        return ([{'role': 'P', 'node': n, 'rank': i, 'url': 'http://197.33.%d.1:18100' % (n+24)} for i,n in enumerate([21,25,27])],
                [{'role': 'D', 'node': n, 'rank': n-23, 'url': 'http://197.33.%d.1:18200' % (n+24)} for n in [23,24]])
    if topology == 'pcp8-1p':
        return ([{'role': 'P', 'node': 21, 'rank': 0, 'url': 'http://197.33.45.1:18100'}],
                [{'role': 'D', 'node': n, 'rank': n-23, 'url': 'http://197.33.%d.1:18200' % (n+24)} for n in [23,24]])
    ps = [{'role': 'P', 'node': n, 'rank': (n - 21) // 2, 'url': 'http://197.33.%d.1:18100' % (n + 24)} for n in ([21, 23] if topology == 'dp32' else [21, 23, 25])]
    ds = [{'role': 'D', 'node': n, 'rank': (n - 25) * 8 + i if topology == 'dp32' else n - 27, 'url': 'http://197.33.%d.1:%d' % (n + 24, 18200 + i)} for n in ([25, 26, 27, 28] if topology == 'dp32' else [27, 28]) for i in (range(8) if topology == 'dp32' else [0])]
    return ps, ds


def metric(target):
    row = dict(target, time=time.time())
    try:
        raw = http(target['url'] + '/metrics', timeout=4).decode()
        row['values'] = {key: 0.0 for key in FIELDS}
        row['raw'] = []
        row['metrics_present'] = []
        for line in raw.splitlines():
            if not line or line.startswith('#'):
                continue
            name = line.split('{', 1)[0].split(' ', 1)[0]
            key = name.removeprefix('vllm:')
            if key in row['values']:
                row['values'][key] += float(line.rsplit(' ', 1)[1])
                row['raw'].append(line)
                row['metrics_present'].append(key)
    except Exception as exc:
        row['error'] = repr(exc)
    return row


def snapshot(path, targets):
    with cf.ThreadPoolExecutor(max_workers=36) as pool:
        rows = list(pool.map(metric, targets))
    path.write_text(json.dumps(rows, indent=2))
    assert all('error' not in x for x in rows), rows
    return rows


def make_data(case, count, input_len, output_len, fraction, request_index_start=0):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MODEL + '/tokenizer.json')
    length = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    # Stable mixed text avoids a uniform repeated-token prompt. The same seed
    # and request index produce identical input across both deployment layouts.
    atoms = []
    for token in sorted(tok.get_vocab()):
        candidate = token.replace('Ġ', ' ', 1).replace('▁', ' ', 1)
        if re.fullmatch(r' [A-Za-z]{3,12}', candidate) and length(candidate) == 1:
            atoms.append(candidate)
            if len(atoms) == 1024:
                break
    assert len(atoms) >= 256, len(atoms)
    prefix_len = int(input_len * fraction) // 32 * 32
    prefix = ''.join(random.Random(1).choices(atoms, k=prefix_len))
    assert length(prefix) == prefix_len
    digest = hashlib.sha256()
    first = None
    dataset = case / 'dataset.jsonl'
    with dataset.open('wb') as stream:
        for index in range(count):
            suffix = ''.join(random.Random(1000003 + request_index_start + index).choices(atoms, k=input_len - prefix_len))
            question = prefix + suffix
            if index < 3 or index == count - 1:
                assert length(question) == input_len, (index, length(question), input_len)
            line = (json.dumps({'question': question, 'answer': '', 'max_tokens': output_len, 'max_out_len': output_len}, ensure_ascii=False) + '\n').encode()
            digest.update(line)
            stream.write(line)
            if first is None:
                first = question
    metadata = {'data_num': count, 'input_tokens': input_len, 'output_tokens': output_len, 'requested_prefix_fraction': fraction, 'prefix_tokens': prefix_len, 'block_aligned_prefix_fraction': prefix_len / input_len, 'data_method': 'mixed single-token words; shared prefix seed=1; suffix seed=1000003+request_index_start+request_index', 'request_index_start': request_index_start, 'vocabulary_sha256': hashlib.sha256(json.dumps(atoms).encode()).hexdigest(), 'prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(), 'dataset_sha256': digest.hexdigest(), 'first_question_sha256': hashlib.sha256(first.encode()).hexdigest(), 'round_trip_tokenizer': 'tokenizers.Tokenizer / add_special_tokens=False', 'path': str(dataset)}
    (case / 'dataset.json').write_text(json.dumps(metadata, indent=2))
    return first, prefix, metadata


def make_config(case, concurrency, output_len):
    config = '''from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator

models = [dict(attr='service', type=VLLMCustomAPI, abbr='pro0813', path=MODEL_PATH,
    model='dsv4-pro-0813', stream=True, request_rate=0, use_timestamp=False,
    retry=1, api_key='', host_ip='197.33.45.1', host_port=18300,
    max_out_len=OUTPUT_LEN, batch_size=CONCURRENCY, trust_remote_code=True,
    generation_kwargs=dict(temperature=0, ignore_eos=True, add_special_tokens=False))]
datasets = [dict(abbr='pro0813-case', type=CustomDataset, path=DATASET_PATH,
    reader_cfg=dict(input_columns=['question','max_out_len'], output_column='answer'),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='{question}'),
        retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)),
    eval_cfg=dict(evaluator=dict(type=AccEvaluator), pred_role='BOT'))]
summarizer = dict(attr='performance')
'''
    config = config.replace('MODEL_PATH', repr(MODEL)).replace('OUTPUT_LEN', str(output_len)).replace('CONCURRENCY', str(concurrency)).replace('DATASET_PATH', repr(str(case / 'dataset.jsonl')))
    (case / 'config.py').write_text(config)


def summarize_monitor(case, ds, before, after):
    samples = []
    for line in (case / 'samples.jsonl').open():
        row = json.loads(line)
        if 'error' not in row:
            samples.append(row)
    rows = []
    for endpoint in ds:
        valid = [s for s in samples if s.get('role') == 'D' and s['rank'] == endpoint['rank']]
        running = [s['values']['num_requests_running'] for s in valid]
        waiting = [s['values']['num_requests_waiting'] for s in valid]
        initial = next(s for s in before if s['role'] == 'D' and s['rank'] == endpoint['rank'])['values']
        final = next(s for s in after if s['role'] == 'D' and s['rank'] == endpoint['rank'])['values']
        rows.append(dict(endpoint, samples=len(valid), running_peak=max(running, default=0), running_mean=statistics.mean(running) if running else None, waiting_peak=max(waiting, default=0), success_delta=final['request_success_total'] - initial['request_success_total'], generation_tokens_delta=final['generation_tokens_total'] - initial['generation_tokens_total'], kv_usage_peak=max((s['values']['kv_cache_usage_perc'] for s in valid),default=None), preemptions_delta=final['num_preemptions_total']-initial['num_preemptions_total'], request_preemptions_sum_delta=final['request_num_preemptions_sum']-initial['request_num_preemptions_sum'], kv_preemption_metrics_present=all(set(['kv_cache_usage_perc','num_preemptions_total','request_num_preemptions_sum']) <= set(s.get('metrics_present',[])) for s in valid)))
    grouped = {}
    for s in samples:
        if s.get('role') == 'D':
            grouped.setdefault(s['tick'], []).append(s)
    simultaneous = []
    for tick, group in grouped.items():
        if len(group) == len(ds):
            values = [s['values']['num_requests_running'] for s in group]
            total = sum(values)
            simultaneous.append({'tick': tick, 'running_total': total, 'waiting_total': sum(s['values']['num_requests_waiting'] for s in group), 'running_min': min(values), 'running_max': max(values), 'cv': statistics.pstdev(values) / statistics.mean(values) if total else 0})
    proxy = [json.loads(line) for line in (case / 'proxy-samples.jsonl').open()]
    active = [s['value'].get('request_num', 0) for s in proxy if 'value' in s]
    summary = {'per_dp': rows, 'all_dp_have_samples': all(s['samples'] > 0 for s in rows), 'success_delta_total': sum(s['success_delta'] for s in rows), 'generation_tokens_delta_total': sum(s['generation_tokens_delta'] for s in rows), 'actual_proxy_inflight_peak': max(active, default=None), 'simultaneous_d_running_peak': max((s['running_total'] for s in simultaneous), default=0), 'peak_sample': max(simultaneous, key=lambda s: s['running_total'], default=None), 'simultaneous_samples': simultaneous}
    (case / 'monitor-summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', choices=['dp32', 'pcp8', 'pcp8-1p', 'pcp8-3p-remap'], required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--data-num', type=int)
    parser.add_argument('--request-index-start', type=int, default=0)
    parser.add_argument('--kv-usage-limit', type=float)
    parser.add_argument('--sample-interval', type=float, default=1)
    parser.add_argument('--input-len', type=int, required=True)
    parser.add_argument('--output-len', type=int, required=True)
    parser.add_argument('--prefix-fraction', type=float, default=0)
    parser.add_argument('--service-tag', required=True)
    parser.add_argument('--p-service-tag')
    parser.add_argument('--fresh-prefix-without-reset', action='store_true')
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--dry-run', action='store_true')
    a = parser.parse_args()
    assert a.kv_usage_limit is None or 0 < a.kv_usage_limit <= 1
    assert a.sample_interval > 0
    case = ROOT / 'benchmarks' / a.case_id
    case.mkdir(parents=True, exist_ok=False)
    count = a.data_num if a.data_num is not None else 6 * a.concurrency
    assert count > 0 and a.request_index_start >= 0
    first, prefix, metadata = make_data(case, count, a.input_len, a.output_len, a.prefix_fraction, a.request_index_start)
    metadata['hardware_cards'] = 32 if a.topology == 'pcp8-1p' else 64
    metadata['topology'] = a.topology
    metadata['p_service_tag'] = a.p_service_tag or a.service_tag
    (case / 'dataset.json').write_text(json.dumps(metadata, indent=2))
    make_config(case, a.concurrency, a.output_len)
    cmd = ['ais_bench', str(case / 'config.py'), '--mode', 'perf', '--work-dir', str(case / 'output'), '--num-prompts', str(count), '--num-warmups', '0', '--debug']
    if a.dry_run:
        # The bundled AISBench debug workflow still sends traffic with --dry-run.
        # Validate the generated config directly, without running its task workflow.
        from mmengine.config import Config
        from ais_bench.benchmark.cli.config_manager import CustomConfigChecker
        config = Config.fromfile(str(case / 'config.py'))
        CustomConfigChecker(config, str(case / 'config.py')).check()
        print(json.dumps({'config_valid': True, 'concurrency': config.models[0].batch_size, 'data_num': count, 'dataset': str(case / 'dataset.jsonl')}), flush=True)
        return
    ps, ds = endpoints(a.topology)
    for target in ps + ds:
        http(target['url'] + '/health')
    proxy_url = 'http://197.33.45.1:18300'
    status = json.loads(http(proxy_url + '/healthcheck'))
    assert status.get('request_num', 0) == 0, status
    snapshot(case / 'pre-warmup.json', ps + ds)
    if prefix:
        resets = []
        for target in ps:
            try:
                result = json.loads(http(target['url'] + '/reset_prefix_cache', {}, timeout=60))
                assert result.get('success', True), result
                resets.append(dict(target, result=result))
            except urllib.error.HTTPError as exc:
                if exc.code != 404 or not a.fresh_prefix_without_reset:
                    raise
                # This image exposes cache reset only in server dev mode.
                # Reusing complete prompts would turn a nominal prefix test
                # into full cache hits, so permit this only for unused inputs
                # within the explicitly recorded P process lifetime.
                for old in (ROOT / 'benchmarks').glob('*/dataset.json'):
                    if not (old.parent / 'before.json').exists():
                        continue
                    prior = json.loads(old.read_text())
                    assert not (
                        prior.get('p_service_tag') == metadata['p_service_tag']
                        and prior.get('first_question_sha256') == metadata['first_question_sha256']
                    ), ('Prompts already used with these P processes; cache reset or a P restart is required', str(old))
                resets.append(dict(target, status='reset endpoint unavailable; inputs unused in this P lifetime', http_status=404))
        (case / 'cache-reset.json').write_text(json.dumps(resets, indent=2))
        warms = []
        for target in ps:
            response = json.loads(http(target['url'] + '/v1/completions', {'model': MODEL_NAME, 'prompt': prefix, 'max_tokens': 1, 'ignore_eos': True, 'add_special_tokens': False, 'temperature': 0}, timeout=1200))
            warms.append(dict(target, response=response))
            assert response['usage']['prompt_tokens'] == metadata['prefix_tokens'], response.get('usage')
        (case / 'warmup.json').write_text(json.dumps(warms, indent=2))
    before = snapshot(case / 'before.json', ps + ds)
    assert sum(s['values']['num_requests_running'] + s['values']['num_requests_waiting'] for s in before) == 0
    logdir = ROOT / 'logs' / a.service_tag
    offsets = {str(p): p.stat().st_size for p in logdir.glob('*d-rank*.log')}
    assert len(offsets) == len(ds), offsets
    (case / 'log-offsets.json').write_text(json.dumps(offsets, indent=2))
    stop = threading.Event()
    guard_events = []
    proc = None
    def abort_client(reason):
        if not guard_events:
            guard_events.append(reason)
            (case / 'guard-triggered.json').write_text(json.dumps(reason,indent=2))
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
    def monitor():
        tick = 0
        with (case / 'samples.jsonl').open('w') as stream, (case / 'proxy-samples.jsonl').open('w') as proxy_stream, cf.ThreadPoolExecutor(max_workers=36) as pool:
            while not stop.is_set():
                start = time.monotonic()
                for row in pool.map(metric, ps + ds):
                    row['tick'] = tick
                    stream.write(json.dumps(row) + '\n')
                    if a.kv_usage_limit is not None and row.get('role')=='D' and 'error' not in row:
                        v=row['values'];baseline=next(z['values'] for z in before if z['role']=='D' and z['rank']==row['rank'])
                        if not set(['kv_cache_usage_perc','num_preemptions_total','request_num_preemptions_sum']) <= set(row.get('metrics_present',[])):
                            abort_client(dict(reason='required KV/preemption metric unavailable',sample=row))
                        elif v['kv_cache_usage_perc']>=a.kv_usage_limit or v['num_preemptions_total']>baseline['num_preemptions_total'] or v['request_num_preemptions_sum']>baseline['request_num_preemptions_sum']:
                            abort_client(dict(reason='KV limit or preemption guard',limit=a.kv_usage_limit,sample=row))
                try:
                    proxy_row = {'tick': tick, 'time': time.time(), 'value': json.loads(http(proxy_url + '/healthcheck', timeout=4))}
                except Exception as exc:
                    proxy_row = {'tick': tick, 'time': time.time(), 'error': repr(exc)}
                proxy_stream.write(json.dumps(proxy_row) + '\n')
                stream.flush(); proxy_stream.flush()
                tick += 1
                stop.wait(max(0, a.sample_interval - (time.monotonic() - start)))
    thread = threading.Thread(target=monitor)
    thread.start()
    started = time.time()
    try:
        with (case / 'aisbench.log').open('w') as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', NO_PROXY='*', TOKENIZERS_PARALLELISM='true', RAYON_NUM_THREADS='16'),start_new_session=True)
            if guard_events:
                os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                proc.wait(timeout=30)
                raise
            result=proc
        ended = time.time()
        (case / 'window.json').write_text(json.dumps({'started': started, 'ended': ended, 'returncode': result.returncode, 'command': cmd, 'concurrency': a.concurrency, 'data_num': count}, indent=2))
        time.sleep(12)
        after = snapshot(case / 'after.json', ps + ds)
    finally:
        stop.set(); thread.join()
    summary = summarize_monitor(case, ds, before, after)
    (case / 'decode-window-logs').mkdir()
    for filename, offset in offsets.items():
        p = pathlib.Path(filename)
        with p.open('rb') as source:
            source.seek(offset)
            (case / 'decode-window-logs' / p.name).write_bytes(source.read())
    recompute_matches=[]
    for p in (case/'decode-window-logs').glob('*.log'):
        for line in p.read_text(errors='replace').splitlines():
            if re.search(r'sending.*request back to P for recomputation|KV offload.*(?:unavailable|failed|raised)|preempted.*(?:recompute|RECOMPUTE)',line,re.I):
                recompute_matches.append(dict(log=p.name,line=line))
    validation={'kv_usage_limit':a.kv_usage_limit,'sample_interval_seconds':a.sample_interval,'guard_events':guard_events,'per_dp':[{k:v for k,v in z.items() if k in ['rank','node','samples','kv_usage_peak','preemptions_delta','request_preemptions_sum_delta','kv_preemption_metrics_present']} for z in summary['per_dp']],'recompute_log_matches':recompute_matches}
    validation['passed']=not guard_events and not recompute_matches and all(z['kv_preemption_metrics_present'] and z['preemptions_delta']==0 and z['request_preemptions_sum_delta']==0 and (a.kv_usage_limit is None or z['kv_usage_peak']<a.kv_usage_limit) for z in summary['per_dp'])
    (case/'kv-preemption-validation.json').write_text(json.dumps(validation,indent=2))
    cache_before = sum(s['values']['prefix_cache_hits_total'] for s in before if s['role'] == 'P')
    cache_after = sum(s['values']['prefix_cache_hits_total'] for s in after if s['role'] == 'P')
    queries_before = sum(s['values']['prefix_cache_queries_total'] for s in before if s['role'] == 'P')
    queries_after = sum(s['values']['prefix_cache_queries_total'] for s in after if s['role'] == 'P')
    (case / 'observed-prefix.json').write_text(json.dumps({'hits_delta': cache_after - cache_before, 'queries_delta': queries_after - queries_before, 'observed_fraction': (cache_after - cache_before) / (queries_after - queries_before) if queries_after > queries_before else None}, indent=2))
    (case / 'exit.json').write_text(json.dumps({'returncode': result.returncode, 'success_delta_total': summary['success_delta_total'], 'expected_requests': count, 'all_dp_have_samples': summary['all_dp_have_samples']}, indent=2))
    print(json.dumps({'case': str(case), 'returncode': result.returncode, 'monitor_summary': {k: v for k, v in summary.items() if k not in ['per_dp', 'simultaneous_samples']}}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
