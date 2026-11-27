"""Reproduce the frozen 128K PD load and record per-rank admission and decode."""

import asyncio
import hashlib
import json
import time

import aiohttp


API = '/v1/chat/completions'
RUNNING = 'vllm:num_requests_running'
WAITING = 'vllm:num_requests_waiting'
FIRST = 'vllm:time_to_first_token_seconds_count'
TOKENS = 'vllm:generation_tokens_total'
HITS = 'vllm:prefix_cache_hits_total'
QUERIES = 'vllm:prefix_cache_queries_total'
KV_FETCH = 'vllm:num_requests_kv_fetch_by_stage'
KV_STAGES = ('waiting_to_start', 'in_progress', 'completed_waiting')
ENDPOINTS = {}
START = time.monotonic()


def save(path, obj):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(obj, indent=2) + '\n')
    pending.replace(path)


def parse_prom(raw):
    result = {}
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        metric, value = line.rsplit(' ', 1)
        name = metric.split('{', 1)[0]
        if name == KV_FETCH:
            stage = metric.partition('stage="')[2].partition('"')[0]
            name = f'{name}:{stage}'
        if not name.endswith(('_created', '_bucket')):
            result[name] = result.get(name, 0) + float(value)
    return result


async def snapshot(session, raw_path=None):
    async def fetch(name, url):
        async with session.get(url + '/metrics', timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            body = await response.text()
        return name, body, time.monotonic() - START

    responses = await asyncio.gather(*(fetch(key, url) for key, url in ENDPOINTS.items()))
    if raw_path is not None:
        raw_path.write_text('\n'.join(f'# endpoint={key}\n{body}' for key, body, _ in responses))
    return dict(time=time.monotonic() - START,
                scrape_span_s=max(row[2] for row in responses) - min(row[2] for row in responses),
                ranks={key: parse_prom(body) for key, body, _ in responses})


def total(sample, role, metric):
    return sum(row.get(metric, 0) for key, row in sample['ranks'].items() if key.startswith(role))


async def send(session, url, api, body, expected_input, output, rid):
    row = dict(id=rid, send=time.monotonic() - START, wall_send=time.time())
    usage = None
    seen_done = False
    checksum = hashlib.sha256()
    try:
        headers = {'Content-Type': 'application/json', 'X-Request-Id': rid}
        async with session.post(url + api, data=body, headers=headers) as response:
            row['http_status'] = response.status
            if response.status != 200:
                raise RuntimeError((await response.text())[:500])
            async for raw in response.content:
                if not raw.startswith(b'data:'):
                    continue
                data = raw[5:].strip()
                now = time.monotonic() - START
                if data == b'[DONE]':
                    seen_done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error'])[:500])
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    generated = (delta.get('reasoning_content') or delta.get('reasoning') or '') + (
                        delta.get('content') or choice.get('text') or '')
                    if generated:
                        row.setdefault('first', now)
                        checksum.update(generated.encode())
                    if choice.get('finish_reason'):
                        row['finish'] = now
                        row['finish_reason'] = choice['finish_reason']
                usage = event.get('usage') or usage
        if not (seen_done and usage and 'finish' in row):
            raise RuntimeError('Missing usage, finish or end-of-stream')
        if (usage.get('prompt_tokens'), usage.get('completion_tokens')) != (expected_input, output):
            raise RuntimeError(f'Unexpected usage {usage}')
        if output > 1 and 'first' not in row:
            raise RuntimeError('Missing first streamed token')
        row.update(success=True, usage=usage, answer_sha256=checksum.hexdigest())
    except (aiohttp.ClientError, TimeoutError, RuntimeError, ValueError) as exc:
        row.update(success=False, error=str(exc), usage=usage)
    row['end'] = time.monotonic() - START
    return row
