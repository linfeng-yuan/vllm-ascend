"""Measure P throughput and queues during the PD benchmark; not a P saturation test."""
import argparse
import json
from pathlib import Path
import statistics

from parse_result import number

p=argparse.ArgumentParser()
p.add_argument('case')
a=p.parse_args()
case=Path(a.case)
summary=json.loads((case/'result-summary.json').read_text())
before=json.loads((case/'before.json').read_text())
after=json.loads((case/'after.json').read_text())
duration=number(summary['common_metrics_as_printed']['Benchmark Duration']['total'])/1000
steady=json.loads((case/'stable-wave01/steady-monitor.json').read_text())
start,end=steady['window_start_unix_estimate'],steady['window_end_unix_estimate']
samples=[json.loads(s) for s in (case/'samples.jsonl').open()]
rows=[]
for b in before:
    if b['role']!='P':continue
    node=b['node']
    e=next(x for x in after if x['role']=='P' and x['node']==node)
    delta={k:e['values'][k]-b['values'][k] for k in b['values']}
    n=delta['request_success_total']
    queries=delta['prefix_cache_queries_total']
    hits=delta['prefix_cache_hits_total']
    selected=[s for s in samples if s.get('role')=='P' and s.get('node')==node and 'error' not in s and start<=s['time']<=end]
    assert n>0 and queries>=hits>=0 and selected
    rows.append(dict(node=node,successful_requests=n,query_tokens=queries,hit_tokens=hits,
                     uncached_query_tokens=queries-hits,query_tokens_per_second=queries/duration,
                     uncached_query_tokens_per_second=(queries-hits)/duration,
                     request_throughput=n/duration,
                     p_request_ttft_avg_ms=1000*delta['time_to_first_token_seconds_sum']/delta['time_to_first_token_seconds_count'],
                     p_queue_avg_ms=1000*delta['request_queue_time_seconds_sum']/delta['request_queue_time_seconds_count'],
                     steady_samples=len(selected),
                     steady_running_mean=statistics.mean(s['values']['num_requests_running'] for s in selected),
                     steady_running_peak=max(s['values']['num_requests_running'] for s in selected),
                     steady_waiting_mean=statistics.mean(s['values']['num_requests_waiting'] for s in selected),
                     steady_waiting_peak=max(s['values']['num_requests_waiting'] for s in selected)))
assert sum(x['successful_requests'] for x in rows)==summary['report_metrics']['success_requests']
result=dict(case=case.name,benchmark_duration_seconds=duration,
            scope='P metrics deltas over the PD test; elapsed AISBench measurement duration, excluding prefix warmup',
            capacity_limit_measured=False,
            uncached_tokens_definition='delta(prefix_cache_queries_total) minus delta(prefix_cache_hits_total); not cached-inclusive AISBench input tokens',
            total_successful_requests=sum(x['successful_requests'] for x in rows),
            total_uncached_query_tokens_per_second=sum(x['uncached_query_tokens_per_second'] for x in rows),
            total_query_tokens_per_second=sum(x['query_tokens_per_second'] for x in rows),
            steady_window_start=start,steady_window_end=end,per_p=rows)
(case/'prefill-throughput.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
