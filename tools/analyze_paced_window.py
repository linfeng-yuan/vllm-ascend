"""Measure actual arrivals and Running during sustained sending, without claiming an AISBench stable stage."""
import collections
import argparse
import io
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('--warmup-seconds',type=float,default=30)
args=parser.parse_args()
case=Path(args.case)
metadata=json.loads((case/'dataset.json').read_text())
measurement=json.loads((case/'window.json').read_text())
path=next(case.glob('output/*/performances/pro0813/pro0813-case_details.jsonl'))
connections={}
starts=[]
ends=[]
ttfts=[]
for line in path.open():
    z=json.loads(line);assert z['success']
    name=z['db_name']
    if name not in connections:connections[name]=sqlite3.connect('file:'+str(path.parent/'db_data'/name)+'?mode=ro',uri=True)
    b=connections[name].execute('select arr_blob from numpy_store where id=?',(z['time_points']['__db_ref__'],)).fetchone()[0]
    a=np.load(io.BytesIO(b),allow_pickle=False);assert len(a)>=3
    starts.append(float(a[0]));ends.append(float(a[-1]));ttfts.append(float(a[1]-a[0]))
for c in connections.values():c.close()
first,last=min(starts),max(starts)
final=max(ends)
events=sorted([(s,1) for s in starts]+[(e,-1) for e in ends])
active=peak=0
for _,change in events:active+=change;peak=max(peak,active)
assert active==0
offset=time.time()-time.perf_counter()
start=first+args.warmup_seconds+offset
end=last+offset
assert start<end
rows=[json.loads(s) for s in (case/'samples.jsonl').open()]
chosen=[z for z in rows if 'error' not in z and start<=z['time']<=end]
per=[]
for role,node,rank in sorted({(z['role'],z['node'],z['rank']) for z in chosen}):
    samples=[z for z in chosen if (z['role'],z['node'],z['rank'])==(role,node,rank)]
    per.append(dict(role=role,node=node,rank=rank,samples=len(samples),
                    running_mean=statistics.mean(z['values']['num_requests_running'] for z in samples),
                    running_max=max(z['values']['num_requests_running'] for z in samples),
                    waiting_mean=statistics.mean(z['values']['num_requests_waiting'] for z in samples),
                    waiting_max=max(z['values']['num_requests_waiting'] for z in samples)))
ticks=collections.defaultdict(list)
for z in chosen:
    if z['role']=='D':ticks[z['tick']].append(z)
totals=[sum(z['values']['num_requests_running'] for z in zs) for zs in ticks.values() if len(zs)==32]
assert len([z for z in per if z['role']=='D'])==32 and totals
result=dict(scope='sustained sending window: first request start + %g seconds through last request start; not an AISBench stable stage'%args.warmup_seconds,
            window_start_unix_estimate=start,window_end_unix_estimate=end,window_duration_seconds=end-start,
            clock_alignment='client request perf_counter and metrics collected on the same node22 kernel',
            aisbench_stable_stage_available=False,aisbench_stable_stage_reason='Paced traffic is reported using native full-test statistics; this separately labelled monitor window is not an AISBench stable stage',
            configured_concurrency_limit=measurement['concurrency'],configured_request_rate=metadata['request_rate'],requests=len(starts),actual_client_peak_inflight=peak,
            first_request_start_perf_counter=first,last_request_start_perf_counter=last,last_response_perf_counter=final,
            first_to_last_send_seconds=last-first,actual_send_rate=(len(starts)-1)/(last-first),
            response_tail_after_last_send_seconds=final-last,measurement_duration_seconds=final-first,
            ttft_avg_from_timestamps_ms=statistics.mean(ttfts)*1000,
            d_total_running_mean=statistics.mean(totals),d_total_running_peak=max(totals),per_dp=per)
(case/'paced-monitor.json').write_text(json.dumps(result,indent=2))
summary=json.loads((case/'result-summary.json').read_text())
summary['reported_max_concurrency_is_configured_limit']=True
summary['actual_client_peak_inflight']=peak
summary['actual_send_rate']=result['actual_send_rate']
summary['aisbench_stable_stage_available']=False
summary['sustained_sending_running_mean']=result['d_total_running_mean']
(case/'result-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='per_dp'},indent=2))
