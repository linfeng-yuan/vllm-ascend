import argparse
import collections
import csv
import json
import pathlib
import statistics
import time

from parse_result import number

parser = argparse.ArgumentParser()
parser.add_argument('case')
a = parser.parse_args()
case = pathlib.Path(a.case)
destination = case / 'stable-wave01'
summary = json.loads((destination / 'result-summary.json').read_text())
validation_path = case / 'kv-preemption-validation.json'
sample_interval = json.loads(validation_path.read_text()).get('sample_interval_seconds', 1) if validation_path.exists() else 1
window = json.loads((destination / 'steady-window.json').read_text())
# AISBench use_timestamp=False stores monotonic/perf_counter timestamps.
# Metrics samples use Unix wall time. Both come from the same node 22 kernel.
offset = time.time() - time.perf_counter()
start = window['absolute_start'] + offset
end = window['absolute_end'] + offset
measurement = json.loads((case / 'window.json').read_text())
assert measurement['started'] < start < end < measurement['ended'], (measurement, start, end)
grouped = collections.defaultdict(list)
for line in (case / 'samples.jsonl').open():
    row = json.loads(line)
    if row.get('role') == 'D' and 'error' not in row and start <= row['time'] <= end:
        grouped[row['tick']].append(row)
dp_count = len(json.loads((case / 'monitor-summary.json').read_text())['per_dp'])
complete = [rows for rows in grouped.values() if len(rows) == dp_count]
assert complete
per_dp = []
for rank in sorted({s['rank'] for rows in complete for s in rows}):
    values = [s['values']['num_requests_running'] for rows in complete for s in rows if s['rank'] == rank]
    per_dp.append({'rank': rank, 'running_mean': statistics.mean(values), 'running_min': min(values), 'running_max': max(values)})
totals = [sum(s['values']['num_requests_running'] for s in rows) for rows in complete]
original_csv = next(case.glob('output/*/performances/pro0813/*.csv'))
with original_csv.open(newline='') as stream:
    e2el = next(row for row in csv.DictReader(stream) if row['Performance Parameters'] == 'E2EL')
max_e2el_ms = number(e2el['Max'])
result = {'window_start_unix_estimate': start, 'window_end_unix_estimate': end,
          'clock_offset_unix_minus_perf_counter': offset,
          'clock_alignment': 'same-host monotonic-to-wall offset measured during analysis; %g-second metrics sampling' % sample_interval,
          'sample_interval_seconds': sample_interval,
          'samples': len(complete), 'dp_count': dp_count,
          'd_total_running_mean': statistics.mean(totals),
          'per_dp_running_mean': statistics.mean(totals) / dp_count,
          'd_total_running_median': statistics.median(totals),
          'd_total_running_peak': max(totals), 'per_dp': per_dp,
          'full_test_max_e2el_ms': max_e2el_ms,
          'stable_duration_ms': window['duration_seconds'] * 1000,
          'aisbench_doc_duration_confidence_condition_met': max_e2el_ms < window['duration_seconds'] * 1000 / 3}
(destination / 'steady-monitor.json').write_text(json.dumps(result, indent=2))
summary['steady_running'] = {k: v for k, v in result.items() if k != 'per_dp'}
summary['original_monitor_fields_scope'] = 'whole source test, not only the stable stage'
(destination / 'result-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary['steady_running'], indent=2))
