import argparse
import collections
import csv
import json
import pathlib
import re


def number(value):
    if isinstance(value, (int, float)):
        return value
    match = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?', str(value))
    return float(match.group().replace(',', '')) if match else None


def parse(case):
    case = pathlib.Path(case)
    files = list(case.glob('output/*/performances/pro0813/pro0813-case.json'))
    assert len(files) == 1, files
    common = json.loads(files[0].read_text())
    csv_path = files[0].with_suffix('.csv')
    with csv_path.open(newline='') as stream:
        table = list(csv.reader(stream))
    header = next(row for row in table if row and 'Stage' in row)
    metrics = [dict(zip(header, row)) for row in table if row and row[0] in ['TPOT', 'TTFT', 'ITL', 'E2EL']]
    # Preserve every stage; the printed All stage is the comparison authority.
    tpot = [row for row in metrics if list(row.values())[0] == 'TPOT']
    tpot_row = next((row for row in tpot if str(row.get('Stage', '')).lower() in ['all', 'all stages', 'total']), tpot[0] if len(tpot) == 1 else None)
    assert tpot_row, tpot
    average_key = next((key for key in tpot_row if key.lower() in ['average', 'avg', 'mean']), None)
    assert average_key, header
    mean_tpot_ms = number(tpot_row[average_key])
    def common_metric(label):
        value = common[label]
        if isinstance(value, dict):
            all_keys = [key for key in value if key.lower() in ['all', 'all stages', 'total']]
            value = value[all_keys[0]] if all_keys else next(iter(value.values())) if len(value) == 1 else value
        return number(value)
    output_tps = common_metric('Output Token Throughput')
    counts = collections.Counter()
    inputs = collections.Counter()
    successes = collections.Counter()
    rows = 0
    with files[0].with_name('pro0813-case_details.jsonl').open() as stream:
        for line in stream:
            item = json.loads(line)
            rows += 1
            counts[item.get('output_tokens')] += 1
            inputs[item.get('input_tokens')] += 1
            successes[str(item.get('success'))] += 1
    expected = json.loads((case / 'dataset.json').read_text())
    cards = expected.get('hardware_cards', 64)
    monitor = json.loads((case / 'monitor-summary.json').read_text())
    ttft = [row for row in metrics if list(row.values())[0] == 'TTFT']
    ttft_row = next((row for row in ttft if str(row.get('Stage', '')).lower() in ['all', 'all stages', 'total']), ttft[0] if len(ttft) == 1 else None)
    assert ttft_row, ttft
    report_metrics = {
        'maximum_concurrency': common_metric('Max Concurrency'),
        'average_concurrency': common_metric('Concurrency'),
        'total_requests': common_metric('Total Requests'),
        'success_requests': common_metric('Success Requests'),
        'failed_requests': common_metric('Failed Requests'),
        'ttft_avg_ms': number(ttft_row[average_key]),
        'tpot_avg_ms': mean_tpot_ms,
        'input_tokens_per_second_per_card': common_metric('Input Token Throughput') / cards,
        'output_tokens_per_second_per_card': output_tps / cards,
        'end_to_end_tokens_per_second_per_card': common_metric('Total Token Throughput') / cards,
        'qps': common_metric('Request Throughput'),
        'hardware_cards': cards,
        'per_card_denominator': 'all active P and D cards',
        'end_to_end_definition': '(input tokens + output tokens) / benchmark duration / hardware cards',
    }
    summary = {
        'case': case.name, 'mean_tpot_ms': mean_tpot_ms,
        'aisbench_output_tps': output_tps,
        'hardware_cards': cards,
        'average_output_tps_per_card': output_tps / cards,
        'average_output_tps_per_card_%d' % cards: output_tps / cards,
        'report_metrics': report_metrics,
        'common_metrics_as_printed': common,
        'latency_metrics_as_printed': metrics,
        'actual_detail_rows': rows,
        'request_success_values': dict(successes),
        'input_token_counts': dict(inputs), 'output_token_counts': dict(counts),
        'valid_request_count': rows == expected['data_num'],
        'all_output_lengths_match': counts.get(expected['output_tokens'], 0) == expected['data_num'],
        'all_input_lengths_match': inputs.get(expected['input_tokens'], 0) == expected['data_num'],
        'all_successful': successes.get('True', 0) == expected['data_num'],
        'all_d_dp_sampled': monitor['all_dp_have_samples'],
        'd_success_delta': monitor['success_delta_total'],
        'd_generation_tokens_delta': monitor['generation_tokens_delta_total'],
        'd_running_peak_simultaneous': monitor['simultaneous_d_running_peak'],
        'proxy_actual_inflight_peak': monitor['actual_proxy_inflight_peak'],
        'common_result_path': str(files[0]), 'csv_result_path': str(csv_path),
    }
    (case / 'result-summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    args = parser.parse_args()
    print(json.dumps(parse(args.case), indent=2))
