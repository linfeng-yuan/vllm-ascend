"""AISBench entry point with the requested steady-stage tolerance."""
import json
import os
import pathlib
import time

import ais_bench.benchmark.calculators.stable_perf_metric_calculator as stable

original_offset = stable.WAVE_OFFSET
stable.WAVE_OFFSET = 0.1
original_select = stable.StablePerfMetricCalculator._get_requests_id


def select_and_record(self, perf_details):
    ids = original_select(self, perf_details)
    path = os.environ.get('YLF_STEADY_WINDOW_JSON')
    if path:
        offset = time.time() - time.perf_counter()
        pathlib.Path(path).write_text(json.dumps({
            'original_wave_offset': original_offset,
            'effective_wave_offset': stable.WAVE_OFFSET,
            'client_max_concurrency': self.max_concurrency,
            'client_concurrency_tolerance_threshold': int(self.max_concurrency * (1 - stable.WAVE_OFFSET)),
            'absolute_start': self.stage_section[0], 'absolute_end': self.stage_section[1],
            'clock_domain': 'AISBench use_timestamp=False / perf_counter',
            'window_start_unix_estimate': self.stage_section[0] + offset,
            'window_end_unix_estimate': self.stage_section[1] + offset,
            'duration_seconds': self.stage_section[1] - self.stage_section[0],
            'selected_request_indices': ids,
        }, indent=2))
    return ids


stable.StablePerfMetricCalculator._get_requests_id = select_and_record

from ais_bench.benchmark.cli.main import main

if __name__ == '__main__':
    main()
