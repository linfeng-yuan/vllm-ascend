import argparse
import asyncio
import time
import json
import os
from pathlib import Path

import aiohttp

from bench_d import API, ENDPOINTS, FIRST, KV_FETCH, RUNNING, TOKENS, WAITING, save, send, snapshot, total

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=Path, help="JSONL with the frozen 128K messages in the input field")
parser.add_argument("--profile", action="store_true", help="Capture 10 Decode steps on rank 0 after measurement")
args = parser.parse_args()
TARGET_BATCH = 80
CONCURRENCY = 16 * TARGET_BATCH
OBSERVE_SECONDS = 6
PROMPT_TOKENS = 131102
OUTPUT_TOKENS = 2048
CACHEABLE_PREFIX_TOKENS = 131072
P_IP, D0_IP, D1_IP = (os.environ[key] for key in ("P_IP", "D0_IP", "D1_IP"))
PROXY = f"http://{D0_IP}:1999"
ENDPOINTS.clear()
ENDPOINTS.update({
    **{f"P{rank}": f"http://{P_IP}:{8000 + rank}" for rank in range(8)},
    **{f"D{rank}": f"http://{D0_IP if rank < 8 else D1_IP}:{8000 + rank % 8}"
       for rank in range(16)},
})
DRAFTS = "vllm:spec_decode_num_drafts_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
RUN_DIR = Path(os.environ["RUN_DIR"])
OUT = RUN_DIR / ("profile-bench" if args.profile else "pure-d-128k-2k")


def request(messages, output):
    return json.dumps(
        dict(model="dsv4_1", messages=messages, stream=True, max_tokens=output,
             temperature=0, ignore_eos=True, stream_options={"include_usage": True}),
        ensure_ascii=False, separators=(",", ":"),
    ).encode()


async def status(session, endpoint, path):
    async with session.get(endpoint + path, timeout=aiohttp.ClientTimeout(total=4)) as response:
        response.raise_for_status()


async def capture_profile(session):
    rank_url = ENDPOINTS["D0"]
    payload = {"profile_prefix": "pure-d-128k-2k-dp0-bs80", "max_iterations": 10}
    timeout = aiohttp.ClientTimeout(total=100)
    async with session.post(rank_url + "/start_profile", json=payload, timeout=timeout) as response:
        response.raise_for_status()
    try:
        await asyncio.sleep(2)
    finally:
        async with session.post(rank_url + "/stop_profile", timeout=timeout) as response:
            response.raise_for_status()


def occupancy(sample, before):
    p_work = sum(total(sample, "P", m) for m in (RUNNING, WAITING))
    d_running = total(sample, "D", RUNNING)
    running_by_rank = [sample["ranks"][f"D{i}"].get(RUNNING, 0) for i in range(16)]
    fetch = sum(total(sample, "D", f"{KV_FETCH}:{stage}") for stage in ("waiting_to_start", "in_progress"))
    completed = total(sample, "D", f"{KV_FETCH}:completed_waiting")
    d_started = total(sample, "D", FIRST) - total(before, "D", FIRST)
    d_output_tokens = total(sample, "D", TOKENS) - total(before, "D", TOKENS)
    return dict(time=sample["time"], wall_time=time.time(), p_running_or_waiting=p_work, d_running=d_running,
                d_running_by_rank=running_by_rank, d_kv_fetch=fetch, d_kv_fetch_completed=completed, d_first_tokens=d_started,
                d_output_tokens=d_output_tokens, per_rank=[
                    {metric: sample["ranks"][f"D{i}"].get(metric, 0.0)
                     for metric in (TOKENS, DRAFTS, ACCEPTED, "vllm:inter_token_latency_seconds_sum",
                                    "vllm:inter_token_latency_seconds_count")}
                    for i in range(16)
                ])


def summarize_steady_decode(samples):
    before, after = samples[0], samples[-1]
    duration = after["time"] - before["time"]
    ranks = []
    for rank, (start, stop) in enumerate(zip(before["per_rank"], after["per_rank"])):
        tokens = int(stop[TOKENS] - start[TOKENS])
        drafts = int(stop[DRAFTS] - start[DRAFTS])
        accepted = int(stop[ACCEPTED] - start[ACCEPTED])
        itl_count = stop["vllm:inter_token_latency_seconds_count"] - start["vllm:inter_token_latency_seconds_count"]
        itl_sum = stop["vllm:inter_token_latency_seconds_sum"] - start["vllm:inter_token_latency_seconds_sum"]
        ranks.append(dict(rank=rank, host=D0_IP if rank < 8 else D1_IP, avg_generation_tps=tokens / duration,
                          generated_tokens=tokens, draft_steps=drafts, accepted_draft_tokens=accepted,
                          mean_acceptance_length=1 + accepted / drafts if drafts else None,
                          mean_itl_ms=1000 * itl_sum / itl_count if itl_count else None))
    aggregate_tps = sum(row["avg_generation_tps"] for row in ranks)
    itl_count = itl_sum = 0
    for start, stop in zip(before["per_rank"], after["per_rank"]):
        itl_count += stop["vllm:inter_token_latency_seconds_count"] - start["vllm:inter_token_latency_seconds_count"]
        itl_sum += stop["vllm:inter_token_latency_seconds_sum"] - start["vllm:inter_token_latency_seconds_sum"]
    mean_itl_ms = 1000 * itl_sum / itl_count if itl_count else None
    return dict(target_batch=TARGET_BATCH, seconds=duration, sample_count=len(samples), mean_itl_ms=mean_itl_ms,
                start_wall_time=before["wall_time"], end_wall_time=after["wall_time"],
                running_by_rank=after["d_running_by_rank"], p_running_or_waiting=after["p_running_or_waiting"],
                outstanding_kv_fetch=after["d_kv_fetch"], aggregate_generation_tps=aggregate_tps,
                mean_per_dp_generation_tps=aggregate_tps / len(ranks), ranks=ranks)


async def run():
    with args.prompt.open(encoding="utf-8") as input_file:
        messages = json.loads(input_file.readline())["input"]
    timeout = aiohttp.ClientTimeout(total=1000, sock_read=240)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 40)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        for attempt in range(240):
            checks = await asyncio.gather(
                *(status(session, url, "/health") for url in ENDPOINTS.values()),
                status(session, PROXY, "/healthcheck"), return_exceptions=True,
            )
            ready = sum(result is None for result in checks)
            if ready == len(checks):
                break
            if any((RUN_DIR / node / "launcher.exit").exists() for node in ("p", "d0", "d1")):
                raise RuntimeError("The P or D launcher exited")
            if attempt % 6 == 0:
                print(f"readiness {ready}/{len(checks)}", flush=True)
            await asyncio.sleep(5)
        else:
            raise TimeoutError("P/D readiness timed out")

        hit_rates = []
        for attempt in range(2):
            before_warmup = await snapshot(session)
            results = await asyncio.gather(*(send(
                session, ENDPOINTS[f"P{rank}"], API, request(messages, 1), PROMPT_TOKENS, 1,
                f"pure-d-throughput-warm-{attempt}-{rank}",
            ) for rank in range(8)))
            await asyncio.sleep(2)
            after_warmup = await snapshot(session)
            hit_rates.append([
                int(after_warmup["ranks"][f"P{i}"].get("vllm:prefix_cache_hits_total", 0)
                    - before_warmup["ranks"][f"P{i}"].get("vllm:prefix_cache_hits_total", 0))
                for i in range(8)
            ])
            save(OUT / f"warmup-{attempt}.json", dict(results=results, cache_hits=hit_rates[-1]))
            if not all(row["success"] for row in results):
                raise RuntimeError(f"Prefill warmup failed on pass {attempt}")
        if hit_rates[-1] != [131072] * 8:
            raise RuntimeError(f"P cache hits differ from the 128K baseline: {hit_rates[-1]}")
        smoke = await send(session, PROXY, API, request(messages, 16), PROMPT_TOKENS, 16, "pure-d-throughput-smoke")
        save(OUT / "pd-smoke.json", smoke)
        if not smoke["success"]:
            raise RuntimeError(f"P/D smoke failed: {smoke}")
        print("P/D smoke passed and 131072 prefix tokens are cached on all P ranks.", flush=True)
        await asyncio.sleep(2)
        before = await snapshot(session, OUT / "before.prom")
        payload = request(messages, OUTPUT_TOKENS)
        clients = [asyncio.create_task(send(
            session, PROXY, API, payload, PROMPT_TOKENS, OUTPUT_TOKENS, f"pure-d-throughput-bs{TARGET_BATCH}-{i}",
        )) for i in range(CONCURRENCY)]
        window = []
        steady = []
        steady_summary = None
        try:
            for _ in range(180):
                current = occupancy(await snapshot(session), before)
                window.append(current)
                if len(window) % 10 == 1:
                    print(f"running={current['d_running_by_rank']} p_active={current['p_running_or_waiting']}"
                          f" kv_in_flight={current['d_kv_fetch']} first_tokens={current['d_first_tokens']}"
                          f" output_tokens={current['d_output_tokens']}", flush=True)
                qualifies = (
                    current["p_running_or_waiting"] == 0 and current["d_kv_fetch"] == 0
                    and current["d_first_tokens"] >= CONCURRENCY
                    and all(running == TARGET_BATCH for running in current["d_running_by_rank"])
                )
                if qualifies and (not steady or current["d_output_tokens"] > steady[-1]["d_output_tokens"]):
                    if not steady:
                        print(f"All 16 D ranks at running {TARGET_BATCH}. Beginning steady-state measurement.", flush=True)
                    steady.append(current)
                else:
                    steady.clear()
                if len(steady) >= 2 and current["time"] - steady[0]["time"] >= OBSERVE_SECONDS:
                    steady_summary = summarize_steady_decode(steady)
                    save(OUT / "steady-throughput.json", steady_summary)
                    print(f"Running {TARGET_BATCH}, steady-state mean {steady_summary['mean_per_dp_generation_tps']:.2f} tok/s/DP",
                          flush=True)
                    save(OUT / "decode-window.json", window)
                    if args.profile:
                        print("Capturing 10 Decode steps on rank 0.", flush=True)
                        await capture_profile(session)
                    break
                if all(task.done() for task in clients):
                    break
                await asyncio.sleep(1)
        finally:
            save(OUT / "decode-window.json", window)
            if steady_summary is None:
                for task in clients:
                    if not task.done():
                        task.cancel()
            results = await asyncio.gather(*clients, return_exceptions=True)
            save(OUT / "requests.json", [
                result if isinstance(result, dict) else dict(success=False, error=repr(result))
                for result in results
            ])
        if steady_summary is None:
            raise RuntimeError("No pure steady Decode window qualified")
        await asyncio.sleep(2)
        after = await snapshot(session, OUT / "after.prom")
        drafts = total(after, "D", DRAFTS) - total(before, "D", DRAFTS)
        accepted = total(after, "D", ACCEPTED) - total(before, "D", ACCEPTED)
        successes = sum(isinstance(row, dict) and row.get("success", False) for row in results)
        cache = {
            name: {
                metric: int(after["ranks"][name].get(metric, 0) - before["ranks"][name].get(metric, 0))
                for metric in (FIRST, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
            }
            for name in ENDPOINTS if name.startswith("P")
        }
        prefix_cached = sum(row[FIRST] for row in cache.values()) == CONCURRENCY and all(
            row["vllm:prefix_cache_hits_total"] == row[FIRST] * CACHEABLE_PREFIX_TOKENS
            for row in cache.values()
        )
        summary = dict(
            p_cache_by_rank=cache, all_cacheable_prefix_tokens_reused=prefix_cached,
            cacheable_prefix_tokens=CACHEABLE_PREFIX_TOKENS,
            successes=successes, requests=CONCURRENCY, input_tokens=PROMPT_TOKENS,
            output_tokens=OUTPUT_TOKENS, synthetic_target_length=3.51, accepted_draft_tokens=accepted,
            draft_steps=drafts, measured_length_including_bonus=1 + accepted / drafts if drafts else None,
            steady_measurement=steady_summary,
        )
        save(OUT / "summary.json", summary)
        print(summary, flush=True)
        if successes != CONCURRENCY:
            raise RuntimeError(f"{successes}/{CONCURRENCY} P/D requests succeeded")
        if not prefix_cached:
            raise RuntimeError("100% of cacheable P prefix tokens was not verified")


if __name__ == "__main__":
    OUT.mkdir(parents=True)
    try:
        asyncio.run(run())
    except Exception as exc:
        save(OUT / "error.json", {"error": repr(exc)})
        raise
