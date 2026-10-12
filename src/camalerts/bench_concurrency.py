"""concurrency: closed-loop load at several levels, then cost per 1,000 alerts.

    python -m camalerts.bench_concurrency --base-url http://localhost:30000 --variant <id> [--n 96]

At level C, C workers each send requests back to back until the level's total is done.
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import httpx

from camalerts import client as api
from camalerts import common, cost
from camalerts.common import Bench, mean, percentile

TEST = "concurrency"
NOT_SET = "not set"


async def run_level(
    http: httpx.AsyncClient,
    bench: Bench,
    work: Sequence[common.Prepared],
    concurrency: int,
    total: int,
) -> tuple[list[dict[str, Any]], float]:
    """Send `total` requests from `concurrency` workers. Returns the records and the wall time."""
    counter = itertools.count()
    records: list[dict[str, Any]] = []

    async def worker(worker_id: int) -> None:
        while (i := next(counter)) < total:
            record = await common.send(http, bench, work[i % len(work)])
            records.append({**record, "concurrency": concurrency, "worker": worker_id})

    start = time.perf_counter()
    await asyncio.gather(*(worker(w) for w in range(concurrency)))
    return records, time.perf_counter() - start


async def run(bench: Bench, frames: Sequence[Any]) -> dict[str, Any]:
    cfg = bench.cfg
    gpu_price = bench.prices.get("gpu_usd_per_hour")
    # One question per frame. Asking every question of a frame back to back would let a single
    # worker reuse the cached image for two requests in three, while many workers send the three
    # together and cannot, so low and high levels would not see the same cache state.
    tasks = bench.alerts.tasks
    work = [common.prepare(bench, frame, tasks[i % len(tasks)]) for i, frame in enumerate(frames)]
    records: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []

    async with api.make_client(cfg["timeout_s"], max_connections=max(cfg["levels"]) + 4) as http:
        for level in cfg["levels"]:
            total = max(cfg["min_requests"], cfg["requests_per_worker"] * level)
            # The last prompt is the warm-up because the level reaches it last, if at all.
            records.append(await common.fresh_start(http, bench, work[-1]))
            rows, wall_s = await run_level(http, bench, work, level, total)
            records += rows
            good = [r for r in rows if r["ok"]]
            rps = len(good) / wall_s if wall_s > 0 else None
            levels.append(
                {
                    "concurrency": level,
                    **common.counts(rows),
                    "repeated_prompts": max(0, total - len(work)),
                    "wall_s": wall_s,
                    "requests_per_s": rps,
                    "output_tokens_per_s": sum(r["completion_tokens"] or 0 for r in good) / wall_s,
                    "p50_total_s": percentile((r["total_s"] for r in good), 50),
                    "p95_total_s": percentile((r["total_s"] for r in good), 95),
                    "p50_ttft_s": percentile((r["ttft_s"] for r in good), 50),
                    "p95_ttft_s": percentile((r["ttft_s"] for r in good), 95),
                    **common.cache_use(rows),
                    "usd_per_1k_alerts": cost.self_hosted_usd_per_1k_alerts(gpu_price, rps),
                }
            )

    good = common.answered(records)
    (width, height), _ = Counter((p.info["width"], p.info["height"]) for p in work).most_common(1)[
        0
    ]
    summary = {
        "levels": levels,
        "distinct_prompts": len(work),
        "cost": {
            "gpu_usd_per_hour": gpu_price if gpu_price is not None else NOT_SET,
            "cloud_reference": cost.cloud_reference(
                bench.prices["cloud"],
                width=width,
                height=height,
                prompt_chars=round(mean(p.info["prompt_chars"] for p in work) or 0),
                output_tokens=mean(r["completion_tokens"] for r in good),
            ),
        },
    }
    return common.write_results(bench, frames, summary, records)


def usd(value: float | None) -> str:
    return NOT_SET if value is None else f"{value:.4f}"


def table(result: dict[str, Any]) -> str:
    summary = result["summary"]
    rows = [
        [
            s["concurrency"],
            s["n"],
            s["repeated_prompts"],
            s["errors"],
            common.num(s["requests_per_s"], 2),
            common.num(s["output_tokens_per_s"]),
            common.ms(s["p50_total_s"]),
            common.ms(s["p95_total_s"]),
            common.ms(s["p50_ttft_s"]),
            common.ms(s["p95_ttft_s"]),
            usd(s["usd_per_1k_alerts"]),
        ]
        for s in summary["levels"]
    ]
    headers = [
        "C", "n", "repeats", "errors", "req/s", "out tok/s", "p50 total ms", "p95 total ms",
        "p50 TTFT ms", "p95 TTFT ms", "USD per 1k alerts",
    ]  # fmt: skip
    cloud = summary["cost"]["cloud_reference"]
    price = summary["cost"]["gpu_usd_per_hour"]
    lines = [common.format_table(headers, rows)]
    if price == NOT_SET:
        lines.append("GPU price: not set (prices.gpu_usd_per_hour in the bench config)")
    else:
        lines.append(f"GPU price: {price} USD/hour")
    lines.append(
        f"Cloud reference ({cloud['as_of']}, {cloud['image_px']} frame = {cloud['image_tokens']} "
        f"image tokens + ~{cloud['prompt_text_tokens_estimated']} text tokens in, "
        f"{common.num(cloud['output_tokens'])} out):"
    )
    lines += [
        f"  {name}: {common.num(m['usd_per_1k_alerts'], 4)} USD per 1k alerts"
        for name, m in cloud["models"].items()
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    return common.run_cli(TEST, run, table, argv)


if __name__ == "__main__":
    sys.exit(main())
