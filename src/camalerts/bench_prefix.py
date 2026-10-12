"""prefix_cache: time to first token with the shared instructions first vs a unique line first.

python -m camalerts.bench_prefix --base-url http://localhost:30000 --variant <id> [--n 31]
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Sequence
from typing import Any

from camalerts import client as api
from camalerts import common, prompts
from camalerts.common import Bench, mean, percentile

TEST = "prefix_cache"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return numerator / denominator if numerator is not None and denominator else None


async def run(bench: Bench, frames: Sequence[Any]) -> dict[str, Any]:
    cfg = bench.cfg
    task = bench.alerts.task(cfg["task"])
    records: list[dict[str, Any]] = []

    async with api.make_client(cfg["timeout_s"]) as http:
        for ordering in cfg["orderings"]:
            # A fresh nonce every time, so a rerun against the same server cannot hit old entries.
            work = [
                common.prepare(
                    bench,
                    frame,
                    task,
                    ordering=ordering,
                    unique_line=prompts.request_line(
                        f"cam-{i % cfg['cameras'] + 1:02d}", uuid.uuid4().hex
                    ),
                )
                for i, frame in enumerate(frames)
            ]
            records.append(await common.fresh_start(http, bench, work[0]))
            # More than one warm-up: on a variant that runs only this test, the first requests
            # also pay one-off costs that have nothing to do with the prompt order.
            for _ in range(cfg.get("warmups", 1) - 1):
                warm = await common.send(http, bench, work[0])
                records.append({**warm, "warmup": True})
            for item in work[1:]:  # strictly one at a time
                records.append(await common.send(http, bench, item))

    summary: dict[str, Any] = {"orderings": {}}
    for ordering in cfg["orderings"]:
        rows = [r for r in common.measured(records) if r["ordering"] == ordering]
        good = common.answered(rows)
        ttft = [r["ttft_s"] for r in good]
        summary["orderings"][ordering] = {
            **common.counts(rows),
            "mean_ttft_s": mean(ttft),
            "p50_ttft_s": percentile(ttft, 50),
            "p95_ttft_s": percentile(ttft, 95),
            "mean_total_s": mean(r["total_s"] for r in good),
            "mean_prompt_tokens": mean(r["prompt_tokens"] for r in good),
            **common.cache_use(rows),
        }
    by = summary["orderings"]
    if {"shared_first", "unique_first"} <= by.keys():
        shared, unique = by["shared_first"], by["unique_first"]
        # Above 1 means the unique line at the start made the first token slower.
        summary["ttft_ratio_unique_over_shared"] = {
            "mean": _ratio(unique["mean_ttft_s"], shared["mean_ttft_s"]),
            "p50": _ratio(unique["p50_ttft_s"], shared["p50_ttft_s"]),
        }
    return common.write_results(bench, frames, summary, records)


def table(result: dict[str, Any]) -> str:
    summary = result["summary"]
    rows = [
        [
            ordering,
            s["n"],
            s["errors"],
            common.ms(s["mean_ttft_s"]),
            common.ms(s["p50_ttft_s"]),
            common.ms(s["p95_ttft_s"]),
            common.ms(s["mean_total_s"]),
            common.num(s["mean_prompt_tokens"]),
            common.num(s["mean_cached_tokens"]),
            f"{s['cache_hits']}/{s['n'] - s['errors']}",
        ]
        for ordering, s in summary["orderings"].items()
    ]
    headers = [
        "ordering", "n", "errors", "mean TTFT ms", "p50 TTFT ms", "p95 TTFT ms",
        "mean total ms", "prompt tokens", "cached tokens", "cache hits",
    ]  # fmt: skip
    out = common.format_table(headers, rows)
    ratio = summary.get("ttft_ratio_unique_over_shared")
    if ratio:
        out += (
            f"\nTTFT unique_first / shared_first: mean {common.num(ratio['mean'], 2)}x, "
            f"p50 {common.num(ratio['p50'], 2)}x"
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    return common.run_cli(TEST, run, table, argv)


if __name__ == "__main__":
    sys.exit(main())
