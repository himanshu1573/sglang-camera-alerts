"""image_size: the same frames sent at several maximum side lengths.

    python -m camalerts.bench_image_size --base-url http://localhost:30000 --variant <id> [--n 60]

Each frame gets one question, rotating over the scored tasks, so every request carries an image
the server has not seen and the timings include the full cost of that image.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence
from typing import Any

from camalerts import client as api
from camalerts import common, scoring
from camalerts.common import Bench, mean

TEST = "image_size"


def size_name(max_side: int | None) -> str:
    return "native" if max_side is None else str(max_side)


async def run(bench: Bench, frames: Sequence[Any]) -> dict[str, Any]:
    cfg = bench.cfg
    tasks = bench.alerts.scored_tasks or list(bench.alerts.tasks)
    records: list[dict[str, Any]] = []

    async with api.make_client(cfg["timeout_s"]) as http:
        for max_side in cfg["max_sides"]:
            work = [
                common.prepare(bench, frame, tasks[i % len(tasks)], max_side=max_side)
                for i, frame in enumerate(frames)
            ]
            # The first frame of each size is the warm-up and is not measured.
            records.append(await common.fresh_start(http, bench, work[0]))
            records += await common.run_pool(
                work[1:], lambda p: common.send(http, bench, p), cfg["concurrency"]
            )

    summary: dict[str, Any] = {"sizes": []}
    for max_side in cfg["max_sides"]:
        rows = [r for r in common.measured(records) if r["max_side"] == max_side]
        good = common.answered(rows)
        sizes = Counter((r["width"], r["height"]) for r in rows).most_common(1)
        width, height = sizes[0][0] if sizes else (None, None)
        summary["sizes"].append(
            {
                "max_side": size_name(max_side),
                **common.counts(rows),
                "actual_px": f"{width}x{height}" if sizes else None,  # the most common size sent
                "mean_width": mean(r["width"] for r in rows),
                "mean_height": mean(r["height"] for r in rows),
                "mean_prompt_tokens": mean(r["prompt_tokens"] for r in good),
                "mean_ttft_s": mean(r["ttft_s"] for r in good),
                "mean_total_s": mean(r["total_s"] for r in good),
                "schema_valid_rate": common.validity(rows)["schema_valid_rate"],
                "accuracy": scoring.score_records(rows, [t.id for t in tasks if t.scored]),
            }
        )
    return common.write_results(bench, frames, summary, records)


def table(result: dict[str, Any]) -> str:
    rows = [
        [
            s["max_side"],
            s["actual_px"],
            s["n"],
            s["errors"],
            common.num(s["mean_prompt_tokens"]),
            common.ms(s["mean_ttft_s"]),
            common.ms(s["mean_total_s"]),
            common.pct(s["schema_valid_rate"]),
            common.accuracy_cell(s["accuracy"]),
        ]
        for s in result["summary"]["sizes"]
    ]
    headers = [
        "max side", "sent px", "n", "errors", "prompt tokens", "mean TTFT ms",
        "mean total ms", "schema valid", "accuracy",
    ]  # fmt: skip
    return common.format_table(headers, rows)


def main(argv: Sequence[str] | None = None) -> int:
    return common.run_cli(TEST, run, table, argv)


if __name__ == "__main__":
    sys.exit(main())
