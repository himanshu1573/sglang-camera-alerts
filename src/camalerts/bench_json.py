"""json_validity: the same frames and questions with and without constrained decoding.

python -m camalerts.bench_json --base-url http://localhost:30000 --variant <id> [--n 40]
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from camalerts import client as api
from camalerts import common, scoring
from camalerts.common import Bench, mean

TEST = "json_validity"


async def run(bench: Bench, frames: Sequence[Any]) -> dict[str, Any]:
    cfg = bench.cfg
    tasks = bench.alerts.tasks
    modes: dict[str, dict[str, Any]] = cfg["modes"]

    def prepared(mode: str, frame: Any, task: common.Task) -> common.Prepared:
        opts = modes[mode]
        return common.prepare(
            bench,
            frame,
            task,
            instructions=opts["instructions"],
            constrained=opts["response_format"],
            mode=mode,
        )

    # Alternate which mode goes first for each frame, so neither always meets a warmer cache.
    order = list(modes)
    work = [
        prepared(mode, frame, task)
        for i, frame in enumerate(frames)
        for task in tasks
        for mode in (order if i % 2 == 0 else order[::-1])
    ]

    async with api.make_client(cfg["timeout_s"]) as http:
        records = [await common.fresh_start(http, bench, prepared(order[0], frames[0], tasks[0]))]
        for mode in order[1:]:  # one unmeasured request per mode, e.g. to compile its grammar
            warm = await common.send(http, bench, prepared(mode, frames[0], tasks[0]))
            records.append({**warm, "warmup": True})
        records += await common.run_pool(
            work, lambda p: common.send(http, bench, p), cfg["concurrency"]
        )

    scored_ids = [t.id for t in bench.alerts.scored_tasks]
    summary: dict[str, Any] = {"modes": {}}
    for mode in order:
        rows = [r for r in common.measured(records) if r["mode"] == mode]
        good = common.answered(rows)
        summary["modes"][mode] = {
            **common.counts(rows),
            **common.validity(rows),
            "response_format": modes[mode]["response_format"],
            "instructions": modes[mode]["instructions"],
            "mean_completion_tokens": mean(r["completion_tokens"] for r in good),
            "mean_total_s": mean(r["total_s"] for r in good),
            "truncated": sum(r["finish_reason"] == "length" for r in good),
            # Scored on the object extracted from the answer, so a fenced answer still counts.
            "accuracy": scoring.score_records(rows, scored_ids),
        }
    return common.write_results(bench, frames, summary, records)


def table(result: dict[str, Any]) -> str:
    rows = [
        [
            mode,
            s["n"],
            s["errors"],
            common.pct(s["strict_json_rate"]),
            common.pct(s["schema_valid_rate"]),
            common.pct(s["loose_json_rate"]),
            common.num(s["mean_completion_tokens"]),
            common.ms(s["mean_total_s"]),
            common.accuracy_cell(s["accuracy"]),
        ]
        for mode, s in result["summary"]["modes"].items()
    ]
    headers = [
        "mode", "n", "errors", "strict JSON", "schema valid", "loose JSON",
        "mean out tokens", "mean total ms", "accuracy",
    ]  # fmt: skip
    return common.format_table(headers, rows)


def main(argv: Sequence[str] | None = None) -> int:
    return common.run_cli(TEST, run, table, argv)


if __name__ == "__main__":
    sys.exit(main())
