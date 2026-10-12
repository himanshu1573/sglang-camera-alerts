"""Shared plumbing for the benchmarks: configs, statistics, request building, result files, CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import zlib
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from camalerts import client as api
from camalerts import prompts, scoring
from camalerts.schema import check_answer

NA = "n/a"


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    positive_labels: tuple[str, ...]
    scored: bool


@dataclass(frozen=True)
class Alerts:
    tasks: tuple[Task, ...]
    schema: dict[str, Any]
    schema_name: str
    site_brief: str
    text_scenes: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def scored_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.scored]

    def task(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(f"no alert task {task_id!r}; have {[t.id for t in self.tasks]}")


@dataclass
class Bench:
    """Everything one benchmark run needs. Build it with `connect`."""

    test: str
    variant: str
    base_url: str
    server: api.ServerInfo
    alerts: Alerts
    cfg: dict[str, Any]  # `request` defaults merged with this test's section of bench.yaml
    prices: dict[str, Any]
    results_dir: Path
    frames_root: Path | None = None  # base for relative frame paths; None = current directory
    flushes: list[bool] = field(default_factory=list)


@dataclass
class Prepared:
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None
    info: dict[str, Any]  # copied into the request's record


# --- configs -----------------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_alerts(path: str | Path = "configs/alerts.yaml") -> Alerts:
    raw = load_yaml(path)
    tasks = tuple(
        Task(
            id=t["id"],
            question=t["question"],
            positive_labels=tuple(t.get("positive_labels") or ()),
            scored=bool(t.get("scored", True)),
        )
        for t in raw["tasks"]
    )
    scenes = {label: tuple(texts) for label, texts in (raw.get("text_scenes") or {}).items()}
    return Alerts(
        tasks, raw["schema"], raw.get("schema_name", "camera_alert"), raw["site_brief"], scenes
    )


def scene_for(frame: Any, alerts: Alerts) -> str:
    """Stand-in text for a frame in text-only mode, consistent with the frame's ground truth."""
    label = frame.label
    if getattr(frame, "in_event", None) is False or label not in alerts.text_scenes:
        label = "normal"
    options = alerts.text_scenes.get(label)
    if not options:
        raise ValueError("text-only mode needs `text_scenes` with a `normal` entry in alerts.yaml")
    return options[zlib.crc32(str(frame.frame).encode()) % len(options)]


def section_config(config: dict[str, Any], test: str) -> dict[str, Any]:
    return {**config["request"], **config[test]}


async def connect(
    test: str,
    *,
    base_url: str,
    variant: str,
    results_dir: str | Path,
    alerts: Alerts,
    config: dict[str, Any],
    frames_root: Path | None = None,
) -> Bench:
    server = await api.get_server_info(base_url)
    if not server.ok:
        raise ConnectionError(f"cannot read {base_url}/v1/models: {server.error}")
    return Bench(
        test=test,
        variant=variant,
        base_url=base_url,
        server=server,
        alerts=alerts,
        cfg=section_config(config, test),
        prices=config["prices"],
        results_dir=Path(results_dir),
        frames_root=frames_root,
    )


# --- statistics --------------------------------------------------------------------------------


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: Iterable[float | None], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]. None for no data."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    pos = (len(vals) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(vals) - 1)
    return vals[low] + (vals[high] - vals[low]) * (pos - low)


def rate(flags: Iterable[bool]) -> float | None:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def measured(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if not r.get("warmup")]


def answered(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in measured(records) if r["ok"]]


def validity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """JSON validity rates over the requests that got an answer."""
    good = answered(records)
    return {
        "strict_json_rate": rate(r["strict_json"] for r in good),
        "schema_valid_rate": rate(r["schema_valid"] for r in good),
        "loose_json_rate": rate(r["loose_json"] for r in good),
    }


def cache_use(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cached prompt tokens over the answered requests.

    SGLang leaves `cached_tokens` out of the usage when it is zero, so a missing value counts
    as 0 here. A server that never reports the field therefore shows 0 hits; that is not proof
    that nothing was cached.
    """
    good = answered(records)
    return {
        "mean_cached_tokens": mean((r["cached_tokens"] or 0) for r in good),
        "cache_hits": sum(bool(r["cached_tokens"]) for r in good),
    }


def counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    rows = measured(records)
    return {"n": len(rows), "errors": sum(not r["ok"] for r in rows)}


# --- requests ----------------------------------------------------------------------------------


def prepare(
    bench: Bench,
    frame: Any,
    task: Task,
    *,
    ordering: str = "shared_first",
    instructions: str = "schema",
    constrained: bool = True,
    max_side: int | None = None,
    unique_line: str | None = None,
    **tags: Any,
) -> Prepared:
    """Encode one frame and build the request for one task. `tags` are copied into the record."""
    image = None
    if not bench.cfg.get("text_only"):
        path = Path(frame.frame)
        if bench.frames_root is not None and not path.is_absolute():
            path = bench.frames_root / path
        image = prompts.encode_frame(path, max_side, bench.cfg["jpeg_quality"])
    messages = prompts.build_messages(
        site_brief=bench.alerts.site_brief,
        schema=bench.alerts.schema,
        question=task.question,
        image_b64=image.b64 if image else None,
        ordering=ordering,
        instructions=instructions,
        unique_line=unique_line,
        scene_text=None if image else scene_for(frame, bench.alerts),
    )
    truth, source = scoring.ground_truth(frame, task.positive_labels)
    info = {
        "frame": frame.frame,
        "label": frame.label,
        "task": task.id,
        "ordering": ordering,
        "instructions": instructions,
        "constrained": constrained,
        "max_side": max_side,
        "truth": truth if task.scored else None,
        "truth_source": source,
        "width": image.width if image else None,
        "height": image.height if image else None,
        "prompt_chars": len(messages[0]["content"]) + len(messages[1]["content"][-1]["text"]),
        **tags,
    }
    fmt = prompts.response_format(bench.alerts.schema, bench.alerts.schema_name)
    return Prepared(messages, fmt if constrained else None, info)


async def send(http: httpx.AsyncClient, bench: Bench, prepared: Prepared) -> dict[str, Any]:
    """Send one prepared request and return its record: tags, timings, tokens, validity, answer."""
    res = await api.chat(
        http,
        bench.base_url,
        bench.server.model_id or "",
        prepared.messages,
        max_tokens=bench.cfg["max_tokens"],
        temperature=bench.cfg["temperature"],
        response_format=prepared.response_format,
        extra_body=bench.cfg.get("extra_body"),
    )
    check = check_answer(res.text, bench.alerts.schema)
    return {
        **prepared.info,
        **asdict(res),
        "strict_json": check.strict_json,
        "schema_valid": check.schema_valid,
        "loose_json": check.loose_json,
        "predicted": scoring.predicted_event(check.parsed),
    }


async def fresh_start(http: httpx.AsyncClient, bench: Bench, warmup: Prepared) -> dict[str, Any]:
    """Flush the server's prefix cache if it lets us, then send one request that is not measured.

    Without this a block would inherit cached prefixes (including whole frames) from the block or
    benchmark before it. The warm-up puts the shared site brief back and absorbs one-off costs.
    """
    if bench.cfg.get("flush_cache", True):
        bench.flushes.append(await api.flush_cache(bench.base_url))
    record = await send(http, bench, warmup)
    record["warmup"] = True
    return record


async def run_pool[T, R](
    items: Sequence[T], fn: Callable[[T], Awaitable[R]], concurrency: int
) -> list[R]:
    """Run fn over items with at most `concurrency` in flight. Results keep the order of items."""
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(item: T) -> R:
        async with gate:
            return await fn(item)

    return list(await asyncio.gather(*(one(item) for item in items)))


# --- results -----------------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_results(
    bench: Bench, frames: Sequence[Any], summary: dict[str, Any], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Write results/<variant>/<test>.json and raw/<test>.jsonl. Returns the summary file content."""
    rows = measured(records)
    if not bench.cfg.get("flush_cache", True):
        flush = "off"
    else:
        flush = "ok" if bench.flushes and all(bench.flushes) else "unsupported"
    meta = {
        "test": bench.test,
        "variant": bench.variant,
        "model": bench.server.model_id,
        "base_url": bench.base_url,
        "timestamp_utc": utc_now(),
        "mock": bench.server.mock,
        "text_only": bool(bench.cfg.get("text_only")),
        "n_frames": len(frames),
        "n_requests": len(rows),
        "errors": sum(not r["ok"] for r in rows),
        # weak = ground truth is the video's class, because the frame has no in_event annotation
        "labels": dict(
            Counter(
                scoring.WEAK if getattr(f, "in_event", None) is None else scoring.IN_EVENT
                for f in frames
            )
        ),
        "usage_source": dict(Counter(r["usage_source"] or "none" for r in rows)),
        "cache_flush": flush,
        "config": bench.cfg,
    }
    result = {"meta": meta, "summary": summary}
    out = bench.results_dir / bench.variant
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / f"{bench.test}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (out / "raw" / f"{bench.test}.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return result


# --- text output -------------------------------------------------------------------------------


def ms(seconds: float | None) -> str:
    return NA if seconds is None else f"{seconds * 1000:.0f}"


def pct(fraction: float | None) -> str:
    return NA if fraction is None else f"{fraction * 100:.1f}%"


def num(value: float | None, digits: int = 1) -> str:
    return NA if value is None else f"{value:.{digits}f}"


def accuracy_cell(scores: dict[str, Any]) -> str:
    """One cell for the per-task scores.

    Accuracy alone flatters a model that never alerts, because most frames are negatives. So the
    cell also shows recall and precision as counts, and how many answers could be scored at all:
    `concealment: acc 82.5%, recall 5/8, precision 5/7, scored 40/40`.
    """

    def one(task: str, s: dict[str, Any]) -> str:
        positives = s["tp"] + s["fn"]
        alerts = s["tp"] + s["fp"]
        return (
            f"{task}: acc {pct(s['accuracy'])}, recall {s['tp']}/{positives}, "
            f"precision {s['tp']}/{alerts}, scored {s['scored']}/{s['n']}"
        )

    return "; ".join(one(task, s) for task, s in scores.items()) or NA


def format_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], markdown: bool = False
) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    if markdown:
        lines = ["| " + " | ".join(row) + " |" for row in cells]
        lines.insert(1, "|" + "|".join(" --- " for _ in headers) + "|")
        return "\n".join(lines)
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


# --- command line ------------------------------------------------------------------------------


def build_parser(test: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {test} benchmark against a server.")
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--variant", required=True, help="results are written to <dir>/<variant>/")
    parser.add_argument("--n", type=int, help="number of frames (default: n_frames in the config)")
    parser.add_argument("--config", default="configs/bench.yaml")
    parser.add_argument("--alerts", default="configs/alerts.yaml")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=0, help="frame sampling seed")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="send a text scene description in place of each frame (for servers with no vision)",
    )
    return parser


def run_cli(
    test: str,
    run: Callable[[Bench, Sequence[Any]], Awaitable[dict[str, Any]]],
    table: Callable[[dict[str, Any]], str],
    argv: Sequence[str] | None = None,
) -> int:
    """Shared `main` for the benchmarks. Exit 0 only if at least one request was answered."""
    from camalerts.data import load_manifest  # written by the data owner; only the CLI needs it

    args = build_parser(test).parse_args(argv)
    config = load_yaml(args.config)
    if args.n is not None:
        config[test]["n_frames"] = args.n
    if args.text_only:
        config["request"]["text_only"] = True
    if config["request"].get("text_only") and test == "image_size":
        print("image_size: skipped, no image is sent in text-only mode")
        return 0
    try:
        frames = load_manifest(args.manifest, limit=config[test]["n_frames"], seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    async def go() -> dict[str, Any]:
        bench = await connect(
            test,
            base_url=args.base_url,
            variant=args.variant,
            results_dir=args.results_dir,
            alerts=load_alerts(args.alerts),
            config=config,
        )
        return await run(bench, frames)

    try:
        result = asyncio.run(go())
    except ConnectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    meta = result["meta"]
    mock = "  ** MOCK SERVER - not a real measurement **" if meta["mock"] else ""
    text_only = "  ** TEXT-ONLY - no images sent **" if meta["text_only"] else ""
    print(f"\n{test}: variant={meta['variant']} model={meta['model']}{mock}{text_only}")
    print(f"{meta['n_frames']} frames, {meta['n_requests']} requests, {meta['errors']} errors")
    print(table(result))
    print(f"wrote {Path(args.results_dir) / args.variant / (test + '.json')}")
    return 0 if meta["errors"] < meta["n_requests"] else 1
