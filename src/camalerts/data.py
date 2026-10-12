"""Manifest reader, plus the small helpers shared by the data scripts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_MANIFEST = "data/manifest.jsonl"
DEFAULT_CONFIG = "configs/data.yaml"


@dataclass(frozen=True)
class Frame:
    frame: str
    video: str
    label: str
    t_sec: float
    width: int
    height: int
    source: str
    in_event: bool | None = None


_FIELDS = {f.name for f in fields(Frame)}


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST,
    labels: list[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[Frame]:
    """Read the manifest. `limit` samples evenly across labels, deterministically for a seed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found at '{path}' (looked in {Path.cwd()}). Create it from the "
            "repository root with:\n"
            "  uv run scripts/download_data.py\n"
            "  uv run scripts/extract_frames.py"
        )

    frames: list[Frame] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frames.append(Frame(**{k: v for k, v in record.items() if k in _FIELDS}))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: bad manifest line: {exc}") from exc
    if not frames:
        raise ValueError(f"Manifest '{path}' has no frames. Re-run scripts/extract_frames.py.")

    if labels is not None:
        present = sorted({f.label for f in frames})
        unknown = sorted(set(labels) - set(present))
        if unknown:
            raise ValueError(f"Labels {unknown} are not in '{path}'. Labels present: {present}")
        frames = [f for f in frames if f.label in labels]

    if limit is not None:
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        frames = _sample_evenly(frames, limit, seed)
    return frames


def _sample_evenly(frames: list[Frame], limit: int, seed: int) -> list[Frame]:
    """Take frames label by label in turn, so a short label passes its share to the others."""
    rng = random.Random(seed)
    pools: dict[str, list[int]] = {}
    for i, f in enumerate(frames):
        pools.setdefault(f.label, []).append(i)
    queues = [rng.sample(pools[label], len(pools[label])) for label in sorted(pools)]

    picked: list[int] = []
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop())
    return [frames[i] for i in sorted(picked)]


def load_data_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data config not found at '{path}'.")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in ("source", "labels", "max_total_gb", "frames") if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    return cfg


def parse_temporal_annotations(text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse UCF-Crime's Temporal_Anomaly_Annotation_for_Testing_Videos.txt.

    Each line is `<video>.mp4 <Class> <start1> <end1> <start2> <end2>` in frame numbers, with -1
    for "no event". Returns video stem -> list of (start, end); normal videos map to [].
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        numbers = [int(p) for p in parts[2:]]
        pairs = zip(numbers[0::2], numbers[1::2])
        out[Path(parts[0]).stem] = [(s, e) for s, e in pairs if s >= 0 and e >= s]
    return out
