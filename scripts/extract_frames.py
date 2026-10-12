"""Sample JPEG frames from the downloaded videos and write data/manifest.jsonl.

  uv run scripts/extract_frames.py

Reads data/videos/<label>/*.mp4, writes data/frames/<label>/<video>/<frame_index:06d>.jpg.
The manifest format is described in docs/DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from camalerts.data import DEFAULT_CONFIG, load_data_config, parse_temporal_annotations

ROOT = Path(__file__).resolve().parents[1]
Ranges = list[tuple[int, int]]


@dataclass(frozen=True)
class Sampling:
    every_sec: float = 5.0
    skip_sec: float = 2.0
    max_per_video: int = 12
    jpeg_quality: int = 90
    event_fraction: float = 0.5
    event_min_gap_sec: float = 1.0


def spread(items: list[int], n: int) -> list[int]:
    """At most n items, evenly spaced over the whole list."""
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    return [items[int((k + 0.5) * len(items) / n)] for k in range(n)]


def in_any(index: int, ranges: Ranges) -> bool:
    return any(start <= index <= end for start, end in ranges)


def plan_frame_indices(
    n_frames: int, fps: float, sampling: Sampling, ranges: Ranges | None = None
) -> list[int]:
    """Frame indices to save. `ranges` are annotated event ranges (inclusive frame numbers).

    Without ranges: one frame every `every_sec`, capped at `max_per_video` spread over the video.
    With ranges: about `event_fraction` of the frames come from inside the ranges, the rest from
    the regular grid outside them, so an annotated video yields both positives and negatives.
    """
    grid = []
    k = 0
    while (index := round((sampling.skip_sec + k * sampling.every_sec) * fps)) < n_frames:
        grid.append(index)
        k += 1
    if not ranges:
        return spread(grid, sampling.max_per_video)

    event = [i for s, e in ranges for i in range(max(s, 0), min(e, n_frames - 1) + 1)]
    min_gap = max(1, round(sampling.event_min_gap_sec * fps))
    n_inside = min(round(sampling.max_per_video * sampling.event_fraction), len(event) // min_gap)
    inside = spread(event, max(n_inside, 1)) if event else []
    outside = [i for i in grid if not in_any(i, ranges)]
    return sorted(set(inside + spread(outside, sampling.max_per_video - len(inside))))


def event_ranges(video: str, anomaly: bool, annotations: dict[str, Ranges]) -> Ranges | None:
    """[] for a normal video (no event anywhere), None when an anomaly video has no annotation."""
    if not anomaly:
        return []
    return annotations.get(video)


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def extract_video(
    video_path: Path,
    label: str,
    out_dir: Path,
    root: Path,
    sampling: Sampling,
    ranges: Ranges | None,
    source: str,
) -> list[dict]:
    """Write the sampled frames of one video and return their manifest records."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not fps > 0 or n_frames <= 0:
            raise ValueError(f"cannot read video (fps={fps}, frames={n_frames})")

        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index in plan_frame_indices(n_frames, fps, sampling, ranges):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:  # the container's frame count can overstate what is decodable
                continue
            path = out_dir / f"{index:06d}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, sampling.jpeg_quality]):
                raise ValueError(f"cannot write {path}")
            records.append(
                {
                    "frame": relative_to_root(path, root),
                    "video": video_path.stem,
                    "label": label,
                    "t_sec": round(index / fps, 3),
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "source": source,
                    "in_event": None if ranges is None else in_any(index, ranges),
                }
            )
    finally:
        cap.release()

    keep = {Path(r["frame"]).name for r in records}
    for stale in out_dir.glob("*.jpg"):  # frames left over from a run with other settings
        if stale.name not in keep:
            stale.unlink()
    return records


def extract_all(
    videos_dir: Path,
    frames_dir: Path,
    manifest_path: Path,
    root: Path,
    cfg: dict,
    annotations: dict[str, Ranges],
) -> list[dict]:
    sampling = Sampling(**cfg["frames"])
    source = cfg["source"]["name"]
    records: list[dict] = []
    for label, spec in cfg["labels"].items():
        videos = sorted((videos_dir / label).glob("*.mp4"))
        if not videos:
            print(f"warning: no videos under {videos_dir / label}", file=sys.stderr)
        for video in tqdm(videos, desc=label, unit="video"):
            ranges = event_ranges(video.stem, spec.get("anomaly", True), annotations)
            out_dir = frames_dir / label / video.stem
            try:
                records += extract_video(video, label, out_dir, root, sampling, ranges, source)
            except ValueError as exc:
                tqdm.write(f"warning: skipped {video.name}: {exc}", file=sys.stderr)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    tmp.replace(manifest_path)
    return records


def print_summary(records: list[dict]) -> None:
    for label in dict.fromkeys(r["label"] for r in records):
        rows = [r for r in records if r["label"] == label]
        events = Counter(str(r["in_event"]) for r in rows)
        sizes = Counter(f"{r['width']}x{r['height']}" for r in rows)
        print(
            f"{label}: {len(rows)} frames from {len({r['video'] for r in rows})} videos; "
            f"in_event true={events['True']} false={events['False']} null={events['None']}; "
            f"sizes {dict(sizes)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)

    cfg = load_data_config(args.config)
    data_dir: Path = args.data_dir
    videos_dir = data_dir / "videos"
    if not videos_dir.is_dir():
        print(f"No videos at {videos_dir}. Run scripts/download_data.py first.", file=sys.stderr)
        return 2

    annotations: dict[str, Ranges] = {}
    ann_cfg = cfg.get("annotations")
    if ann_cfg and (data_dir / ann_cfg["local"]).is_file():
        text = (data_dir / ann_cfg["local"]).read_text(encoding="utf-8")
        annotations = parse_temporal_annotations(text)
    else:
        print("warning: no temporal annotations; in_event will be null for anomaly videos")

    records = extract_all(
        videos_dir, data_dir / "frames", data_dir / "manifest.jsonl", ROOT, cfg, annotations
    )
    print_summary(records)
    print(f"Wrote {len(records)} frames and {data_dir.name}/manifest.jsonl")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
