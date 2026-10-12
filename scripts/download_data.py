"""Download a small, size-capped set of UCF-Crime videos.

The mirror stores multi-GB zip archives. A zip can be read at random offsets, so `remotezip` uses
HTTP range requests to read only the archive's file list and then only the chosen .mp4 members.

  uv run scripts/download_data.py --dry-run     # print the plan and its size, download nothing
  uv run scripts/download_data.py               # download (skips files already present)
  uv run scripts/download_data.py --fit         # trim the plan to the cap instead of refusing
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import zipfile
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import urllib3
from remotezip import RemoteZip, RemoteZipError
from tqdm import tqdm

from camalerts.data import DEFAULT_CONFIG, load_data_config, parse_temporal_annotations

ROOT = Path(__file__).resolve().parents[1]
GB = 10**9
HARD_MAX_GB = 5.0  # ceiling for max_total_gb under any setting
MIN_FREE_AFTER_GB = 2.0  # never fill the disk to the last byte
RETRIES = 3
# requests errors are OSError; a stream that breaks mid-read raises urllib3's own type
RETRYABLE = (OSError, RemoteZipError, urllib3.exceptions.HTTPError, zipfile.BadZipFile, zlib.error)


class CapError(Exception):
    pass


@dataclass(frozen=True)
class Member:
    """One file inside a remote zip, from the zip's central directory."""

    name: str
    size: int


@dataclass(frozen=True)
class PlanItem:
    label: str
    archive: str
    member: str
    size: int
    annotated: bool

    @property
    def video(self) -> str:
        return Path(self.member).stem

    @property
    def rel_path(self) -> str:
        return f"videos/{self.label}/{Path(self.member).name}"


# --- planning: pure functions, no network -------------------------------------------------------


def cap_bytes(max_total_gb: float) -> int:
    if not 0 < max_total_gb <= HARD_MAX_GB:
        raise CapError(
            f"max_total_gb must be above 0 and at most {HARD_MAX_GB}, got {max_total_gb}"
        )
    return int(max_total_gb * GB)


def select_videos(
    members: list[Member], pattern: str, count: int, annotated: set[str]
) -> list[Member]:
    """Videos with temporal annotations first, then by name."""
    matches = [m for m in members if fnmatch.fnmatchcase(m.name, pattern)]
    matches.sort(key=lambda m: (Path(m.name).stem not in annotated, m.name))
    return matches[:count]


def build_plan(cfg: dict, listings: dict[str, list[Member]], annotated: set[str]) -> list[PlanItem]:
    plan: list[PlanItem] = []
    for label, spec in cfg["labels"].items():
        count = spec.get("videos", cfg.get("videos_per_label", 12))
        chosen = select_videos(listings[spec["archive"]], spec["pattern"], count, annotated)
        if not chosen:
            raise ValueError(f"No member of {spec['archive']} matches '{spec['pattern']}'")
        plan += [
            PlanItem(label, spec["archive"], m.name, m.size, Path(m.name).stem in annotated)
            for m in chosen
        ]
    return plan


def plan_bytes(plan: list[PlanItem]) -> int:
    return sum(item.size for item in plan)


def fit_plan(plan: list[PlanItem], budget: int) -> list[PlanItem]:
    """Keep what fits, taking one video per label in turn so the labels stay balanced."""
    by_label: dict[str, list[PlanItem]] = {}
    for item in plan:
        by_label.setdefault(item.label, []).append(item)
    kept: set[PlanItem] = set()
    used = 0
    for rank in range(max((len(items) for items in by_label.values()), default=0)):
        for items in by_label.values():
            if rank < len(items) and used + items[rank].size <= budget:
                kept.add(items[rank])
                used += items[rank].size
    return [item for item in plan if item in kept]


def check_cap(plan: list[PlanItem], cap: int, other_bytes: int = 0) -> int:
    """Return the projected size of data/ after the download, or raise if it is over the cap."""
    total = other_bytes + plan_bytes(plan)
    if total > cap:
        raise CapError(
            f"planned videos {plan_bytes(plan) / GB:.2f} GB + other files already under data/ "
            f"{other_bytes / GB:.2f} GB = {total / GB:.2f} GB, over the cap of {cap / GB:.2f} GB. "
            "Lower videos_per_label or pass --fit to trim the plan."
        )
    return total


def is_present(data_dir: Path, item: PlanItem) -> bool:
    dest = data_dir / item.rel_path
    return dest.is_file() and dest.stat().st_size == item.size


def other_bytes_on_disk(data_dir: Path, plan: list[PlanItem]) -> int:
    """Bytes under data/ that are not a planned video (frames, videos from an older plan, ...)."""
    planned = {data_dir / item.rel_path for item in plan}
    planned |= {p.with_name(p.name + ".part") for p in planned}
    if not data_dir.is_dir():
        return 0
    return sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file() and p not in planned)


# --- network ------------------------------------------------------------------------------------


def file_url(source: dict, filename: str) -> str:
    return (
        f"https://huggingface.co/datasets/{source['hf_repo']}/resolve/"
        f"{source.get('revision', 'main')}/{filename}"
    )


def auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def open_zip(url: str) -> RemoteZip:
    return RemoteZip(url, headers=auth_headers(), timeout=60)


def list_archive(url: str) -> list[Member]:
    with open_zip(url) as zf:
        return [Member(i.filename, i.file_size) for i in zf.infolist() if not i.is_dir()]


def fetch_text(url: str) -> str:
    response = httpx.get(url, headers=auth_headers(), follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.text


def download_member(zf: RemoteZip, item: PlanItem, dest: Path, bar: tqdm) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done = 0
    try:
        # zipfile verifies the member's CRC-32 when the stream is read to the end.
        with zf.open(item.member) as src, part.open("wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                bar.update(len(chunk))
        if done != item.size:
            raise OSError(f"got {done} bytes, expected {item.size}")
        part.replace(dest)
    except BaseException:
        bar.update(-done)
        part.unlink(missing_ok=True)
        raise


def download_plan(plan: list[PlanItem], data_dir: Path, source: dict) -> dict[PlanItem, str]:
    """Download what is missing. Returns a status per item."""
    status = {item: "already_present" for item in plan if is_present(data_dir, item)}
    todo = [item for item in plan if item not in status]
    with tqdm(total=plan_bytes(todo), unit="B", unit_scale=True, desc="download") as bar:
        for archive in dict.fromkeys(item.archive for item in todo):
            url = file_url(source, archive)
            zf = None
            for item in (i for i in todo if i.archive == archive):
                bar.set_postfix_str(item.video)
                for attempt in range(1, RETRIES + 1):
                    try:
                        if zf is None:
                            zf = open_zip(url)
                        download_member(zf, item, data_dir / item.rel_path, bar)
                        status[item] = "downloaded"
                        break
                    except RETRYABLE as exc:
                        status[item] = f"failed: {type(exc).__name__}: {exc}"
                        bar.write(f"  {item.video}: attempt {attempt}/{RETRIES} failed: {exc}")
                        if zf is not None:
                            zf.close()
                            zf = None
                else:
                    continue
                bar.write(f"  ok {item.label}/{item.video} ({item.size / 1e6:.1f} MB)")
            if zf is not None:
                zf.close()
    return status


# --- output -------------------------------------------------------------------------------------


def print_plan(plan: list[PlanItem], data_dir: Path, cap: int, other: int) -> None:
    print(f"{'label':<12} {'video':<28} {'MB':>8}  annotated  on disk")
    for item in plan:
        print(
            f"{item.label:<12} {item.video:<28} {item.size / 1e6:>8.1f}  "
            f"{'yes' if item.annotated else 'no':<9}  {'yes' if is_present(data_dir, item) else 'no'}"
        )
    for label in dict.fromkeys(item.label for item in plan):
        items = [i for i in plan if i.label == label]
        print(
            f"{label}: {len(items)} videos, {plan_bytes(items) / GB:.3f} GB, "
            f"{sum(i.annotated for i in items)} with temporal annotations"
        )
    missing = [i for i in plan if not is_present(data_dir, i)]
    print(
        f"Planned videos: {len(plan)} files, {plan_bytes(plan) / GB:.3f} GB "
        f"({plan_bytes(missing) / GB:.3f} GB still to download). "
        f"Other files under {data_dir.name}/: {other / GB:.3f} GB. Cap: {cap / GB:.2f} GB."
    )


def write_log(
    path: Path, cfg: dict, plan: list[PlanItem], status: dict[PlanItem, str], data_dir: Path
) -> None:
    log = {
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": cfg["source"],
        "max_total_gb": cfg["max_total_gb"],
        "planned_bytes": plan_bytes(plan),
        "data_dir_bytes": other_bytes_on_disk(data_dir, []),
        "videos": [
            {
                "label": item.label,
                "video": item.video,
                "archive": item.archive,
                "member": item.member,
                "path": f"{data_dir.name}/{item.rel_path}",
                "bytes": item.size,
                "annotated": item.annotated,
                "status": status[item],
            }
            for item in plan
        ],
    }
    path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    parser.add_argument("--fit", action="store_true", help="trim the plan to fit under the cap")
    parser.add_argument("--max-total-gb", type=float, help=f"override the cap (max {HARD_MAX_GB})")
    parser.add_argument("--videos-per-label", type=int, help="override videos_per_label")
    args = parser.parse_args(argv)

    cfg = load_data_config(args.config)
    if args.max_total_gb is not None:
        cfg["max_total_gb"] = args.max_total_gb
    if args.videos_per_label is not None:
        cfg["videos_per_label"] = args.videos_per_label
    data_dir: Path = args.data_dir
    source = cfg["source"]

    try:
        cap = cap_bytes(cfg["max_total_gb"])
    except CapError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    ann_cfg = cfg.get("annotations")
    ann_text = fetch_text(file_url(source, ann_cfg["remote"])) if ann_cfg else ""
    annotated = set(parse_temporal_annotations(ann_text))
    print(f"Temporal annotations: {len(annotated)} videos listed")

    listings = {}
    for archive in sorted({spec["archive"] for spec in cfg["labels"].values()}):
        listings[archive] = list_archive(file_url(source, archive))
        print(f"Listed {archive}: {len(listings[archive])} files")

    plan = build_plan(cfg, listings, annotated)
    if args.fit:
        plan = fit_plan(plan, cap - other_bytes_on_disk(data_dir, plan))
    other = other_bytes_on_disk(data_dir, plan)
    print_plan(plan, data_dir, cap, other)

    try:
        check_cap(plan, cap, other)
    except CapError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("Dry run: nothing downloaded.")
        return 0

    data_dir.mkdir(parents=True, exist_ok=True)
    to_fetch = plan_bytes([i for i in plan if not is_present(data_dir, i)])
    free = shutil.disk_usage(data_dir).free
    if to_fetch + MIN_FREE_AFTER_GB * GB > free:
        print(
            f"REFUSING: need {to_fetch / GB:.2f} GB and must leave {MIN_FREE_AFTER_GB} GB free, "
            f"but only {free / GB:.2f} GB is free.",
            file=sys.stderr,
        )
        return 2

    if ann_cfg:
        ann_path = data_dir / ann_cfg["local"]
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(ann_text, encoding="utf-8")

    status = download_plan(plan, data_dir, source)
    write_log(data_dir / "download_log.json", cfg, plan, status, data_dir)

    failed = [item for item in plan if status[item].startswith("failed")]
    counts = {s: sum(v == s for v in status.values()) for s in ("downloaded", "already_present")}
    print(
        f"Done: {counts['downloaded']} downloaded, {counts['already_present']} already present, "
        f"{len(failed)} failed. Log: {data_dir.name}/download_log.json"
    )
    for item in failed:
        print(f"  FAILED {item.label}/{item.video}: {status[item]}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
