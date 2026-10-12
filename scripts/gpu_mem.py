#!/usr/bin/env python3
"""Sample nvidia-smi while a benchmark runs and write results/<variant>/gpu_mem.json.

    uv run python scripts/gpu_mem.py --variant bf16 &      # start before the benchmark
    kill -TERM $!                                          # stop it afterwards
    uv run python scripts/gpu_mem.py --variant bf16 --duration 60

Start it after the server is ready and before any request is sent: the first sample is
reported as idle_mb. Without nvidia-smi it writes {"available": false} and exits 0.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results"
QUERY = "memory.used,memory.total,utilization.gpu"
NVIDIA_SMI = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse 'used, total, util' CSV lines (MiB, MiB, %). One dict per GPU line."""
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            used, total = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        try:
            util = float(parts[2])
        except ValueError:  # nvidia-smi prints "[N/A]" when it cannot read a value
            util = None
        rows.append({"used_mb": used, "total_mb": total, "util_pct": util})
    return rows


def summarize(samples: list[dict]) -> dict:
    """idle_mb is the first sample; peak_mb the largest; util is averaged over valid samples."""
    if not samples:
        return {
            "idle_mb": None,
            "peak_mb": None,
            "total_mb": None,
            "mean_util_pct": None,
            "samples": 0,
        }
    utils = [s["util_pct"] for s in samples if s.get("util_pct") is not None]
    return {
        "idle_mb": samples[0]["used_mb"],
        "peak_mb": max(s["used_mb"] for s in samples),
        "total_mb": samples[0]["total_mb"],
        "mean_util_pct": round(sum(utils) / len(utils), 1) if utils else None,
        "samples": len(samples),
    }


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_sample(gpu_index: int) -> dict | None:
    try:
        result = subprocess.run(
            [*NVIDIA_SMI, "-i", str(gpu_index)],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = parse_nvidia_smi(result.stdout) if result.returncode == 0 else []
    return rows[0] if rows else None


def gpu_name(gpu_index: int) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(gpu_index)],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def server_meta(out_dir: Path) -> dict:
    """Model and URL from serve.py's server.json, when it exists."""
    try:
        server = json.loads((out_dir / "server.json").read_text())
    except (OSError, ValueError):
        return {}
    keys = ("hf_model", "quantization", "base_url", "served_model_name", "sglang_version")
    return {key: server.get(key) for key in keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="variant id; picks results/<variant>/")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args(argv)

    out_dir = Path(args.results_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gpu_mem.json"
    meta = {
        "variant": args.variant,
        "test": "gpu_mem",
        "timestamp": now_iso(),
        "command": " ".join(NVIDIA_SMI),
        "interval_s": args.interval,
        "gpu_index": args.gpu_index,
        **server_meta(out_dir),
    }

    if shutil.which("nvidia-smi") is None:
        out_path.write_text(
            json.dumps({"available": False, "meta": meta, "summary": summarize([])}, indent=2)
            + "\n"
        )
        print(f"nvidia-smi not found; wrote {out_path} with available=false")
        return 0

    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    meta["gpu_name"] = gpu_name(args.gpu_index)
    samples = []
    start = time.monotonic()
    while not stop["requested"]:
        sample = read_sample(args.gpu_index)
        if sample is not None:
            samples.append({"t_s": round(time.monotonic() - start, 2), **sample})
        if args.duration is not None and time.monotonic() - start >= args.duration:
            break
        time.sleep(args.interval)

    meta["ended_at"] = now_iso()
    meta["duration_s"] = round(time.monotonic() - start, 1)
    raw_path = out_dir / "raw" / "gpu_mem.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    result = {"available": True, "meta": meta, "summary": summarize(samples)}
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}: {json.dumps(result['summary'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
