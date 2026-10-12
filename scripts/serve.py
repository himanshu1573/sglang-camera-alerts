#!/usr/bin/env python3
"""Launch SGLang for one model variant from configs/models.yaml.

    uv run python scripts/serve.py --variant bf16 --dry-run
    uv run python scripts/serve.py --variant bf16            # foreground, Ctrl-C to stop
    uv run python scripts/serve.py --variant bf16 --detach   # returns once the server is ready
    uv run python scripts/serve.py --variant bf16 --stop
    uv run python scripts/serve.py --variant bf16 -- --mem-fraction-static 0.7   # extra SGLang args

SGLang usually lives in its own environment. Point SGLANG_PYTHON at that interpreter
(scripts/runpod_setup.sh does this); otherwise "python3" is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "models.yaml"
DEFAULT_RESULTS = REPO_ROOT / "results"
LAUNCH_MODULE = "sglang.launch_server"
LOG_TAIL_LINES = 60
STOP_GRACE_S = 30

# Finds "<python> -c ..." whether or not sglang is installed, without importing it.
VERSION_PROBE = (
    "import importlib.util, sys\n"
    "if importlib.util.find_spec('sglang') is None: sys.exit(3)\n"
    "try:\n"
    "    from importlib.metadata import version\n"
    "    print(version('sglang'))\n"
    "except Exception:\n"
    "    print('')\n"
)


class ConfigError(ValueError):
    """models.yaml is missing something, or the variant does not exist."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("variants"), list):
        raise ConfigError(f"{path} must contain a top-level 'variants' list")
    return config


def get_variant(config: dict, variant_id: str) -> dict:
    for variant in config["variants"]:
        if variant.get("id") == variant_id:
            return variant
    known = ", ".join(str(v.get("id")) for v in config["variants"])
    raise ConfigError(f"unknown variant '{variant_id}'. Known variants: {known}")


def resolve_python() -> str:
    return os.environ.get("SGLANG_PYTHON") or "python3"


def build_command(
    config: dict,
    variant_id: str,
    *,
    port: int | None = None,
    host: str | None = None,
    python: str | None = None,
    extra: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Return the argv that starts SGLang for this variant. Pure: touches no files or GPU."""
    variant = get_variant(config, variant_id)
    defaults = config.get("defaults") or {}
    if not variant.get("hf_model"):
        raise ConfigError(f"variant '{variant_id}' has no hf_model")
    return [
        python or resolve_python(),
        "-m",
        LAUNCH_MODULE,
        "--model-path",
        str(variant["hf_model"]),
        "--host",
        str(host or defaults.get("host", "127.0.0.1")),
        "--port",
        str(port or defaults.get("port", 30000)),
        *[str(arg) for arg in defaults.get("sglang_args") or []],
        *[str(arg) for arg in variant.get("sglang_args") or []],
        *[str(arg) for arg in extra],
    ]


def flag_value(command: list[str], flag: str) -> str | None:
    """Last value given for a flag, in '--flag value' or '--flag=value' form."""
    value = None
    for index, arg in enumerate(command):
        if arg == flag and index + 1 < len(command):
            value = command[index + 1]
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def expected_model_name(command: list[str]) -> str:
    return flag_value(command, "--served-model-name") or flag_value(command, "--model-path") or ""


def base_url(command: list[str]) -> str:
    host = flag_value(command, "--host") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{flag_value(command, '--port')}"


def tail_lines(path: Path, count: int = LOG_TAIL_LINES) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(errors="replace").splitlines()[-count:]


def parse_weight_mem_gb(log_text: str) -> float | None:
    """GPU memory taken by the weights, from SGLang's 'Load weight end. ... mem usage=X GB.' line."""
    matches = re.findall(r"Load weight end\..*?mem usage=([0-9.]+) GB", log_text)
    return float(matches[-1]) if matches else None


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def probe_sglang_version(python: str) -> tuple[bool, str | None]:
    """(installed, version). Does not import sglang, so no GPU code runs."""
    try:
        result = subprocess.run(
            [python, "-c", VERSION_PROBE], capture_output=True, check=False, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    if result.returncode != 0:
        return False, None
    return True, result.stdout.strip() or None


def model_is_cached(hf_model: str) -> bool:
    """True if the Hugging Face cache already holds a snapshot of this repo."""
    if Path(hf_model).exists():
        return True
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub:
        home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        hub = str(Path(home) / "hub")
    snapshots = Path(hub) / ("models--" + hf_model.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def gpu_info() -> list[str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip().splitlines() if result.returncode == 0 else None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    # In a container whose PID 1 does not reap orphans, a stopped server stays a zombie:
    # it still answers signal 0 but holds no port and no GPU memory.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[-1].split()[:1] != ["Z"]


def pid_command(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def stop_process_group(pid: int, grace_s: float = STOP_GRACE_S) -> None:
    """SIGTERM the server's process group, then SIGKILL whatever is left."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            os.kill(pid, sig)
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return
            try:  # reap the child if it is ours, otherwise it stays a zombie and looks alive
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    return
            except ChildProcessError:
                pass
            time.sleep(0.2)


def wait_until_ready(proc: subprocess.Popen, url: str, timeout_s: float, stop: dict) -> str:
    """Poll /health. Returns 'ready', 'exited', 'timeout' or 'interrupted'."""
    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            if stop["requested"]:
                return "interrupted"
            if proc.poll() is not None:
                return "exited"
            try:
                if client.get(f"{url}/health").status_code == 200:
                    return "ready"
            except httpx.HTTPError:
                pass
            time.sleep(2)
    return "timeout"


def collect_server_facts(url: str) -> dict:
    """Ask the running server who it is. Every call is best effort."""
    facts: dict = {}
    with httpx.Client(timeout=60, trust_env=False) as client:
        try:
            models = client.get(f"{url}/v1/models").json()
            facts["served_model_name"] = models["data"][0]["id"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            pass
        try:
            info = client.get(f"{url}/server_info").json()
            facts["sglang_version"] = info.get("version")
            facts["max_total_num_tokens"] = info.get("max_total_num_tokens")
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        try:
            facts["health_generate_ok"] = client.get(f"{url}/health_generate").status_code == 200
        except httpx.HTTPError:
            facts["health_generate_ok"] = False
    return facts


def fail(out_dir: Path, base: dict, reason: str, log_path: Path, **more) -> int:
    error = {
        **base,
        "error": reason,
        **more,
        "failed_at": now_iso(),
        "log_tail": tail_lines(log_path),
    }
    write_json(out_dir / "server_error.json", error)
    print(f"ERROR: {reason}", file=sys.stderr)
    print(f"Details: {out_dir / 'server_error.json'}", file=sys.stderr)
    return 1


def stop_detached(out_dir: Path, variant_id: str) -> int:
    pid_file = out_dir / "server.pid"
    if not pid_file.is_file():
        print(f"No detached server recorded for '{variant_id}' ({pid_file} is missing).")
        return 0
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        print(f"ERROR: {pid_file} does not hold a pid.", file=sys.stderr)
        return 1
    command = pid_command(pid)
    if not pid_alive(pid):
        print(f"Server for '{variant_id}' (pid {pid}) is not running.")
    # SGLang may rename its processes, so match the name loosely. An empty command means `ps`
    # is unavailable; the pid came from our own pid file, so stop it anyway.
    elif command and "sglang" not in command.lower():
        print(
            f"ERROR: pid {pid} is not an SGLang server ({command[:80]!r}); not touching it.",
            file=sys.stderr,
        )
        return 1
    else:
        print(f"Stopping server for '{variant_id}' (pid {pid}) ...")
        stop_process_group(pid)
        if pid_alive(pid):
            print(f"ERROR: pid {pid} is still alive.", file=sys.stderr)
            return 1
        print("Stopped.")
    pid_file.unlink()
    return 0


def run(args: argparse.Namespace, config: dict, command: list[str]) -> int:
    variant = get_variant(config, args.variant)
    out_dir = Path(args.results_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    pid_file = out_dir / "server.pid"
    url = base_url(command)
    base = {
        "variant": args.variant,
        "hf_model": variant["hf_model"],
        "quantization": variant.get("quantization"),
        "status": variant.get("status"),
        "command": command,
        "command_str": shlex.join(command),
        "base_url": url,
        "log_path": str(log_path),
    }
    for stale in ("server.json", "server_error.json"):
        (out_dir / stale).unlink(missing_ok=True)

    if pid_file.is_file():
        old_pid = pid_file.read_text().strip()
        if old_pid.isdigit() and pid_alive(int(old_pid)):
            print(
                f"ERROR: a server for '{args.variant}' is already running (pid {old_pid}). "
                "Stop it with --stop first.",
                file=sys.stderr,
            )
            return 1
        pid_file.unlink()

    installed, version = probe_sglang_version(command[0])
    if not installed:
        log_path.write_text("")
        return fail(
            out_dir,
            base,
            f"'{command[0]}' cannot find the sglang package. Run scripts/runpod_setup.sh "
            "or set SGLANG_PYTHON to the interpreter that has SGLang installed.",
            log_path,
        )
    host = url.split("//", 1)[1].rsplit(":", 1)[0]
    port = int(flag_value(command, "--port"))
    if port_in_use(host, port):
        log_path.write_text("")
        return fail(out_dir, base, f"port {port} is already in use on {host}", log_path)

    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    wanted = config.get("sglang_version")
    if wanted and version and str(wanted) != version:
        print(f"WARNING: models.yaml was checked against SGLang {wanted}, found {version}.")
    if not os.environ.get("HF_HOME") and Path("/workspace").is_dir():
        print("WARNING: HF_HOME is not set; models will not be cached on /workspace.")
    cached = model_is_cached(variant["hf_model"])
    print(f"Starting {shlex.join(command)}")
    print(f"Log: {log_path}")
    started_at = now_iso()
    start = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    base["pid"] = proc.pid

    outcome = wait_until_ready(proc, url, args.timeout_s, stop)
    load_seconds = round(time.monotonic() - start, 1)
    if outcome != "ready":
        exit_code = proc.poll()
        stop_process_group(proc.pid)
        if outcome == "interrupted":
            print("Interrupted before the server was ready. Stopped.")
            return 130
        reason = (
            f"server exited with code {exit_code} before it was ready"
            if outcome == "exited"
            else f"server was not ready after {args.timeout_s:.0f} s"
        )
        return fail(
            out_dir, base, reason, log_path, exit_code=exit_code, waited_seconds=load_seconds
        )

    facts = collect_server_facts(url)
    name = facts.get("served_model_name") or expected_model_name(command)
    info = {
        **base,
        "served_model_name": name,
        "sglang_version": facts.get("sglang_version") or version,
        "load_seconds": load_seconds,
        "model_cached_before_start": cached,
        "weights_mem_gb": parse_weight_mem_gb(log_path.read_text(errors="replace")),
        "max_total_num_tokens": facts.get("max_total_num_tokens"),
        "health_generate_ok": facts.get("health_generate_ok"),
        "gpu": gpu_info(),
        "hf_home": os.environ.get("HF_HOME"),
        "started_at": started_at,
        "ready_at": now_iso(),
    }
    write_json(out_dir / "server.json", info)
    print(f"Ready after {load_seconds} s.")
    print(f"served_model_name={name}")
    print(f"base_url={url}")
    print(f"pid={proc.pid}")

    if args.detach:
        pid_file.write_text(f"{proc.pid}\n")
        print(f"Detached. Stop it with: scripts/serve.py --variant {args.variant} --stop")
        return 0

    print("Running in the foreground. Press Ctrl-C to stop.")
    while proc.poll() is None and not stop["requested"]:
        time.sleep(0.5)
    if stop["requested"]:
        print("Stopping the server ...")
        stop_process_group(proc.pid)
        print("Stopped.")
        return 0
    return fail(
        out_dir,
        base,
        f"server exited on its own with code {proc.returncode}",
        log_path,
        exit_code=proc.returncode,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="variant id from the config file")
    parser.add_argument("--port", type=int, help="override defaults.port")
    parser.add_argument("--host", help="override defaults.host")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    parser.add_argument("--timeout-s", type=float, default=1800, help="seconds to wait for /health")
    parser.add_argument("--detach", action="store_true", help="return once the server is ready")
    parser.add_argument("--stop", action="store_true", help="stop a detached server")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="ARGS",
        help="more SGLang arguments, e.g. --extra='--mem-fraction-static 0.7' (repeatable)",
    )
    parser.add_argument(
        "passthrough",
        nargs="*",
        metavar="-- ARGS",
        help="everything after a bare -- goes to SGLang unchanged (use this for JSON values)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        extra = [token for chunk in args.extra for token in shlex.split(chunk)]
        extra += args.passthrough
        command = build_command(config, args.variant, port=args.port, host=args.host, extra=extra)
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(shlex.join(command))
        return 0
    if args.stop:
        return stop_detached(Path(args.results_dir) / args.variant, args.variant)
    return run(args, config, command)


if __name__ == "__main__":
    sys.exit(main())
