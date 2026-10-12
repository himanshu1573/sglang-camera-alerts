"""Tests for the serving scripts. No GPU and no internet; two tests start a fake local server."""

import importlib.util
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
SERVE = REPO_ROOT / "scripts" / "serve.py"
GPU_MEM = REPO_ROOT / "scripts" / "gpu_mem.py"

VARIANT_KEYS = {"id", "hf_model", "quantization", "status", "sglang_args", "notes"}

# Flags found in SGLang's server-arguments page on 2026-10-12. A flag that is not listed here
# must be checked against the docs before it goes into models.yaml.
DOCUMENTED_FLAGS = {
    "--attention-backend",
    "--chat-template",
    "--chunked-prefill-size",
    "--context-length",
    "--disable-radix-cache",
    "--dtype",
    "--enable-cache-report",
    "--enable-metrics",
    "--enable-multimodal",
    "--fp8-gemm-backend",
    "--grammar-backend",
    "--kv-cache-dtype",
    "--log-level",
    "--max-running-requests",
    "--max-total-tokens",
    "--mem-fraction-static",
    "--mm-attention-backend",
    "--mm-process-config",
    "--quantization",
    "--sampling-backend",
    "--sampling-defaults",
    "--schedule-policy",
    "--served-model-name",
    "--stream-response-default-include-usage",
    "--trust-remote-code",
}


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = load_script(SERVE)
gpu_mem = load_script(GPU_MEM)
CONFIG = serve.load_config(CONFIG_PATH)
VARIANT_IDS = [variant["id"] for variant in CONFIG["variants"]]


def run_script(script: Path, *args: str, env: dict | None = None, timeout: int = 60):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        check=False,
        text=True,
        env=env,
        timeout=timeout,
    )


# ---------------------------------------------------------------- models.yaml


def test_config_matches_design_shape():
    assert CONFIG["base_model"].count("/") == 1
    assert CONFIG["defaults"]["port"] == 30000
    assert isinstance(CONFIG["defaults"]["host"], str)
    assert len(VARIANT_IDS) == len(set(VARIANT_IDS))
    for variant in CONFIG["variants"]:
        assert set(variant) == VARIANT_KEYS, variant.get("id")
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", variant["id"])
        assert variant["hf_model"].count("/") == 1
        assert isinstance(variant["quantization"], str) and variant["quantization"]
        assert variant["status"] in {"expected", "experimental"}
        assert isinstance(variant["sglang_args"], list)
        assert all(isinstance(arg, str) for arg in variant["sglang_args"])
        assert "http" in variant["notes"], f"{variant['id']} notes need a source URL"


def test_required_variants_exist():
    assert {"bf16", "awq-w4a16"} <= set(VARIANT_IDS)
    bf16 = serve.get_variant(CONFIG, "bf16")
    assert bf16["hf_model"] == CONFIG["base_model"]
    assert bf16["quantization"] == "none"
    assert bf16["sglang_args"][:2] == ["--dtype", "bfloat16"]


def test_only_documented_flags_are_used():
    args = list(CONFIG["defaults"]["sglang_args"])
    for variant in CONFIG["variants"]:
        args += variant["sglang_args"]
    flags = {arg for arg in args if arg.startswith("--")}
    assert flags <= DOCUMENTED_FLAGS, flags - DOCUMENTED_FLAGS


def test_pre_quantized_checkpoints_do_not_pass_quantization_flag():
    # SGLang docs: for an offline-quantized model, pass only --model-path.
    for variant in CONFIG["variants"]:
        if variant["hf_model"] != CONFIG["base_model"]:
            assert "--quantization" not in variant["sglang_args"], variant["id"]


# ------------------------------------------------------------ command building


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_command_has_model_path_and_port(variant_id):
    variant = serve.get_variant(CONFIG, variant_id)
    command = serve.build_command(CONFIG, variant_id, python="python3")
    assert command[:3] == ["python3", "-m", "sglang.launch_server"]
    assert serve.flag_value(command, "--model-path") == variant["hf_model"]
    assert serve.flag_value(command, "--port") == "30000"
    assert serve.flag_value(command, "--host") == CONFIG["defaults"]["host"]
    for arg in variant["sglang_args"]:
        assert arg in command
    assert all(isinstance(arg, str) for arg in command)


def test_overrides_and_extra_args():
    command = serve.build_command(
        CONFIG,
        "bf16",
        port=31000,
        host="0.0.0.0",
        python="/opt/sglang/bin/python",
        extra=["--mem-fraction-static", "0.7"],
    )
    assert command[0] == "/opt/sglang/bin/python"
    assert serve.flag_value(command, "--port") == "31000"
    assert command[-2:] == ["--mem-fraction-static", "0.7"]
    assert serve.base_url(command) == "http://127.0.0.1:31000"
    assert serve.expected_model_name(command) == CONFIG["base_model"]


def test_noradix_variant_only_adds_the_cache_flag():
    if "bf16-noradix" not in VARIANT_IDS:
        pytest.skip("no bf16-noradix variant")
    base = serve.build_command(CONFIG, "bf16", python="python3")
    noradix = serve.build_command(CONFIG, "bf16-noradix", python="python3")
    assert [arg for arg in noradix if arg not in base] == ["--disable-radix-cache"]


def test_unknown_variant_gives_clear_error():
    with pytest.raises(serve.ConfigError) as error:
        serve.build_command(CONFIG, "no-such-variant")
    message = str(error.value)
    assert "no-such-variant" in message
    assert "bf16" in message


def test_unknown_variant_cli_exits_non_zero():
    result = run_script(SERVE, "--variant", "no-such-variant", "--dry-run")
    assert result.returncode == 2
    assert "unknown variant 'no-such-variant'" in result.stderr
    assert result.stdout == ""


def test_parse_weight_mem_from_log():
    log = (
        "[2026-10-13 10:00:00] Load weight begin. avail mem=23.10 GB\n"
        "[2026-10-13 10:00:09] Load weight end. elapsed=9.01 s, type=Qwen3VLForConditional"
        "Generation, avail mem=18.72 GB, mem usage=4.38 GB.\n"
    )
    assert serve.parse_weight_mem_gb(log) == 4.38
    assert serve.parse_weight_mem_gb("nothing here") is None


# ------------------------------------------------------------------- dry run


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_dry_run_prints_the_command(variant_id, tmp_path):
    env = {**os.environ, "SGLANG_PYTHON": "/venvs/sglang/bin/python"}
    result = run_script(
        SERVE, "--variant", variant_id, "--dry-run", "--results-dir", str(tmp_path), env=env
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    assert shlex.split(lines[0]) == serve.build_command(
        CONFIG, variant_id, python="/venvs/sglang/bin/python"
    )
    assert list(tmp_path.iterdir()) == [], "a dry run must not write anything"


def test_dry_run_extra_and_port(tmp_path):
    result = run_script(
        SERVE,
        "--variant",
        "bf16",
        "--port",
        "30123",
        "--extra=--log-level warning",
        "--dry-run",
        "--results-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert serve.flag_value(command, "--port") == "30123"
    assert command[-2:] == ["--log-level", "warning"]


def test_dry_run_passes_json_through_after_double_dash(tmp_path):
    mm_config = '{"image":{"max_pixels":1048576}}'
    result = run_script(
        SERVE,
        "--variant",
        "bf16",
        "--dry-run",
        "--results-dir",
        str(tmp_path),
        "--",
        "--mm-process-config",
        mm_config,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[-2:] == ["--mm-process-config", mm_config]


def test_stop_without_a_server_is_a_no_op(tmp_path):
    result = run_script(SERVE, "--variant", "bf16", "--stop", "--results-dir", str(tmp_path))
    assert result.returncode == 0
    assert "No detached server" in result.stdout


# ------------------------------------------- lifecycle against a fake SGLang

FAKE_LAUNCH_SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
print("Load weight end. elapsed=0.10 s, avail mem=20.00 GB, mem usage=1.25 GB.", flush=True)
if os.environ.get("FAKE_SGLANG_FAIL"):
    print("RuntimeError: fake out of memory", flush=True)
    sys.exit(7)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = {
            "/health": {},
            "/health_generate": {},
            "/v1/models": {"data": [{"id": value("--model-path")}]},
            "/server_info": {"version": "0.0.0-fake", "max_total_num_tokens": 4096},
        }.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(json.dumps(body or {}).encode())

    def log_message(self, *args):
        pass

HTTPServer((value("--host"), int(value("--port"))), Handler).serve_forever()
"""


@pytest.fixture
def fake_sglang_env(tmp_path):
    """An environment in which 'python -m sglang.launch_server' is a tiny local HTTP server."""
    package = tmp_path / "fake_site" / "sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "launch_server.py").write_text(textwrap.dedent(FAKE_LAUNCH_SERVER))
    env = {
        **os.environ,
        "SGLANG_PYTHON": sys.executable,
        "PYTHONPATH": str(package.parent),
        "HF_HOME": str(tmp_path / "hf"),
    }
    env.pop("FAKE_SGLANG_FAIL", None)
    return env


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_detach_writes_server_json_and_stop_ends_it(tmp_path, fake_sglang_env):
    results = tmp_path / "results"
    port = free_port()
    common = ["--variant", "awq-w4a16", "--results-dir", str(results)]
    started = run_script(
        SERVE, *common, "--port", str(port), "--detach", "--timeout-s", "40", env=fake_sglang_env
    )
    try:
        assert started.returncode == 0, started.stdout + started.stderr
        variant = serve.get_variant(CONFIG, "awq-w4a16")
        assert f"served_model_name={variant['hf_model']}" in started.stdout

        info = json.loads((results / "awq-w4a16" / "server.json").read_text())
        assert info["variant"] == "awq-w4a16"
        assert info["hf_model"] == variant["hf_model"]
        assert info["quantization"] == variant["quantization"]
        assert info["served_model_name"] == variant["hf_model"]
        assert info["base_url"] == f"http://127.0.0.1:{port}"
        assert info["sglang_version"] == "0.0.0-fake"
        assert info["weights_mem_gb"] == 1.25
        assert info["max_total_num_tokens"] == 4096
        assert info["health_generate_ok"] is True
        assert info["load_seconds"] >= 0
        assert info["command"][1:3] == ["-m", "sglang.launch_server"]
        assert shlex.split(info["command_str"]) == info["command"]
        pid = int((results / "awq-w4a16" / "server.pid").read_text())
        assert pid == info["pid"]
        assert serve.pid_alive(pid)
        assert not (results / "awq-w4a16" / "server_error.json").exists()

        again = run_script(SERVE, *common, "--port", str(port), "--detach", env=fake_sglang_env)
        assert again.returncode == 1
        assert "already running" in again.stderr
    finally:
        stopped = run_script(SERVE, *common, "--stop", env=fake_sglang_env)
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    assert not (results / "awq-w4a16" / "server.pid").exists()
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0, "the port must be free after --stop"


def test_failed_start_writes_server_error_json(tmp_path, fake_sglang_env):
    results = tmp_path / "results"
    env = {**fake_sglang_env, "FAKE_SGLANG_FAIL": "1"}
    result = run_script(
        SERVE,
        "--variant",
        "bf16",
        "--port",
        str(free_port()),
        "--results-dir",
        str(results),
        "--timeout-s",
        "40",
        env=env,
    )
    assert result.returncode != 0
    error = json.loads((results / "bf16" / "server_error.json").read_text())
    assert error["variant"] == "bf16"
    assert error["exit_code"] == 7
    assert "before it was ready" in error["error"]
    assert any("fake out of memory" in line for line in error["log_tail"])
    assert len(error["log_tail"]) <= 60
    assert not (results / "bf16" / "server.json").exists()


def test_missing_sglang_is_reported(tmp_path):
    env = {**os.environ, "SGLANG_PYTHON": sys.executable}
    env.pop("PYTHONPATH", None)
    result = run_script(
        SERVE, "--variant", "bf16", "--results-dir", str(tmp_path), "--timeout-s", "5", env=env
    )
    assert result.returncode != 0
    error = json.loads((tmp_path / "bf16" / "server_error.json").read_text())
    assert "cannot find the sglang package" in error["error"]


# ------------------------------------------------------------------ gpu_mem


def test_parse_nvidia_smi_single_gpu():
    assert gpu_mem.parse_nvidia_smi("5120, 24564, 37\n") == [
        {"used_mb": 5120.0, "total_mb": 24564.0, "util_pct": 37.0}
    ]


def test_parse_nvidia_smi_handles_many_gpus_and_bad_lines():
    text = "1024, 24564, 0\n\n2048, 24564, [N/A]\nFailed to initialize NVML\n"
    rows = gpu_mem.parse_nvidia_smi(text)
    assert [row["used_mb"] for row in rows] == [1024.0, 2048.0]
    assert rows[1]["util_pct"] is None


def test_summarize_samples():
    samples = (
        gpu_mem.parse_nvidia_smi("4000, 24564, 0")
        + gpu_mem.parse_nvidia_smi("9000, 24564, 80")
        + gpu_mem.parse_nvidia_smi("7000, 24564, 40")
    )
    assert gpu_mem.summarize(samples) == {
        "idle_mb": 4000.0,
        "peak_mb": 9000.0,
        "total_mb": 24564.0,
        "mean_util_pct": 40.0,
        "samples": 3,
    }
    assert gpu_mem.summarize([])["samples"] == 0


def test_gpu_mem_without_nvidia_smi_writes_unavailable(tmp_path):
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    env = {**os.environ, "PATH": str(empty_bin)}
    result = run_script(
        GPU_MEM, "--variant", "bf16", "--duration", "1", "--results-dir", str(tmp_path), env=env
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "bf16" / "gpu_mem.json").read_text())
    assert data["available"] is False
    assert data["meta"]["variant"] == "bf16"
    assert data["summary"]["samples"] == 0


def test_gpu_mem_samples_a_fake_nvidia_smi(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "nvidia-smi"
    fake.write_text('#!/bin/sh\necho "6000, 24564, 55"\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    result = run_script(
        GPU_MEM,
        "--variant",
        "fp8",
        "--duration",
        "0.5",
        "--interval",
        "0.2",
        "--results-dir",
        str(tmp_path),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "fp8" / "gpu_mem.json").read_text())
    assert data["available"] is True
    assert data["summary"]["idle_mb"] == 6000.0
    assert data["summary"]["peak_mb"] == 6000.0
    assert data["summary"]["total_mb"] == 24564.0
    assert data["summary"]["mean_util_pct"] == 55.0
    assert data["summary"]["samples"] >= 2
    assert (tmp_path / "fp8" / "raw" / "gpu_mem.jsonl").read_text().count("\n") >= 2
