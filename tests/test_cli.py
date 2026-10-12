"""The command-line entry points, with a stand-in for the data owner's manifest reader."""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
from conftest import REPO, make_frames

from camalerts import bench_json


@pytest.fixture
def fake_manifest(tmp_path, monkeypatch):
    """Replace camalerts.data so these tests do not depend on it or on data/."""
    frames = make_frames(tmp_path / "frames", 6)
    calls = []

    def load_manifest(path, labels=None, limit=None, seed=0):
        calls.append({"path": path, "limit": limit, "seed": seed})
        if not Path(path).is_file():
            raise FileNotFoundError(f"Manifest not found at '{path}'")
        return frames[:limit]

    monkeypatch.setitem(
        sys.modules, "camalerts.data", types.SimpleNamespace(load_manifest=load_manifest)
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")
    return manifest, calls


def args(url, manifest, results, *extra):
    return [
        "--base-url", url, "--variant", "cli-test", "--manifest", str(manifest),
        "--results-dir", str(results), "--config", str(REPO / "configs" / "bench.yaml"),
        "--alerts", str(REPO / "configs" / "alerts.yaml"), *extra,
    ]  # fmt: skip


async def test_cli_writes_results_and_prints_a_table(mock_url, fake_manifest, tmp_path, capsys):
    manifest, calls = fake_manifest
    results = tmp_path / "out"
    # main() owns its event loop, so run it in a thread while this loop serves the mock.
    code = await asyncio.to_thread(bench_json.main, args(mock_url, manifest, results, "--n", "2"))

    assert code == 0
    assert calls == [{"path": str(manifest), "limit": 2, "seed": 0}]
    result = json.loads((results / "cli-test" / "json_validity.json").read_text())
    assert result["meta"]["n_frames"] == 2 and result["meta"]["config"]["n_frames"] == 2
    out = capsys.readouterr().out
    assert "MOCK SERVER - not a real measurement" in out
    assert "schema valid" in out and "prompt_only" in out


def test_cli_exits_2_when_the_server_is_down(fake_manifest, tmp_path, capsys):
    manifest, _ = fake_manifest
    code = bench_json.main(args("http://127.0.0.1:9", manifest, tmp_path / "out", "--n", "1"))
    assert code == 2
    assert "cannot read http://127.0.0.1:9/v1/models" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_exits_2_when_the_manifest_is_missing(fake_manifest, tmp_path, capsys):
    code = bench_json.main(args("http://127.0.0.1:9", tmp_path / "nope.jsonl", tmp_path / "out"))
    assert code == 2
    assert "Manifest not found" in capsys.readouterr().err
