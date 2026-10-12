import json

import pytest

from camalerts import bench_json, bench_prefix, common, report

MOCK = "mock-test (MOCK - not a real measurement)"


def write(folder, name, data):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data))


@pytest.fixture
async def results(tmp_path, mock_url, frames, alerts, config):
    """A results dir with a mock variant (two tests run), a failed variant and a real-looking one."""
    root = tmp_path / "results"
    for test, module in (("json_validity", bench_json), ("prefix_cache", bench_prefix)):
        bench = await common.connect(
            test,
            base_url=mock_url,
            variant="mock-test",
            results_dir=root,
            alerts=alerts,
            config=config,
        )
        await module.run(bench, frames)

    # The shape scripts/serve.py writes: a generic reason, and the cause at the end of the log.
    write(root / "awq-w4a16", "server_error.json", {
        "variant": "awq-w4a16",
        "error": "server exited with code 1 before it was ready",
        "log_tail": ["Traceback (most recent call last):", "  ...",
                     "torch.OutOfMemoryError: CUDA out of memory | tried 2.1 GiB", ""],
    })  # fmt: skip

    real = json.loads((root / "mock-test" / "json_validity.json").read_text())
    real["meta"].update(variant="bf16", mock=False, model="org/some-vlm", labels={"in_event": 9})
    write(root / "bf16", "json_validity.json", real)
    write(root / "bf16", "server.json", {
        "hf_model": "org/some-vlm", "quantization": "none", "load_seconds": 41.26,
        "weights_mem_gb": 4.26, "max_total_num_tokens": 65536,
    })  # fmt: skip
    write(root / "bf16", "gpu_mem.json", {
        "summary": {"idle_mb": 5120, "peak_mb": 6144.4, "total_mb": 24564},
    })  # fmt: skip
    write(root / "mock-test", "gpu_mem.json", {"available": False, "summary": {}})
    return root


def rows(text: str, section: str) -> list[str]:
    body = text.split(f"## {section}", 1)[1].split("\n## ", 1)[0]
    return [line for line in body.splitlines() if line.startswith("| ") and "---" not in line][1:]


async def test_report_renders_every_section(results):
    assert report.main(["--results-dir", str(results)]) == 0
    text = (results / "REPORT.md").read_text()
    for heading in (
        "## Variants",
        "## JSON validity",
        "## Prefix cache",
        "## Concurrency",
        "## Image size",
    ):
        assert heading in text
    assert len(rows(text, "JSON validity")) == 3 + 1 + 3  # mock modes, failed variant, bf16 modes
    assert "100.0%" in text and "concealment: acc" in text
    assert "recall " in text and "precision " in text and "scored 9/9" in text


async def test_mock_rows_are_labelled_in_every_table(results):
    text = report.render(results)
    intro = text.split("\n## ", 1)[0]
    assert "MOCK - not a real measurement" in intro and "mock-test" in intro
    for section in ("Variants", "JSON validity", "Prefix cache", "Concurrency", "Image size"):
        mock_rows = [row for row in rows(text, section) if "mock-test" in row]
        assert mock_rows and all(row.startswith(f"| {MOCK} |") for row in mock_rows)
    assert not any("MOCK" in row for row in text.splitlines() if row.startswith("| bf16"))


async def test_weak_labels_are_stated_at_the_top(results):
    intro = report.render(results).split("\n## ", 1)[0]
    assert "**Weak labels:** for mock-test," in intro  # bf16 used in_event, so it is not listed
    assert "class of the whole video" in intro


async def test_failed_variant_is_listed_with_its_reason(results):
    text = report.render(results)
    (variant_row,) = [row for row in rows(text, "Variants") if row.startswith("| awq-w4a16")]
    assert "FAILED to start: server exited with code 1 before it was ready" in variant_row
    # The cause is the last log line; its pipe is escaped so the table keeps its columns.
    assert (
        "last log line: torch.OutOfMemoryError: CUDA out of memory \\| tried 2.1 GiB" in variant_row
    )
    for section in ("JSON validity", "Prefix cache", "Concurrency", "Image size"):
        assert any(row.startswith("| awq-w4a16 | failed to start |") for row in rows(text, section))


async def test_missing_tests_show_as_not_run(results):
    text = report.render(results)
    assert any(row.startswith(f"| {MOCK} | not run |") for row in rows(text, "Concurrency"))
    assert any(row.startswith("| bf16 | not run |") for row in rows(text, "Prefix cache"))


async def test_server_and_gpu_files_are_shown(results):
    variants = {
        row.split(" | ")[0].lstrip("| "): row for row in rows(report.render(results), "Variants")
    }
    bf16 = [cell.strip() for cell in variants["bf16"].strip("|").split("|")]
    assert bf16 == [
        "bf16", "org/some-vlm", "org/some-vlm", "none", "41.3", "4.26", "65536",
        "5120", "6144", "24564", "1/4 tests",
    ]  # fmt: skip
    assert variants[MOCK].count("not available") == 3
    assert variants["awq-w4a16"].count("not run") == 3


def test_empty_results_dir(tmp_path):
    assert "No results found." in report.render(tmp_path)
    assert "No results found." in report.render(tmp_path / "missing")


def test_unreadable_result_file_is_reported(tmp_path):
    (tmp_path / "bf16").mkdir()
    (tmp_path / "bf16" / "server.json").write_text("{not json")
    (tmp_path / "bf16" / "server_error.json").write_text(json.dumps({"reason": "port in use"}))
    text = report.render(tmp_path)
    assert "**Unreadable file:** `bf16/server.json`" in text
    assert "FAILED to start: port in use" in text
