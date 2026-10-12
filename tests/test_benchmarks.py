"""Every benchmark, end to end, against the in-process mock server."""

import json

import pytest
from aiohttp import web
from conftest import make_frames

from camalerts import bench_concurrency, bench_image_size, bench_json, bench_prefix, common

META_KEYS = {
    "test", "variant", "model", "base_url", "timestamp_utc", "mock", "n_frames", "n_requests",
    "errors", "labels", "usage_source", "cache_flush", "config",
}  # fmt: skip


@pytest.fixture
def connect(tmp_path, alerts, config):
    async def make(test: str, url: str) -> common.Bench:
        return await common.connect(
            test,
            base_url=url,
            variant="mock-test",
            results_dir=tmp_path / "results",
            alerts=alerts,
            config=config,
        )

    return make


def check_files(bench: common.Bench, result: dict, n_records: int) -> None:
    out = bench.results_dir / bench.variant
    assert json.loads((out / f"{bench.test}.json").read_text()) == result
    raw = [
        json.loads(line) for line in (out / "raw" / f"{bench.test}.jsonl").read_text().splitlines()
    ]
    assert len(raw) == n_records
    assert {"frame", "task", "ok", "ttft_s", "total_s", "text", "schema_valid"} <= raw[0].keys()
    assert "messages" not in raw[0]  # no base64 images in the raw log

    meta = result["meta"]
    assert META_KEYS <= meta.keys()
    assert meta["test"] == bench.test and meta["variant"] == "mock-test"
    assert meta["model"] == "mock-vlm" and meta["mock"] is True
    assert meta["base_url"] == bench.base_url and meta["timestamp_utc"].endswith("+00:00")
    assert meta["cache_flush"] == "ok" and meta["errors"] == 0
    assert meta["config"]["max_tokens"] == 96 and meta["config"]["temperature"] == 0


async def test_json_validity(mock_url, frames, connect):
    bench = await connect("json_validity", mock_url)
    result = await bench_json.run(bench, frames)

    n = len(frames) * 3  # every frame x three tasks, per mode
    check_files(bench, result, n_records=3 * n + 3)  # three modes, one warm-up each
    assert result["meta"]["n_frames"] == 9 and result["meta"]["n_requests"] == 3 * n
    assert result["meta"]["labels"] == {"weak": 9}
    assert result["meta"]["usage_source"] == {"server": 3 * n}

    modes = result["summary"]["modes"]
    assert list(modes) == ["constrained", "schema_prompt_only", "prompt_only"]
    # The control mode sends the constrained mode's prompt without the decoder constraint.
    control = modes["schema_prompt_only"]
    assert (control["response_format"], control["instructions"]) == (False, "schema")
    assert modes["constrained"]["instructions"] == "schema"
    for mode in modes.values():
        assert {
            "n", "errors", "strict_json_rate", "schema_valid_rate", "loose_json_rate",
            "mean_completion_tokens", "mean_total_s", "accuracy",
        } <= mode.keys()  # fmt: skip
        assert mode["n"] == n and mode["mean_total_s"] > 0 and mode["mean_completion_tokens"] > 0
        assert set(mode["accuracy"]) == {"concealment", "aggression"}  # presence is not scored
        assert mode["accuracy"]["concealment"]["n"] == 9

    constrained, prompt_only = modes["constrained"], modes["prompt_only"]
    assert constrained["strict_json_rate"] == constrained["schema_valid_rate"] == 1.0
    assert prompt_only["schema_valid_rate"] < constrained["schema_valid_rate"]
    assert prompt_only["schema_valid_rate"] <= prompt_only["strict_json_rate"]
    assert prompt_only["strict_json_rate"] < prompt_only["loose_json_rate"]
    assert constrained["accuracy"]["concealment"]["unscorable"] == 0
    assert "prompt_only" in bench_json.table(result)


async def test_prefix_cache(mock_url, frames, connect):
    bench = await connect("prefix_cache", mock_url)
    result = await bench_prefix.run(bench, frames)

    warmups = bench.cfg["warmups"]
    assert warmups == 3
    check_files(bench, result, n_records=2 * (len(frames) - 1 + warmups))
    orderings = result["summary"]["orderings"]
    assert list(orderings) == ["shared_first", "unique_first"]
    for ordering in orderings.values():
        assert {
            "n", "errors", "mean_ttft_s", "p50_ttft_s", "p95_ttft_s", "mean_total_s",
            "mean_prompt_tokens", "mean_cached_tokens",
        } <= ordering.keys()  # fmt: skip
        assert ordering["n"] == len(frames) - 1  # the first request is the warm-up
        assert ordering["p50_ttft_s"] <= ordering["p95_ttft_s"] < ordering["mean_total_s"] * 2

    shared, unique = orderings["shared_first"], orderings["unique_first"]
    assert shared["mean_ttft_s"] < unique["mean_ttft_s"]
    assert shared["mean_cached_tokens"] > 1000 and unique["mean_cached_tokens"] == 0
    assert shared["cache_hits"] == shared["n"] and unique["cache_hits"] == 0
    assert "cache hits" in bench_prefix.table(result)
    # Same content in both orderings, so the prompt is the same size.
    assert shared["mean_prompt_tokens"] == pytest.approx(unique["mean_prompt_tokens"], abs=1)
    ratio = result["summary"]["ttft_ratio_unique_over_shared"]
    assert ratio["mean"] > 1 and ratio["p50"] > 1
    assert "TTFT unique_first / shared_first" in bench_prefix.table(result)


async def test_concurrency(mock_url, frames, connect):
    bench = await connect("concurrency", mock_url)
    result = await bench_concurrency.run(bench, frames)

    check_files(bench, result, n_records=2 * 8 + 2)
    levels = result["summary"]["levels"]
    assert [level["concurrency"] for level in levels] == [1, 4]
    for level in levels:
        assert {
            "n", "errors", "requests_per_s", "output_tokens_per_s", "p50_total_s", "p95_total_s",
            "p50_ttft_s", "p95_ttft_s", "usd_per_1k_alerts",
        } <= level.keys()  # fmt: skip
        assert level["n"] == 8 and level["errors"] == 0  # min_requests applies at both levels
        assert level["requests_per_s"] > 0 and level["output_tokens_per_s"] > 0
        assert level["p50_ttft_s"] < level["p95_total_s"]
        assert level["usd_per_1k_alerts"] is None  # no GPU price in the config
    assert levels[1]["requests_per_s"] > levels[0]["requests_per_s"]

    cost = result["summary"]["cost"]
    assert cost["gpu_usd_per_hour"] == "not set"
    assert cost["cloud_reference"]["image_tokens"] == 258  # 320x240 frame
    assert cost["cloud_reference"]["models"]["gemini-3.8-flash"]["usd_per_1k_alerts"] > 0
    text = bench_concurrency.table(result)
    assert "not set" in text and "gemini-3.1-flash-lite" in text


async def test_concurrency_cost_uses_the_gpu_price(mock_url, frames, connect, config):
    config["prices"]["gpu_usd_per_hour"] = 0.5
    config["concurrency"]["levels"] = [2]
    bench = await connect("concurrency", mock_url)
    result = await bench_concurrency.run(bench, frames)

    level = result["summary"]["levels"][0]
    expected = 0.5 / (level["requests_per_s"] * 3600) * 1000
    assert level["usd_per_1k_alerts"] == pytest.approx(expected)
    assert result["summary"]["cost"]["gpu_usd_per_hour"] == 0.5


async def test_concurrency_reports_repeated_prompts(mock_url, tmp_path, connect):
    bench = await connect("concurrency", mock_url)
    result = await bench_concurrency.run(bench, make_frames(tmp_path / "two", 2))
    assert result["summary"]["distinct_prompts"] == 2  # one question per frame
    assert [level["repeated_prompts"] for level in result["summary"]["levels"]] == [6, 6]


async def test_image_size(mock_url, frames, connect):
    bench = await connect("image_size", mock_url)
    result = await bench_image_size.run(bench, frames)

    check_files(bench, result, n_records=2 * len(frames))
    native, small = result["summary"]["sizes"]
    for size in (native, small):
        assert {
            "max_side", "actual_px", "n", "errors", "mean_prompt_tokens", "mean_ttft_s",
            "mean_total_s", "schema_valid_rate", "accuracy",
        } <= size.keys()  # fmt: skip
        assert size["n"] == len(frames) - 1 and size["schema_valid_rate"] == 1.0
        assert set(size["accuracy"]) == {"concealment", "aggression"}
        assert sum(task["n"] for task in size["accuracy"].values()) == size["n"]
    assert (native["max_side"], native["actual_px"]) == ("native", "320x240")
    assert (small["max_side"], small["actual_px"]) == ("128", "128x96")
    assert small["mean_prompt_tokens"] < native["mean_prompt_tokens"]
    assert "128x96" in bench_image_size.table(result)


async def test_in_event_labels_are_recorded(mock_url, tmp_path, connect):
    frames = make_frames(tmp_path / "annotated", 4, in_event=True) + make_frames(
        tmp_path / "weak", 2
    )
    bench = await connect("image_size", mock_url)
    result = await bench_image_size.run(bench, frames)

    assert result["meta"]["labels"] == {"in_event": 4, "weak": 2}
    accuracy = result["summary"]["sizes"][0]["accuracy"]
    # Frame 0 is the warm-up; of the five measured frames the last two have no annotation.
    assert sum(task["n"] for task in accuracy.values()) == 5
    assert sum(task["weak_labels"] for task in accuracy.values()) == 2
    raw = bench.results_dir / bench.variant / "raw" / "image_size.jsonl"
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    fight = next(r for r in records if r["label"] == "fighting" and r["truth_source"] == "in_event")
    assert (fight["task"], fight["truth"]) == ("aggression", True)


async def test_failed_requests_are_counted_not_raised(start_app, frames, connect):
    """A server that is up but rejects every completion, e.g. an unsupported response_format."""

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "broken-vlm", "owned_by": "sglang"}]})

    async def chat(_: web.Request) -> web.Response:
        return web.json_response({"error": {"message": "json_schema not supported"}}, status=400)

    app = web.Application(client_max_size=2**24)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    bench = await connect("json_validity", await start_app(app))
    result = await bench_json.run(bench, frames[:2])

    meta = result["meta"]
    assert meta["mock"] is False and meta["model"] == "broken-vlm"
    assert meta["errors"] == meta["n_requests"] == 18  # 2 frames x 3 tasks x 3 modes
    assert meta["cache_flush"] == "unsupported"
    constrained = result["summary"]["modes"]["constrained"]
    assert constrained["errors"] == 6 and constrained["schema_valid_rate"] is None
    assert constrained["accuracy"]["concealment"]["unscorable"] == 2
    assert "n/a" in bench_json.table(result)


async def test_connect_fails_clearly_when_the_server_is_down(connect):
    with pytest.raises(ConnectionError, match="/v1/models"):
        await connect("json_validity", "http://127.0.0.1:9")
