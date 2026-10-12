"""Text-only mode: a written scene stands in for the frame, for servers that cannot read images."""

from types import SimpleNamespace

import pytest

from camalerts import (
    bench_concurrency,
    bench_image_size,
    bench_json,
    bench_prefix,
    common,
    cost,
    prompts,
    report,
)


def test_messages_without_image_carry_the_scene(alerts):
    system, user = prompts.build_messages(
        site_brief="brief",
        schema=alerts.schema,
        question="Is there a fight?",
        image_b64=None,
        scene_text="Two people push each other.",
    )
    assert system["role"] == "system"
    assert [part["type"] for part in user["content"]] == ["text"]
    assert "Two people push each other." in user["content"][0]["text"]
    assert user["content"][0]["text"].endswith("Is there a fight?")


def test_messages_without_image_need_a_scene(alerts):
    with pytest.raises(ValueError, match="scene_text"):
        prompts.build_messages(site_brief="b", schema=alerts.schema, question="q", image_b64=None)


@pytest.mark.parametrize(
    ("label", "in_event", "expected_pool"),
    [
        ("shoplifting", True, "shoplifting"),
        ("shoplifting", None, "shoplifting"),
        ("shoplifting", False, "normal"),  # frame is outside the annotated event
        ("fighting", True, "fighting"),
        ("normal", False, "normal"),
        ("vandalism", None, "normal"),  # no scenes written for this label
    ],
)
def test_scene_matches_ground_truth(alerts, label, in_event, expected_pool):
    frame = SimpleNamespace(
        frame=f"data/frames/{label}/v/000001.jpg", label=label, in_event=in_event
    )
    scene = common.scene_for(frame, alerts)
    assert scene in alerts.text_scenes[expected_pool]
    assert scene == common.scene_for(frame, alerts)  # deterministic


async def test_json_benchmark_in_text_only_mode(mock_url, frames, alerts, config, tmp_path):
    config["request"]["text_only"] = True
    bench = await common.connect(
        "json_validity",
        base_url=mock_url,
        variant="text-test",
        results_dir=tmp_path / "results",
        alerts=alerts,
        config=config,
    )
    # A path that does not exist proves no image is read or sent.
    ghosts = [SimpleNamespace(frame=f"missing/{i}.jpg", label=f.label, in_event=None)
              for i, f in enumerate(frames)]  # fmt: skip
    result = await bench_json.run(bench, ghosts)

    assert result["meta"]["text_only"] is True
    assert result["meta"]["errors"] == 0
    assert result["summary"]["modes"]["constrained"]["schema_valid_rate"] == 1.0

    text = report.render(tmp_path / "results")
    assert report.TEXT_ONLY_LABEL in text


async def test_concurrency_benchmark_in_text_only_mode(mock_url, frames, alerts, config, tmp_path):
    """Regression: the cost block needs an image size, and a text-only request has none."""
    config["request"]["text_only"] = True
    bench = await common.connect(
        "concurrency",
        base_url=mock_url,
        variant="text-test",
        results_dir=tmp_path / "results",
        alerts=alerts,
        config=config,
    )
    result = await bench_concurrency.run(bench, frames)

    cloud = result["summary"]["cost"]["cloud_reference"]
    assert cloud["image_tokens"] == 0 and cloud["image_px"] == "no image"
    assert all(m["usd_per_1k_alerts"] > 0 for m in cloud["models"].values())
    assert report.render(tmp_path / "results")  # the cloud table renders without an image size


async def test_prefix_benchmark_in_text_only_mode(mock_url, frames, alerts, config, tmp_path):
    config["request"]["text_only"] = True
    bench = await common.connect(
        "prefix_cache",
        base_url=mock_url,
        variant="text-test",
        results_dir=tmp_path / "results",
        alerts=alerts,
        config=config,
    )
    result = await bench_prefix.run(bench, frames)
    assert result["meta"]["text_only"] is True and result["meta"]["errors"] == 0


def test_cloud_reference_without_an_image(config):
    with_image = cost.cloud_reference(
        config["prices"]["cloud"], width=320, height=240, prompt_chars=4000, output_tokens=40
    )
    without = cost.cloud_reference(
        config["prices"]["cloud"], width=None, height=None, prompt_chars=4000, output_tokens=40
    )
    assert with_image["image_tokens"] == 258 and without["image_tokens"] == 0
    for name, model in without["models"].items():
        assert model["usd_per_1k_alerts"] < with_image["models"][name]["usd_per_1k_alerts"]


def test_image_size_is_skipped_in_text_only_mode(capsys):
    code = bench_image_size.main(["--variant", "x", "--text-only", "--manifest", "nope.jsonl"])
    assert code == 0
    assert "skipped" in capsys.readouterr().out
