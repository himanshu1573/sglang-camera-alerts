import pytest
from conftest import REPO

from camalerts import common, cost


@pytest.mark.parametrize(
    ("width", "height", "tokens"),
    [
        (320, 240, 258),  # both sides <= 384: flat charge
        (384, 384, 258),
        (385, 100, 258),  # one 768 tile
        (769, 769, 4 * 258),
        (1920, 1080, 1548),  # ceil(1920/768)=3 x ceil(1080/768)=2 = 6 tiles
    ],
)
def test_image_tokens(width, height, tokens):
    assert cost.image_tokens(width, height) == tokens


def test_text_tokens_round_up():
    assert cost.text_tokens(8) == 2
    assert cost.text_tokens(9) == 3


def test_self_hosted_cost():
    # 0.36 USD/h at 10 req/s = 36,000 alerts/h -> 0.00001 USD each -> 0.01 USD per 1,000
    assert cost.self_hosted_usd_per_1k_alerts(0.36, 10) == pytest.approx(0.01)
    # 0.50 USD/h at 2 req/s = 7,200 alerts/h -> 0.0694 USD per 1,000
    assert cost.self_hosted_usd_per_1k_alerts(0.50, 2) == pytest.approx(0.069444, abs=1e-6)


@pytest.mark.parametrize(("price", "rps"), [(None, 10), (0.5, 0), (0.5, None)])
def test_self_hosted_cost_is_none_without_inputs(price, rps):
    assert cost.self_hosted_usd_per_1k_alerts(price, rps) is None


def test_cloud_cost_per_call():
    # 2,000 input tokens at 0.75/1M = 0.0015; 40 output tokens at 3.75/1M = 0.00015
    assert cost.cloud_usd_per_call(2000, 40, 0.75, 3.75) == pytest.approx(0.00165)


def test_cloud_reference_uses_the_configured_prices():
    prices = common.load_yaml(REPO / "configs" / "bench.yaml")["prices"]
    assert prices["gpu_usd_per_hour"] is None
    # 320x240 frame = 258 image tokens; 6,968 characters / 4 = 1,742 text tokens; 2,000 in total
    ref = cost.cloud_reference(
        prices["cloud"], width=320, height=240, prompt_chars=6968, output_tokens=40
    )
    assert ref["image_tokens"] == 258
    assert ref["prompt_text_tokens_estimated"] == 1742
    assert ref["as_of"] == "2026-10-12"
    # flash: (2000 x 0.75 + 40 x 3.75) / 1M x 1000 = 1.65; flash-lite: (500 + 60) / 1000 = 0.56
    assert ref["models"]["gemini-3.8-flash"]["usd_per_1k_alerts"] == pytest.approx(1.65)
    assert ref["models"]["gemini-3.1-flash-lite"]["usd_per_1k_alerts"] == pytest.approx(0.56)


def test_cloud_reference_without_output_tokens_has_no_cost():
    prices = common.load_yaml(REPO / "configs" / "bench.yaml")["prices"]
    ref = cost.cloud_reference(
        prices["cloud"], width=1920, height=1080, prompt_chars=400, output_tokens=None
    )
    assert ref["image_tokens"] == 1548
    assert all(m["usd_per_1k_alerts"] is None for m in ref["models"].values())
