"""Cost arithmetic. Pure functions: no I/O, no defaults for prices."""

from __future__ import annotations

import math
from typing import Any


def self_hosted_usd_per_1k_alerts(
    gpu_usd_per_hour: float | None, requests_per_second: float | None
) -> float | None:
    """GPU rent spread over the alerts it serves. None when the price or the rate is missing."""
    if gpu_usd_per_hour is None or not requests_per_second or requests_per_second <= 0:
        return None
    return gpu_usd_per_hour / (requests_per_second * 3600) * 1000


def image_tokens(
    width: int,
    height: int,
    small_max_side: int = 384,
    tokens_per_tile: int = 258,
    tile_side: int = 768,
) -> int:
    """Cloud image token rule: a small image is one flat charge, a larger one is charged per tile."""
    if width <= small_max_side and height <= small_max_side:
        return tokens_per_tile
    return math.ceil(width / tile_side) * math.ceil(height / tile_side) * tokens_per_tile


def text_tokens(n_chars: int, chars_per_token: float = 4) -> int:
    """Estimate from character count. Not a tokeniser."""
    return math.ceil(n_chars / chars_per_token)


def cloud_usd_per_call(
    input_tokens: float, output_tokens: float, input_usd_per_1m: float, output_usd_per_1m: float
) -> float:
    return (input_tokens * input_usd_per_1m + output_tokens * output_usd_per_1m) / 1_000_000


def cloud_reference(
    cloud_prices: dict[str, Any],
    *,
    width: int | None,
    height: int | None,
    prompt_chars: int,
    output_tokens: float | None,
) -> dict[str, Any]:
    """Per-model cloud cost for one alert of this shape. Costs are None without output tokens.

    `width` and `height` are None for a request that carried no image (text-only mode).
    """
    rule = cloud_prices["image_tokens"]
    has_image = width is not None and height is not None
    img = 0
    if has_image:
        img = image_tokens(
            width, height, rule["small_max_side"], rule["tokens_per_tile"], rule["tile_side"]
        )
    txt = text_tokens(prompt_chars, cloud_prices["chars_per_text_token"])
    models = {}
    for name, price in cloud_prices["models"].items():
        usd = None
        if output_tokens is not None:
            per_call = cloud_usd_per_call(
                img + txt, output_tokens, price["input_usd_per_1m"], price["output_usd_per_1m"]
            )
            usd = per_call * 1000
        models[name] = {**price, "usd_per_1k_alerts": usd}
    return {
        "as_of": cloud_prices["as_of"],
        "source": cloud_prices["source"],
        "image_px": f"{width}x{height}" if has_image else "no image",
        "image_tokens": img,
        "prompt_text_tokens_estimated": txt,
        "output_tokens": output_tokens,
        "models": models,
    }
