"""Build chat messages for one camera frame and one alert question."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

ORDERINGS = ("shared_first", "unique_first")
INSTRUCTIONS = ("schema", "unconstrained")


@dataclass(frozen=True)
class EncodedImage:
    b64: str
    width: int
    height: int


def encode_frame(path: str | Path, max_side: int | None = None, quality: int = 90) -> EncodedImage:
    """Base64 JPEG of a frame, downscaled so the longer side is <= max_side. Never upscales.

    A frame that needs no resizing is sent as the original file bytes, with no re-encoding.
    """
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as img:
        width, height = img.size
        longer = max(width, height)
        if max_side is not None and longer > max_side:
            scale = max_side / longer
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
            buf = io.BytesIO()
            img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(
                buf, format="JPEG", quality=quality
            )
            raw = buf.getvalue()
    return EncodedImage(base64.b64encode(raw).decode("ascii"), width, height)


def output_instructions(schema: dict[str, Any], variant: str = "schema") -> str:
    """`schema` pastes the JSON schema into the prompt; `unconstrained` only names the fields."""
    if variant == "schema":
        return (
            "Answer with one JSON object that matches this JSON schema, and nothing else:\n"
            + json.dumps(schema, sort_keys=True)
        )
    if variant == "unconstrained":
        return "Answer in JSON with these fields: " + ", ".join(schema["properties"]) + "."
    raise ValueError(f"unknown instructions variant {variant!r}; expected one of {INSTRUCTIONS}")


def request_line(camera_id: str, nonce: str) -> str:
    return f"Request: camera={camera_id} nonce={nonce}"


def response_format(schema: dict[str, Any], name: str = "camera_alert") -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def build_messages(
    *,
    site_brief: str,
    schema: dict[str, Any],
    question: str,
    image_b64: str | None,
    ordering: str = "shared_first",
    instructions: str = "schema",
    unique_line: str | None = None,
    scene_text: str | None = None,
) -> list[dict[str, Any]]:
    """System message (site brief + output instructions), then user message [image, question].

    `unique_line` is a per-request line. With `unique_first` it is required and opens the system
    message, so no two requests share a prefix. With `shared_first` it is optional and goes after
    the question, so both orderings carry the same content and differ only in where it sits.

    With `image_b64=None` no image is sent and `scene_text` stands in for the frame. That is a
    plumbing test for text-only servers; it says nothing about how a model reads pictures.
    """
    system = site_brief.strip() + "\n\n" + output_instructions(schema, instructions)
    user_text = question
    if ordering == "unique_first":
        if not unique_line:
            raise ValueError("unique_first needs a unique_line")
        system = unique_line + "\n\n" + system
    elif ordering == "shared_first":
        if unique_line:
            user_text = question + "\n" + unique_line
    else:
        raise ValueError(f"unknown ordering {ordering!r}; expected one of {ORDERINGS}")

    if image_b64 is None:
        if not scene_text:
            raise ValueError("a request without an image needs scene_text")
        text = f"Scene description: {scene_text}\n{user_text}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]

    image_url = {"url": f"data:image/jpeg;base64,{image_b64}"}
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": user_text},
            ],
        },
    ]
