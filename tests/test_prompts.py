import base64
import io
import os
from pathlib import Path

import pytest
from conftest import make_frames
from PIL import Image

from camalerts import prompts


def messages(alerts, ordering, nonce, instructions="schema", image="AAAA"):
    return prompts.build_messages(
        site_brief=alerts.site_brief,
        schema=alerts.schema,
        question=alerts.tasks[0].question,
        image_b64=image,
        ordering=ordering,
        instructions=instructions,
        unique_line=prompts.request_line("cam-01", nonce),
    )


def test_message_layout(alerts):
    system, user = prompts.build_messages(
        site_brief=alerts.site_brief,
        schema=alerts.schema,
        question="Is anyone there?",
        image_b64="QUJD",
    )
    assert system["role"] == "system" and system["content"].startswith(alerts.site_brief.strip())
    assert '"additionalProperties": false' in system["content"]
    assert [part["type"] for part in user["content"]] == ["image_url", "text"]
    assert user["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert user["content"][1]["text"] == "Is anyone there?"


def test_shared_first_requests_share_a_long_prefix(alerts):
    a = messages(alerts, "shared_first", "n1", image="AAAA")[0]["content"]
    b = messages(alerts, "shared_first", "n2", image="BBBB")[0]["content"]
    assert a == b
    assert len(os.path.commonprefix([a, b])) > 5000


def test_unique_first_requests_share_almost_nothing(alerts):
    a = messages(alerts, "unique_first", "8f1c")[0]["content"]
    b = messages(alerts, "unique_first", "27ab")[0]["content"]
    assert a.startswith("Request: camera=cam-01 nonce=8f1c\n")
    assert len(os.path.commonprefix([a, b])) < 40


def test_both_orderings_carry_the_same_content(alerts):
    line = prompts.request_line("cam-01", "n1")
    shared_system, shared_user = messages(alerts, "shared_first", "n1")
    unique_system, unique_user = messages(alerts, "unique_first", "n1")
    assert unique_system["content"] == line + "\n\n" + shared_system["content"]
    assert shared_user["content"][1]["text"] == unique_user["content"][1]["text"] + "\n" + line


def test_unique_first_needs_a_line_and_orderings_are_checked(alerts):
    kwargs = {"site_brief": "b", "schema": alerts.schema, "question": "q", "image_b64": "AAAA"}
    with pytest.raises(ValueError, match="unique_line"):
        prompts.build_messages(**kwargs, ordering="unique_first")
    with pytest.raises(ValueError, match="unknown ordering"):
        prompts.build_messages(**kwargs, ordering="shuffled")


def test_unconstrained_instructions_only_ask_in_words(alerts):
    system = messages(alerts, "shared_first", "n1", instructions="unconstrained")[0]["content"]
    assert system.endswith("Answer in JSON with these fields: event, confidence, people, reason.")
    assert "additionalProperties" not in system


def test_response_format(alerts):
    fmt = prompts.response_format(alerts.schema, "camera_alert")
    assert fmt == {
        "type": "json_schema",
        "json_schema": {"name": "camera_alert", "schema": alerts.schema},
    }


def decoded_size(image: prompts.EncodedImage) -> tuple[int, int]:
    with Image.open(io.BytesIO(base64.b64decode(image.b64))) as img:
        assert img.format == "JPEG"
        return img.size


@pytest.mark.parametrize(
    ("size", "max_side", "expected"),
    [
        ((320, 240), 256, (256, 192)),
        ((320, 240), 128, (128, 96)),
        ((240, 320), 128, (96, 128)),  # portrait: the longer side is the height
        ((320, 240), 320, (320, 240)),
    ],
)
def test_downscale_limits_the_longer_side(tmp_path, size, max_side, expected):
    frame = make_frames(tmp_path, 1, size=size)[0]
    image = prompts.encode_frame(frame.frame, max_side)
    assert (image.width, image.height) == expected == decoded_size(image)


@pytest.mark.parametrize("max_side", [None, 640, 4000])
def test_never_upscales_and_sends_the_original_bytes(tmp_path, max_side):
    frame = make_frames(tmp_path, 1)[0]
    image = prompts.encode_frame(frame.frame, max_side)
    assert (image.width, image.height) == (320, 240)
    assert base64.b64decode(image.b64) == Path(frame.frame).read_bytes()
