import json

import pytest

from camalerts.schema import check_answer, extract_object

GOOD = {"event": True, "confidence": "high", "people": 2, "reason": "Person pockets an item."}


@pytest.fixture
def schema(alerts):
    return alerts.schema


def test_valid_answer(schema):
    check = check_answer(json.dumps(GOOD), schema)
    assert (check.strict_json, check.schema_valid, check.loose_json) == (True, True, True)
    assert check.parsed == GOOD


def test_fenced_answer_is_only_loose(schema):
    check = check_answer(f"```json\n{json.dumps(GOOD)}\n```", schema)
    assert (check.strict_json, check.schema_valid, check.loose_json) == (False, False, True)
    assert check.parsed == GOOD


def test_trailing_comment_is_only_loose(schema):
    check = check_answer(json.dumps(GOOD) + "\n// partly blocked view", schema)
    assert (check.strict_json, check.schema_valid, check.loose_json) == (False, False, True)
    assert check.parsed == GOOD


def test_json_inside_prose_is_extracted(schema):
    check = check_answer(
        f"Sure! Here it is: {json.dumps(GOOD)} Hope that helps {{not json}}", schema
    )
    assert (check.strict_json, check.loose_json) == (False, True)
    assert check.parsed == GOOD


@pytest.mark.parametrize("text", ["The aisle is empty.", "", "{broken", "see {this} and {that}"])
def test_prose_has_no_json(schema, text):
    check = check_answer(text, schema)
    assert (check.strict_json, check.schema_valid, check.loose_json) == (False, False, False)
    assert check.parsed is None


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "people": "2"},
        {**GOOD, "event": "yes"},
        {**GOOD, "event": 1},
        {**GOOD, "confidence": "certain"},
        {**GOOD, "people": -1},
        {**GOOD, "people": 1.5},
        {**GOOD, "reason": "x" * 161},
        {**GOOD, "camera": "cam-01"},
        {k: v for k, v in GOOD.items() if k != "reason"},
    ],
    ids=[
        "people-string", "event-string", "event-int", "confidence-enum", "people-negative",
        "people-float", "reason-too-long", "extra-field", "missing-field",
    ],
)  # fmt: skip
def test_parses_but_breaks_schema(schema, bad):
    check = check_answer(json.dumps(bad), schema)
    assert (check.strict_json, check.schema_valid, check.loose_json) == (True, False, True)


def test_reason_at_the_limit_is_valid(schema):
    assert check_answer(json.dumps({**GOOD, "reason": "x" * 160}), schema).schema_valid


def test_json_that_is_not_an_object(schema):
    check = check_answer("[1, 2]", schema)
    assert (check.strict_json, check.schema_valid) == (True, False)


def test_extract_prefers_the_fenced_block():
    text = 'Earlier guess {"event": false}\n```json\n{"event": true}\n```'
    assert extract_object(text) == {"event": True}
