"""Check a model answer: is it JSON, does it match the schema, and can JSON be dug out of it."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class AnswerCheck:
    strict_json: bool  # the whole text parses as JSON
    schema_valid: bool  # strict_json and the value passes the schema
    loose_json: bool  # strict_json, or a JSON object could be extracted from the text
    parsed: Any = None  # the strict value if any, else the extracted object, else None


def check_answer(text: str, schema: dict[str, Any]) -> AnswerCheck:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        extracted = extract_object(text) if isinstance(text, str) else None
        return AnswerCheck(False, False, extracted is not None, extracted)
    return AnswerCheck(True, Draft202012Validator(schema).is_valid(value), True, value)


def extract_object(text: str) -> dict[str, Any] | None:
    """First JSON object found inside a markdown fence, or else anywhere in the text."""
    for candidate in [*_FENCE.findall(text), text]:
        found = _first_object(candidate)
        if found is not None:
            return found
    return None


def _first_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None
