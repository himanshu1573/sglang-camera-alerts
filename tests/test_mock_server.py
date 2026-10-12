import json

from camalerts.mock_server import MockConfig, PrefixCache, flatten, make_answer
from camalerts.schema import check_answer


def test_prefix_cache_needs_a_long_shared_start():
    cache = PrefixCache()
    brief = "site rule. " * 400  # 4,400 characters
    assert cache.cached_chars(brief + "frame A") == 0  # nothing seen yet
    assert cache.cached_chars(brief + "frame B") == 4000  # whole 500-character blocks only
    assert cache.cached_chars("cam-07 " + brief) == 0  # a different first line shares nothing
    assert cache.cached_chars("short prompt") == 0
    cache.clear()
    assert cache.cached_chars(brief + "frame A") == 0


def test_prefix_cache_ignores_a_short_shared_start():
    cache = PrefixCache()
    start = "x" * 1500
    cache.cached_chars(start + "a" * 3000)
    assert cache.cached_chars(start + "b" * 3000) == 0  # 1,500 shared characters < 2,000


def test_answers_are_deterministic_and_seeded(alerts):
    cfg = MockConfig(seed=0)
    prompts_ = [f"prompt {i}" for i in range(200)]
    first = [make_answer(p, True, cfg) for p in prompts_]
    assert first == [make_answer(p, True, cfg) for p in prompts_]
    assert first != [make_answer(p, True, MockConfig(seed=1)) for p in prompts_]
    assert all(check_answer(text, alerts.schema).schema_valid for text in first)
    assert {json.loads(text)["event"] for text in first} == {True, False}


def test_unconstrained_answers_are_sometimes_broken(alerts):
    cfg = MockConfig(seed=0)
    checks = [
        check_answer(make_answer(f"prompt {i}", False, cfg), alerts.schema) for i in range(400)
    ]
    strict = sum(c.strict_json for c in checks)
    valid = sum(c.schema_valid for c in checks)
    loose = sum(c.loose_json for c in checks)
    assert 0 < valid < strict < loose < len(checks)  # wrong type < fenced/comment < prose


def test_flatten_replaces_images_with_a_marker():
    url = "data:image/jpeg;base64,AAAA"
    messages = [
        {"role": "system", "content": "brief"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": "question"}]},
    ]  # fmt: skip
    text, image_tokens = flatten(messages)
    assert text.startswith("<system>\nbrief<user>\n<image:") and text.endswith(">question")
    assert "AAAA" not in text and image_tokens == 0  # not a decodable image
