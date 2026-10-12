"""Statistics helpers that every reported number passes through."""

import pytest

from camalerts import common


def test_percentile_interpolates_between_ranks():
    values = [10, 20, 30, 40]
    assert common.percentile(values, 0) == 10
    assert common.percentile(values, 50) == 25  # halfway between the two middle values
    assert common.percentile(values, 95) == pytest.approx(38.5)
    assert common.percentile(values, 100) == 40


def test_percentile_ignores_missing_values_and_order():
    assert common.percentile([None, 3, 1, None, 2], 50) == 2
    assert common.percentile([7], 95) == 7
    assert common.percentile([], 50) is None and common.percentile([None], 50) is None


def test_mean_ignores_missing_values():
    assert common.mean([1, None, 3]) == 2
    assert common.mean([None]) is None


def record(cached, ok=True, warmup=False):
    return {"ok": ok, "warmup": warmup, "cached_tokens": cached}


def test_cache_use_counts_a_missing_report_as_zero():
    """SGLang omits cached_tokens when it is 0; averaging only the hits would overstate the cache."""
    rows = [record(None), record(None), record(None), record(1200), record(800)]
    assert common.cache_use(rows) == {"mean_cached_tokens": 400, "cache_hits": 2}


def test_cache_use_with_no_hits_is_zero_not_missing():
    assert common.cache_use([record(None), record(0)]) == {"mean_cached_tokens": 0, "cache_hits": 0}


def test_cache_use_skips_warmups_and_failed_requests():
    rows = [record(900, warmup=True), record(900, ok=False), record(300)]
    assert common.cache_use(rows) == {"mean_cached_tokens": 300, "cache_hits": 1}
    assert common.cache_use([]) == {"mean_cached_tokens": None, "cache_hits": 0}


def test_accuracy_cell_shows_recall_precision_and_what_was_scored():
    scores = {"concealment": {"n": 40, "scored": 38, "accuracy": 0.8, "tp": 3, "fn": 5, "fp": 2}}
    cell = common.accuracy_cell(scores)
    assert cell == "concealment: acc 80.0%, recall 3/8, precision 3/5, scored 38/40"
    assert common.accuracy_cell({}) == "n/a"
