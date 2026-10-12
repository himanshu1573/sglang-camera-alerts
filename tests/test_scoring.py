from types import SimpleNamespace

import pytest

from camalerts import scoring


def frame(label, in_event=None):
    return SimpleNamespace(frame="x.jpg", label=label, in_event=in_event)


@pytest.mark.parametrize(
    ("label", "in_event", "expected"),
    [
        ("shoplifting", None, (True, "weak")),  # the video class is all we know
        ("normal", None, (False, "weak")),
        ("shoplifting", True, (True, "in_event")),
        ("shoplifting", False, (False, "in_event")),  # event video, frame outside the event
        ("fighting", True, (False, "in_event")),  # a fight is not a concealment
        ("normal", False, (False, "in_event")),
    ],
)
def test_ground_truth(label, in_event, expected):
    assert scoring.ground_truth(frame(label, in_event), ["shoplifting"]) == expected


def test_ground_truth_accepts_objects_without_in_event():
    assert scoring.ground_truth(SimpleNamespace(label="fighting"), ["fighting"]) == (True, "weak")


@pytest.mark.parametrize(
    ("parsed", "expected"),
    [
        ({"event": True}, True),
        ({"event": False}, False),
        ({"event": "yes"}, None),
        ({"event": 1}, None),
        ({}, None),
        (None, None),
        ([True], None),
    ],
)
def test_predicted_event(parsed, expected):
    assert scoring.predicted_event(parsed) is expected


def test_score_counts_and_rates():
    pairs = [(True, True), (True, True), (True, False), (False, True), (False, False), (True, None)]
    result = scoring.score(pairs)
    assert result == {
        "n": 6, "scored": 5, "unscorable": 1, "tp": 2, "fp": 1, "tn": 1, "fn": 1,
        "accuracy": 3 / 5, "precision": 2 / 3, "recall": 2 / 3,
    }  # fmt: skip


def test_score_with_nothing_to_score():
    result = scoring.score([(True, None)])
    assert result["unscorable"] == 1
    assert result["accuracy"] is None and result["precision"] is None and result["recall"] is None


def test_score_records_per_task():
    records = [
        {"task": "concealment", "truth": True, "truth_source": "weak", "predicted": True},
        {"task": "concealment", "truth": False, "truth_source": "in_event", "predicted": True},
        {"task": "aggression", "truth": False, "truth_source": "weak", "predicted": None},
        {"task": "presence", "truth": None, "truth_source": "weak", "predicted": True},
    ]
    result = scoring.score_records(records, ["concealment", "aggression"])
    assert set(result) == {"concealment", "aggression"}
    assert result["concealment"]["accuracy"] == 0.5
    assert result["concealment"]["weak_labels"] == 1
    assert result["aggression"]["unscorable"] == 1 and result["aggression"]["weak_labels"] == 1
