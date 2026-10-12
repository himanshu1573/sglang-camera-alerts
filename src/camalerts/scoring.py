"""Turn answers into accuracy numbers for the scored alert tasks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

IN_EVENT = "in_event"  # per-frame annotation
WEAK = "weak"  # the video's class only: not every frame of an event video shows the act


def ground_truth(frame: Any, positive_labels: Sequence[str]) -> tuple[bool, str]:
    """(truth, source) for one frame and one task.

    A frame whose video class is not one of the task's positive labels is a negative for that
    task. Otherwise `in_event` decides when the manifest has it, and the class alone when not.
    """
    in_event = getattr(frame, "in_event", None)
    source = WEAK if in_event is None else IN_EVENT
    if frame.label not in positive_labels:
        return False, source
    return (True if in_event is None else bool(in_event)), source


def predicted_event(parsed: Any) -> bool | None:
    """The answer's `event` value, or None when the answer cannot be scored."""
    if isinstance(parsed, dict) and isinstance(parsed.get("event"), bool):
        return parsed["event"]
    return None


def score(pairs: Iterable[tuple[bool, bool | None]]) -> dict[str, Any]:
    """Score (truth, predicted) pairs. A predicted value of None is counted as unscorable."""
    tp = fp = tn = fn = unscorable = 0
    for truth, predicted in pairs:
        if predicted is None:
            unscorable += 1
        elif truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    scored = tp + fp + tn + fn
    return {
        "n": scored + unscorable,
        "scored": scored,
        "unscorable": unscorable,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / scored if scored else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def score_records(records: Iterable[dict[str, Any]], task_ids: Sequence[str]) -> dict[str, Any]:
    """Per-task scores from request records carrying `task`, `truth`, `truth_source`, `predicted`."""
    records = list(records)
    out: dict[str, Any] = {}
    for task_id in task_ids:
        rows = [r for r in records if r["task"] == task_id]
        result = score((r["truth"], r["predicted"]) for r in rows)
        result["weak_labels"] = sum(r["truth_source"] == WEAK for r in rows)
        out[task_id] = result
    return out
