"""Collect results/<variant>/*.json into results/REPORT.md.

python -m camalerts.report [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from camalerts.common import NA, accuracy_cell, format_table, ms, num, pct, utc_now

MOCK_LABEL = "MOCK - not a real measurement"
TEXT_ONLY_LABEL = "TEXT-ONLY - no images sent"
NOT_RUN = "not run"
NOT_SET = "not set"

Row = list[Any]


@dataclass
class Variant:
    name: str
    tests: dict[str, dict[str, Any]] = field(default_factory=dict)
    gpu_mem: dict[str, Any] | None = None
    server: dict[str, Any] | None = None
    server_error: dict[str, Any] | None = None

    @property
    def mock(self) -> bool:
        return any(t["meta"].get("mock") for t in self.tests.values())

    @property
    def text_only(self) -> bool:
        return any(t["meta"].get("text_only") for t in self.tests.values())

    @property
    def label(self) -> str:
        tags = [MOCK_LABEL] * self.mock + [TEXT_ONLY_LABEL] * self.text_only
        return f"{self.name} ({'; '.join(tags)})" if tags else self.name


def _read(path: Path, problems: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"`{path.parent.name}/{path.name}` could not be read: {exc}")
        return None
    return data if isinstance(data, dict) else None


def _field(data: dict[str, Any] | None, key: str) -> Any:
    """Look a key up at the top level, then under `summary`. Other owners write these files."""
    if not data:
        return None
    if key in data:
        return data[key]
    return (data.get("summary") or {}).get(key)


def load_variants(results_dir: Path, problems: list[str]) -> list[Variant]:
    variants = []
    for folder in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        variant = Variant(folder.name)
        for test in SECTIONS:
            data = _read(folder / f"{test}.json", problems)
            if data and "meta" in data and "summary" in data:
                variant.tests[test] = data
        variant.gpu_mem = _read(folder / "gpu_mem.json", problems)
        variant.server = _read(folder / "server.json", problems)
        variant.server_error = _read(folder / "server_error.json", problems)
        if variant.tests or variant.gpu_mem or variant.server or variant.server_error:
            variants.append(variant)
    return variants


def failure_reason(error: dict[str, Any]) -> str:
    """One line for a variant that did not start: the recorded reason and the last log line."""
    reason = "no reason recorded"
    for key in ("error", "reason", "message", "detail"):
        value = _field(error, key)
        if value:
            lines = [line.strip() for line in str(value).splitlines() if line.strip()]
            if lines:
                reason = lines[-1][:200]
                break
    # serve.py records "server exited with code 1"; the cause is the end of the server log.
    tail = [str(line).strip() for line in (_field(error, "log_tail") or []) if str(line).strip()]
    if tail and tail[-1][:200] != reason:
        reason += f"; last log line: {tail[-1][:200]}"
    return reason.replace("|", "\\|")  # a bare pipe would break the Markdown table


# --- one function per section: each returns (headers, rows for one variant) ---------------------


def variants_table(variants: Sequence[Variant]) -> str:
    # `KV pool tokens` must be equal across variants for the GPU MB columns to be comparable:
    # SGLang fills spare memory with cache, so a different pool means a different measurement.
    headers = [
        "variant", "served model", "HF model", "quantization", "load s", "weights GB",
        "KV pool tokens", "GPU idle MB", "GPU peak MB", "GPU total MB", "status",
    ]  # fmt: skip
    rows = []
    for v in variants:
        model = next((t["meta"].get("model") for t in v.tests.values()), None)
        if v.gpu_mem is None:
            gpu = [NOT_RUN] * 3
        elif _field(v.gpu_mem, "available") is False:
            gpu = ["not available"] * 3
        else:
            gpu = [num(_field(v.gpu_mem, k), 0) for k in ("idle_mb", "peak_mb", "total_mb")]
        if v.server_error is not None:
            status = f"FAILED to start: {failure_reason(v.server_error)}"
        else:
            status = f"{len(v.tests)}/{len(SECTIONS)} tests" if v.tests else "no benchmark results"
        rows.append(
            [
                v.label,
                model or NA,
                _field(v.server, "hf_model") or NA,
                _field(v.server, "quantization") or NA,
                num(_field(v.server, "load_seconds")),
                num(_field(v.server, "weights_mem_gb"), 2),
                _field(v.server, "max_total_num_tokens") or NA,
                *gpu,
                status,
            ]
        )
    return format_table(headers, rows, markdown=True)


def json_rows(summary: dict[str, Any]) -> list[Row]:
    return [
        [
            mode,
            s["instructions"],
            s["n"],
            s["errors"],
            pct(s["strict_json_rate"]),
            pct(s["schema_valid_rate"]),
            pct(s["loose_json_rate"]),
            num(s["mean_completion_tokens"]),
            s.get("truncated", NA),
            ms(s["mean_total_s"]),
            accuracy_cell(s["accuracy"]),
        ]
        for mode, s in summary["modes"].items()
    ]


def prefix_rows(summary: dict[str, Any]) -> list[Row]:
    ratio = summary.get("ttft_ratio_unique_over_shared") or {}
    return [
        [
            ordering,
            s["n"],
            s["errors"],
            ms(s["mean_ttft_s"]),
            ms(s["p50_ttft_s"]),
            ms(s["p95_ttft_s"]),
            ms(s["mean_total_s"]),
            num(s["mean_prompt_tokens"]),
            num(s["mean_cached_tokens"]),
            f"{s['cache_hits']}/{s['n'] - s['errors']}" if "cache_hits" in s else NA,
            num(ratio.get("mean"), 2) if ordering == "unique_first" else "",
        ]
        for ordering, s in summary["orderings"].items()
    ]


def usd(value: float | None) -> str:
    return NOT_SET if value is None else f"{value:.4f}"


def concurrency_rows(summary: dict[str, Any]) -> list[Row]:
    return [
        [
            s["concurrency"],
            s["n"],
            s["repeated_prompts"],
            s["errors"],
            num(s["requests_per_s"], 2),
            num(s["output_tokens_per_s"]),
            ms(s["p50_total_s"]),
            ms(s["p95_total_s"]),
            ms(s["p50_ttft_s"]),
            ms(s["p95_ttft_s"]),
            num(s.get("mean_cached_tokens")),
            usd(s["usd_per_1k_alerts"]),
        ]
        for s in summary["levels"]
    ]


def image_rows(summary: dict[str, Any]) -> list[Row]:
    return [
        [
            s["max_side"],
            s["actual_px"] or NA,
            s["n"],
            s["errors"],
            num(s["mean_prompt_tokens"]),
            ms(s["mean_ttft_s"]),
            ms(s["mean_total_s"]),
            pct(s["schema_valid_rate"]),
            accuracy_cell(s["accuracy"]),
        ]
        for s in summary["sizes"]
    ]


@dataclass(frozen=True)
class Section:
    title: str
    intro: str
    headers: tuple[str, ...]
    rows: Callable[[dict[str, Any]], list[Row]]


SECTIONS: dict[str, Section] = {
    "json_validity": Section(
        "JSON validity",
        "The same frames and questions in each mode. `constrained` sends `response_format`; the "
        "other modes do not. `prompt asks with` is the output instruction in the prompt: `schema` "
        "pastes the JSON schema, `unconstrained` only names the fields. Compare `constrained` "
        "with `schema_prompt_only` to see the decoder alone, since their prompts are identical. "
        "Rates are over answered requests. `truncated` counts answers cut off by the token "
        "limit. The modes are sent mixed together, several at a time, so `mean total ms` is a "
        "rough guide and not a clean per-mode latency. In the accuracy cell, recall and "
        "precision are counts, and `scored` is how many answers held a usable `event` value; "
        "answers that could not be scored are left out of accuracy, so compare modes only when "
        "their `scored` counts match.",
        ("mode", "prompt asks with", "n", "errors", "strict JSON", "schema valid", "loose JSON",
         "mean out tokens", "truncated", "mean total ms (mixed load)", "accuracy"),
        json_rows,
    ),
    "prefix_cache": Section(
        "Prefix cache",
        "Sequential requests. `shared_first` starts with the site brief; `unique_first` starts "
        "with a per-request line, so nothing can be reused. The ratio is mean TTFT of "
        "unique_first over shared_first. `cached tokens` is the mean the server reported, "
        "counting a request with no report as 0, and `cache hits` is how many requests reported "
        "any. A server that never reports the field shows 0 hits even if it does cache.",
        ("ordering", "n", "errors", "mean TTFT ms", "p50 TTFT ms", "p95 TTFT ms",
         "mean total ms", "prompt tokens", "cached tokens", "cache hits", "TTFT ratio"),
        prefix_rows,
    ),
    "concurrency": Section(
        "Concurrency",
        "Closed loop: C workers send requests back to back, one question per frame. `repeats` is "
        "how many requests reused a prompt already sent at that level, which a prefix cache "
        "answers faster. `cached tokens` is the mean per request; it should be similar at every "
        "level, or the levels did not see the same cache state. "
        "USD per 1k alerts = GPU price per "
        "hour / (requests per second x 3600) x 1000, and reads `not set` until "
        "`prices.gpu_usd_per_hour` is filled in. It assumes the GPU is busy all the time at that "
        "level and counts one question as one alert, so it is a lower bound on real cost.",
        ("C", "n", "repeats", "errors", "req/s", "out tok/s", "p50 total ms", "p95 total ms",
         "p50 TTFT ms", "p95 TTFT ms", "cached tokens", "USD per 1k alerts"),
        concurrency_rows,
    ),
    "image_size": Section(
        "Image size",
        "The same frames, downscaled so the longer side is at most `max side` (never upscaled). "
        "One question per frame.",
        ("max side", "sent px", "n", "errors", "prompt tokens", "mean TTFT ms",
         "mean total ms", "schema valid", "accuracy"),
        image_rows,
    ),
}  # fmt: skip


def section_table(test: str, variants: Sequence[Variant]) -> str:
    section = SECTIONS[test]
    rows: list[Row] = []
    for v in variants:
        if test in v.tests:
            rows += [[v.label, *row] for row in section.rows(v.tests[test]["summary"])]
        else:
            why = "failed to start" if v.server_error is not None else NOT_RUN
            rows.append([v.label, why, *[""] * (len(section.headers) - 1)])
    return format_table(["variant", *section.headers], rows, markdown=True)


def cloud_table(variants: Sequence[Variant]) -> str | None:
    headers = ["variant", "cloud model", "prices as of", "image tokens", "prompt text tokens (est.)",
               "output tokens", "input USD/1M", "output USD/1M", "USD per 1k alerts"]  # fmt: skip
    rows = []
    for v in variants:
        cloud = (
            (v.tests.get("concurrency") or {})
            .get("summary", {})
            .get("cost", {})
            .get("cloud_reference")
        )
        for name, m in (cloud or {}).get("models", {}).items():
            rows.append(
                [
                    v.label,
                    name,
                    cloud["as_of"],
                    cloud["image_tokens"],
                    cloud["prompt_text_tokens_estimated"],
                    num(cloud["output_tokens"]),
                    m["input_usd_per_1m"],
                    m["output_usd_per_1m"],
                    num(m["usd_per_1k_alerts"], 4),
                ]
            )
    return format_table(headers, rows, markdown=True) if rows else None


def notes(variants: Sequence[Variant], problems: Sequence[str]) -> list[str]:
    out = []
    mocks = [v.name for v in variants if v.mock]
    if mocks:
        out.append(
            f"**{MOCK_LABEL}:** {', '.join(mocks)} ran against the built-in mock server. "
            "Those rows only show that the pipeline works; they say nothing about SGLang or any "
            "model."
        )
    text_only = [v.name for v in variants if v.text_only]
    if text_only:
        out.append(
            f"**{TEXT_ONLY_LABEL}:** {', '.join(text_only)} received a written scene description "
            "in place of each frame. Timings and JSON validity are real for that server; the "
            "accuracy figures only show that the model can read the description, not that it can "
            "read a picture."
        )

    def names(test_filter: Callable[[dict[str, Any]], bool]) -> list[str]:
        return [v.name for v in variants if any(test_filter(t["meta"]) for t in v.tests.values())]

    weak = names(lambda meta: meta.get("labels", {}).get("weak", 0) > 0)
    if weak:
        out.append(
            f"**Weak labels:** for {', '.join(weak)}, some or all frames had no per-frame "
            "`in_event` annotation, so the ground truth is the class of the whole video. Not "
            "every frame of a shoplifting or fighting video shows the act, so accuracy, precision "
            "and recall are rough."
        )
    elif any(v.tests for v in variants):
        out.append("Ground truth for accuracy is the per-frame `in_event` annotation.")
    approx = names(lambda meta: meta.get("usage_source", {}).get("approx", 0) > 0)
    if approx:
        out.append(
            f"**Approximate tokens:** for {', '.join(approx)}, the server did not report usage "
            "on some requests, so output tokens are counted streamed chunks and prompt tokens "
            "are missing."
        )
    unflushed = names(lambda meta: meta.get("cache_flush") == "unsupported")
    if unflushed:
        out.append(
            f"**Cache not flushed:** for {', '.join(unflushed)}, `/flush_cache` did not succeed, "
            "so timings may include prefixes cached by earlier requests."
        )
    out += [f"**Unreadable file:** {p}" for p in problems]
    return out


def render(results_dir: str | Path) -> str:
    results_dir = Path(results_dir)
    problems: list[str] = []
    variants = load_variants(results_dir, problems) if results_dir.is_dir() else []
    parts = ["# Camera-alert benchmark report", f"Generated {utc_now()} from `{results_dir}/`."]
    parts += notes(variants, problems)
    if not variants:
        parts.append("No results found.")
        return "\n\n".join(parts) + "\n"

    intro = (
        "`load s` is the time from starting the server process until it answers `/health`: "
        "Python start-up, weight loading and CUDA graph capture together, so it is not the "
        "weight loading time alone. `GPU idle MB` is the first `nvidia-smi` sample after the "
        "server is ready and includes the KV pool; `GPU peak MB` is the highest sample during "
        "the tests and includes memory that PyTorch keeps reserved."
    )
    parts += ["## Variants", intro, variants_table(variants)]
    for test, section in SECTIONS.items():
        parts += [f"## {section.title} (`{test}`)", section.intro, section_table(test, variants)]
    cloud = cloud_table(variants)
    if cloud:
        intro = (
            "What one alert of the same shape would cost on a per-token cloud API. Image tokens "
            "follow the provider's published rule; prompt text tokens are estimated from "
            "character count; output tokens are the mean measured in the concurrency run. "
            "Prices are list prices on the date shown, with no caching or batch discounts."
        )
        parts += ["## Cloud API reference price", intro, cloud]
    return "\n\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write REPORT.md from the result files.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    out = Path(args.results_dir) / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.results_dir), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
