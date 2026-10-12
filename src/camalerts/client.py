"""Async client for an OpenAI-compatible chat completions server. Errors are recorded, not raised."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

MOCK_OWNER = "camalerts-mock"


@dataclass
class ChatResult:
    ok: bool = False
    error: str | None = None
    text: str = ""
    ttft_s: float | None = None  # time to the first non-empty content chunk
    total_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    usage_source: str | None = None  # "server" | "approx" (approx = streamed chunks counted)
    http_status: int | None = None
    finish_reason: str | None = None


@dataclass
class ServerInfo:
    ok: bool
    model_id: str | None = None
    mock: bool = False
    error: str | None = None


def make_client(timeout_s: float = 120.0, max_connections: int = 64) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=max_connections, max_keepalive_connections=max_connections
    )
    # trust_env=False: proxy variables on a rented pod must not reroute or slow localhost calls.
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s, connect=10.0), limits=limits, trust_env=False
    )


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


async def chat(
    http: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> ChatResult:
    """Send one streaming chat completion and time it. One attempt, no retries."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if response_format is not None:
        body["response_format"] = response_format
    if extra_body:
        body.update(extra_body)

    result = ChatResult()
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    start = time.perf_counter()
    try:
        async with http.stream("POST", _url(base_url, "/v1/chat/completions"), json=body) as resp:
            result.http_status = resp.status_code
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", "replace")
                result.error = f"HTTP {resp.status_code}: {_one_line(detail)}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        result.error = f"stream error: {_one_line(json.dumps(chunk['error']))}"
                        break
                    # Usage may ride on the last content chunk or on a later chunk with no choices.
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            if result.ttft_s is None:
                                result.ttft_s = time.perf_counter() - start
                            pieces.append(piece)
                        result.finish_reason = choice.get("finish_reason") or result.finish_reason
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        result.error = f"{type(exc).__name__}: {_one_line(str(exc))}"
    result.total_s = time.perf_counter() - start

    result.text = "".join(pieces)
    if result.error is None and not pieces:
        result.error = "no content received"
    result.ok = result.error is None

    if usage and usage.get("completion_tokens") is not None:
        result.usage_source = "server"
        result.prompt_tokens = usage.get("prompt_tokens")
        result.completion_tokens = usage["completion_tokens"]
        result.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    elif pieces:
        result.usage_source = "approx"
        result.completion_tokens = len(pieces)
    return result


async def get_server_info(base_url: str, timeout_s: float = 10.0) -> ServerInfo:
    """Read the served model id from /v1/models and note whether the server says it is the mock."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as http:
            resp = await http.get(_url(base_url, "/v1/models"))
            resp.raise_for_status()
            models = resp.json().get("data") or []
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        return ServerInfo(ok=False, error=f"{type(exc).__name__}: {_one_line(str(exc))}")
    if not models:
        return ServerInfo(ok=False, error="/v1/models returned no models")
    first = models[0]
    return ServerInfo(ok=True, model_id=first.get("id"), mock=first.get("owned_by") == MOCK_OWNER)


async def flush_cache(base_url: str, timeout_s: float = 30.0) -> bool:
    """Ask the server to drop its prefix cache (SGLang: POST /flush_cache). False if unsupported."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as http:
            resp = await http.post(_url(base_url, "/flush_cache"))
            return resp.status_code == 200
    except (httpx.HTTPError, httpx.InvalidURL):
        return False


def _one_line(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]
