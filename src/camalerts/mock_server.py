"""A fake OpenAI-compatible server for local tests. Its numbers are toys, never measurements.

    python -m camalerts.mock_server --port 30999 [--seed 0]

It imitates the parts of the API the benchmarks use: /v1/models, /health, /flush_cache and
/v1/chat/completions (streaming and non-streaming). Answers are a deterministic function of the
request and the seed. Timing is simulated: time to first token grows with the uncached part of
the prompt, each output token costs a fixed delay, and a semaphore caps parallel generations.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from PIL import Image

from camalerts.client import MOCK_OWNER

MODEL_ID = "mock-vlm"
CACHE_BLOCK_CHARS = 500
CACHE_MIN_CHARS = 2000
CHARS_PER_TOKEN = 4
IMAGE_PATCH_PX = 28


@dataclass
class MockConfig:
    seed: int = 0
    speed: float = 1.0  # multiplies every simulated delay; tests use a small value
    max_parallel: int = 4
    ttft_base_s: float = 0.020
    s_per_prompt_token: float = 0.00005
    s_per_output_token: float = 0.002
    usage_mode: str = "separate"  # separate | inline | none: where the stream reports usage
    # Without response_format, the share of answers that come back in each broken shape.
    p_fenced: float = 0.25
    p_comment: float = 0.10
    p_prose: float = 0.10
    p_wrong_type: float = 0.10


class PrefixCache:
    """Toy prefix cache: remembers every 500-character prefix of the prompts it has seen."""

    def __init__(self) -> None:
        self._seen: set[bytes] = set()

    def cached_chars(self, text: str) -> int:
        """Length of the longest prefix seen before (0 if under 2,000 characters), then store."""
        digest = hashlib.sha1()
        hit = 0
        still_matching = True
        for end in range(CACHE_BLOCK_CHARS, len(text) + 1, CACHE_BLOCK_CHARS):
            digest.update(text[end - CACHE_BLOCK_CHARS : end].encode("utf-8"))
            key = digest.digest()
            if still_matching and key in self._seen:
                hit = end
            else:
                still_matching = False
                self._seen.add(key)
        return hit if hit >= CACHE_MIN_CHARS else 0

    def clear(self) -> None:
        self._seen.clear()


def flatten(messages: list[dict[str, Any]]) -> tuple[str, int]:
    """Prompt as one string (images replaced by a hash marker) and a toy image token count."""
    parts: list[str] = []
    image_tokens = 0
    for message in messages:
        parts.append(f"<{message.get('role', 'user')}>\n")
        content = message.get("content") or ""
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content:
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                parts.append(f"<image:{hashlib.sha1(url.encode('ascii')).hexdigest()[:16]}>")
                image_tokens += _image_tokens(url)
    return "".join(parts), image_tokens


def _image_tokens(data_url: str) -> int:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
    except (IndexError, ValueError, OSError):
        return 0
    return math.ceil(width / IMAGE_PATCH_PX) * math.ceil(height / IMAGE_PATCH_PX)


def make_answer(prompt: str, constrained: bool, cfg: MockConfig) -> str:
    """Deterministic answer for a prompt. Only knows the camera-alert answer shape."""
    digest = hashlib.sha256(f"{cfg.seed}\n{prompt}".encode()).digest()
    event = bool(digest[0] & 1)
    answer = {
        "event": event,
        "confidence": ("low", "medium", "high")[digest[1] % 3],
        "people": digest[2] % 4 + (1 if event else 0),
        "reason": (
            "Person at the left shelf moves an item out of view."
            if event
            else "Nothing matching the question is visible in this frame."
        ),
    }
    text = json.dumps(answer)
    if constrained:
        return text

    draw = int.from_bytes(digest[4:8], "big") / 2**32
    if draw < cfg.p_fenced:
        return f"```json\n{text}\n```"
    draw -= cfg.p_fenced
    if draw < cfg.p_comment:
        return f"{text}\n// confidence reflects the partly blocked view"
    draw -= cfg.p_comment
    if draw < cfg.p_prose:
        return "The frame shows a shop aisle. " + answer["reason"]
    draw -= cfg.p_prose
    if draw < cfg.p_wrong_type:
        return json.dumps({**answer, "people": str(answer["people"])})
    return text


def _tokens(text: str) -> list[str]:
    return [text[i : i + CHARS_PER_TOKEN] for i in range(0, len(text), CHARS_PER_TOKEN)]


def create_app(cfg: MockConfig | None = None) -> web.Application:
    cfg = cfg or MockConfig()
    cache = PrefixCache()
    gate = asyncio.Semaphore(cfg.max_parallel)

    async def models(_: web.Request) -> web.Response:
        model = {"id": MODEL_ID, "object": "model", "owned_by": MOCK_OWNER}
        return web.json_response({"object": "list", "data": [model]})

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "mock": True})

    async def flush(_: web.Request) -> web.Response:
        cache.clear()
        return web.Response(text="Cache flushed.\n")

    async def chat(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            messages = body["messages"]
            prompt, image_tokens = flatten(messages)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            error = {"message": f"bad request: {exc!r}", "type": "invalid_request_error"}
            return web.json_response({"error": error}, status=400)

        fmt = (body.get("response_format") or {}).get("type")
        text = make_answer(prompt, fmt in ("json_schema", "json_object"), cfg)
        tokens = _tokens(text)
        max_tokens = body.get("max_tokens")
        finish = "stop"
        if max_tokens is not None and len(tokens) > max_tokens:
            tokens, finish = tokens[:max_tokens], "length"

        prompt_tokens = math.ceil(len(prompt) / CHARS_PER_TOKEN) + image_tokens
        cached_tokens = cache.cached_chars(prompt) // CHARS_PER_TOKEN
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "total_tokens": prompt_tokens + len(tokens),
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        }
        ttft = cfg.speed * (
            cfg.ttft_base_s + cfg.s_per_prompt_token * (prompt_tokens - cached_tokens)
        )
        token_delay = cfg.speed * cfg.s_per_output_token
        base = {
            "id": f"chatcmpl-mock-{time.time_ns()}",
            "created": int(time.time()),
            "model": MODEL_ID,
        }

        if not body.get("stream"):
            async with gate:
                await asyncio.sleep(ttft + token_delay * len(tokens))
            message = {"role": "assistant", "content": "".join(tokens)}
            choice = {"index": 0, "message": message, "finish_reason": finish}
            return web.json_response(
                {**base, "object": "chat.completion", "choices": [choice], "usage": usage}
            )

        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        usage_mode = cfg.usage_mode if want_usage else "none"
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        async def emit(
            delta: dict[str, Any], finish_reason: str | None = None, **extra: Any
        ) -> None:
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            chunk = {**base, "object": "chat.completion.chunk", "choices": [choice], **extra}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

        try:
            async with gate:
                await emit({"role": "assistant", "content": ""})
                await asyncio.sleep(ttft)
                for token in tokens:
                    await emit({"content": token})
                    await asyncio.sleep(token_delay)
            if usage_mode == "inline":
                await emit({}, finish, usage=usage)
            else:
                await emit({}, finish)
            if usage_mode == "separate":
                tail = {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage}
                await resp.write(f"data: {json.dumps(tail)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        except ConnectionResetError:
            pass  # the client went away mid-stream
        return resp

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_route("*", "/flush_cache", flush)
    app.router.add_post("/v1/chat/completions", chat)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible server for local tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="multiplies simulated delays")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--usage-mode", choices=["separate", "inline", "none"], default="separate")
    args = parser.parse_args(argv)
    cfg = MockConfig(
        seed=args.seed, speed=args.speed, max_parallel=args.max_parallel, usage_mode=args.usage_mode
    )
    print(f"MOCK server ({MODEL_ID}) on http://{args.host}:{args.port} - not a real model")
    web.run_app(create_app(cfg), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
