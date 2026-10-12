import httpx
import pytest
from aiohttp import web
from conftest import make_frames

from camalerts import client as api
from camalerts import prompts

DEAD_URL = "http://127.0.0.1:9"  # discard port: nothing listens there


@pytest.fixture
def messages(tmp_path, alerts):
    image = prompts.encode_frame(make_frames(tmp_path, 1)[0].frame)
    return prompts.build_messages(
        site_brief=alerts.site_brief,
        schema=alerts.schema,
        question=alerts.tasks[0].question,
        image_b64=image.b64,
    )


async def ask(url, messages, **kwargs) -> api.ChatResult:
    kwargs = {"max_tokens": 96, "temperature": 0, **kwargs}
    async with api.make_client(10) as http:
        return await api.chat(http, url, "mock-vlm", messages, **kwargs)


async def test_streaming_result_has_timings_and_server_usage(mock_url, messages, alerts):
    result = await ask(mock_url, messages, response_format=prompts.response_format(alerts.schema))
    assert result.ok and result.error is None and result.http_status == 200
    assert 0 < result.ttft_s < result.total_s
    assert result.text.startswith("{") and result.finish_reason == "stop"
    assert result.usage_source == "server"
    assert result.prompt_tokens > 1000 and result.completion_tokens > 0
    assert result.cached_tokens == 0  # first time the server sees this prompt


async def test_usage_on_the_last_content_chunk_is_accepted(start_mock, messages):
    result = await ask(await start_mock(usage_mode="inline"), messages)
    assert result.ok and result.usage_source == "server" and result.prompt_tokens > 1000


async def test_missing_usage_falls_back_to_counting_chunks(start_mock, messages):
    reference = await ask(await start_mock(), messages)
    result = await ask(await start_mock(usage_mode="none"), messages)
    assert result.ok and result.usage_source == "approx"
    assert result.completion_tokens == reference.completion_tokens  # the mock streams 1 token/chunk
    assert result.prompt_tokens is None and result.cached_tokens is None


async def test_truncation_is_reported(mock_url, messages):
    result = await ask(mock_url, messages, max_tokens=5)
    assert result.ok and result.finish_reason == "length" and result.completion_tokens == 5


async def test_extra_body_is_merged_into_the_request(start_app, messages):
    seen = {}

    async def chat(request: web.Request) -> web.Response:
        seen.update(await request.json())
        return web.json_response({"error": {"message": "stop here"}}, status=418)

    app = web.Application(client_max_size=2**24)
    app.router.add_post("/v1/chat/completions", chat)
    await ask(await start_app(app), messages, extra_body={"top_k": 1})
    assert seen["top_k"] == 1 and seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}
    assert "response_format" not in seen


async def test_http_error_is_recorded_not_raised(mock_url):
    result = await ask(mock_url, ["not a message"])
    assert not result.ok and result.http_status == 400
    assert result.error.startswith("HTTP 400") and result.total_s > 0
    assert result.usage_source is None


async def test_connection_error_is_recorded_not_raised(messages):
    result = await ask(DEAD_URL, messages)
    assert not result.ok and "ConnectError" in result.error
    assert result.http_status is None and result.ttft_s is None


async def test_error_inside_the_stream_is_recorded(start_app, messages):
    async def chat(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"error": {"message": "out of memory", "code": 500}}\n\n')
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application(client_max_size=2**24)
    app.router.add_post("/v1/chat/completions", chat)
    result = await ask(await start_app(app), messages)
    assert not result.ok and "out of memory" in result.error


async def test_server_info_flags_the_mock(mock_url):
    info = await api.get_server_info(mock_url)
    assert info.ok and info.model_id == "mock-vlm" and info.mock is True


async def test_server_info_for_a_real_server_is_not_mock(start_app):
    async def models(_: web.Request) -> web.Response:
        model = {"id": "org/some-vlm", "object": "model", "owned_by": "sglang"}
        return web.json_response({"object": "list", "data": [model]})

    app = web.Application()
    app.router.add_get("/v1/models", models)
    info = await api.get_server_info(await start_app(app))
    assert info.ok and info.model_id == "org/some-vlm" and info.mock is False


async def test_server_info_when_unreachable():
    info = await api.get_server_info(DEAD_URL, timeout_s=2)
    assert not info.ok and info.error and info.mock is False


async def test_flush_cache(mock_url, start_app):
    assert await api.flush_cache(mock_url) is True
    assert await api.flush_cache(await start_app(web.Application())) is False  # 404
    assert await api.flush_cache(DEAD_URL, timeout_s=2) is False


async def test_mock_non_streaming_and_health(mock_url, messages):
    body = {"model": "mock-vlm", "messages": messages, "max_tokens": 96}
    async with httpx.AsyncClient() as http:
        health = await http.get(f"{mock_url}/health")
        resp = await http.post(f"{mock_url}/v1/chat/completions", json=body)
    assert health.status_code == 200 and health.json()["mock"] is True
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"]
    assert data["usage"]["completion_tokens"] > 0
