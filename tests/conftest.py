"""Shared fixtures: synthetic frames, the real configs shrunk for speed, and an in-process mock."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from PIL import Image, ImageDraw

from camalerts import common
from camalerts.mock_server import MockConfig, create_app

REPO = Path(__file__).resolve().parents[1]
LABELS = ("shoplifting", "fighting", "normal")


def make_frames(
    directory: Path, n: int, size: tuple[int, int] = (320, 240), in_event: bool | None = None
) -> list[SimpleNamespace]:
    """Write n distinct small JPEGs and return manifest-like objects pointing at them."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        img = Image.new("RGB", size, (i * 37 % 256, i * 91 % 256, 80))
        ImageDraw.Draw(img).rectangle((10 + i, 20, 60 + 2 * i, 90 + i), fill=(250, 250, 250))
        path = directory / f"frame_{i:03d}.jpg"
        img.save(path, "JPEG", quality=85)
        frames.append(
            SimpleNamespace(frame=str(path), label=LABELS[i % len(LABELS)], in_event=in_event)
        )
    return frames


@pytest.fixture
def frames(tmp_path: Path) -> list[SimpleNamespace]:
    return make_frames(tmp_path / "frames", 9)


@pytest.fixture(scope="session")
def alerts() -> common.Alerts:
    return common.load_alerts(REPO / "configs" / "alerts.yaml")


@pytest.fixture
def config() -> dict[str, Any]:
    """configs/bench.yaml with the slow knobs turned down."""
    cfg = common.load_yaml(REPO / "configs" / "bench.yaml")
    cfg["concurrency"].update(levels=[1, 4], requests_per_worker=2, min_requests=8)
    cfg["image_size"]["max_sides"] = [None, 128]
    return cfg


async def serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


@pytest.fixture
async def start_app() -> AsyncIterator[Callable[[web.Application], Awaitable[str]]]:
    """Serve any aiohttp app on a free localhost port; returns its base URL."""
    runners: list[web.AppRunner] = []

    async def start(app: web.Application) -> str:
        runner, url = await serve(app)
        runners.append(runner)
        return url

    yield start
    for runner in runners:
        await runner.cleanup()


@pytest.fixture
async def start_mock(start_app) -> Callable[..., Awaitable[str]]:
    """Start the mock server with MockConfig overrides; delays are scaled down for speed."""

    async def start(**overrides: Any) -> str:
        return await start_app(create_app(MockConfig(**{"speed": 0.2, **overrides})))

    return start


@pytest.fixture
async def mock_url(start_mock) -> str:
    return await start_mock()
