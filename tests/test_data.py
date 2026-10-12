"""Tests for the data part. No network and no downloaded data needed."""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from camalerts.data import Frame, load_manifest, parse_temporal_annotations

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while the class is built
    spec.loader.exec_module(module)
    return module


dl = load_script("download_data")
ex = load_script("extract_frames")

FPS = 10
N_FRAMES = 300
WIDTH, HEIGHT = 64, 48


def gray_level(index):
    return 20 + index * 200 // N_FRAMES


@pytest.fixture(scope="module")
def video_dir(tmp_path_factory):
    """data/videos/<label>/<name>.mp4 with one 30 s clip per label; frame i is a flat gray."""
    root = tmp_path_factory.mktemp("repo")
    for label, name in (
        ("shoplifting", "Shoplifting001_x264"),
        ("normal", "Normal_Videos_001_x264"),
    ):
        path = root / "data" / "videos" / label / f"{name}.mp4"
        path.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        assert writer.isOpened()
        for i in range(N_FRAMES):
            writer.write(np.full((HEIGHT, WIDTH, 3), gray_level(i), np.uint8))
        writer.release()
    return root


CFG = {
    "source": {"name": "ucf-crime"},
    "labels": {"shoplifting": {"anomaly": True}, "normal": {"anomaly": False}},
    "frames": {"every_sec": 5.0, "skip_sec": 2.0, "max_per_video": 12, "jpeg_quality": 90},
}


# --- frame sampling -----------------------------------------------------------------------------


def test_plan_regular_grid():
    assert ex.plan_frame_indices(N_FRAMES, FPS, ex.Sampling()) == [20, 70, 120, 170, 220, 270]


def test_plan_caps_frames_and_covers_whole_video():
    indices = ex.plan_frame_indices(30 * 600, 30, ex.Sampling())  # 10 minutes at 30 fps
    assert len(indices) == 12
    assert indices == sorted(set(indices))
    assert indices[0] < 30 * 60 and indices[-1] > 30 * 540


def test_plan_takes_about_half_from_event_ranges():
    ranges = [(100, 160)]
    indices = ex.plan_frame_indices(N_FRAMES, FPS, ex.Sampling(), ranges)
    inside = [i for i in indices if ex.in_any(i, ranges)]
    assert len(inside) == 6
    assert [i for i in indices if i not in inside] == [20, 70, 170, 220, 270]
    assert min(b - a for a, b in itertools.pairwise(inside)) >= FPS  # at least 1 s apart


def test_plan_short_event_still_gives_a_positive():
    indices = ex.plan_frame_indices(N_FRAMES, FPS, ex.Sampling(), [(200, 204)])
    assert sum(ex.in_any(i, [(200, 204)]) for i in indices) == 1


def test_plan_clamps_ranges_past_the_end():
    indices = ex.plan_frame_indices(N_FRAMES, FPS, ex.Sampling(), [(250, 5000)])
    assert max(indices) < N_FRAMES


def test_event_ranges_meaning():
    annotations = {"Shoplifting001_x264": [(100, 160)]}
    assert ex.event_ranges("Shoplifting001_x264", True, annotations) == [(100, 160)]
    assert ex.event_ranges("Shoplifting999_x264", True, annotations) is None
    assert ex.event_ranges("Normal_Videos_001_x264", False, annotations) == []


def test_extract_all_writes_frames_and_manifest(video_dir):
    data = video_dir / "data"
    manifest = data / "manifest.jsonl"
    annotations = {"Shoplifting001_x264": [(100, 160)]}
    records = ex.extract_all(
        data / "videos", data / "frames", manifest, video_dir, CFG, annotations
    )

    lines = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert lines == records
    fields = {"frame", "video", "label", "t_sec", "width", "height", "source", "in_event"}
    assert all(set(line) == fields for line in lines)

    normal = [r for r in lines if r["label"] == "normal"]
    assert [r["frame"] for r in normal] == [
        f"data/frames/normal/Normal_Videos_001_x264/{i:06d}.jpg"
        for i in (20, 70, 120, 170, 220, 270)
    ]
    assert [r["t_sec"] for r in normal] == [2.0, 7.0, 12.0, 17.0, 22.0, 27.0]
    assert all(r["in_event"] is False and r["source"] == "ucf-crime" for r in normal)

    shop = [r for r in lines if r["label"] == "shoplifting"]
    assert sum(r["in_event"] for r in shop) == 6
    assert sum(not r["in_event"] for r in shop) == 5

    for r in lines:
        image = cv2.imread(str(video_dir / r["frame"]))
        assert image.shape == (HEIGHT, WIDTH, 3) == (r["height"], r["width"], 3)
        # The gray level encodes the frame index, so this fails if seeking lands elsewhere.
        index = int(Path(r["frame"]).stem)
        assert abs(float(image.mean()) - gray_level(index)) < 8

    assert len(load_manifest(manifest)) == len(lines)


def test_extract_removes_stale_frames_and_handles_missing_annotations(video_dir):
    data = video_dir / "data"
    stale = data / "frames" / "shoplifting" / "Shoplifting001_x264" / "999999.jpg"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")
    records = ex.extract_all(
        data / "videos", data / "frames", data / "manifest2.jsonl", video_dir, CFG, {}
    )
    assert not stale.exists()
    assert {r["in_event"] for r in records if r["label"] == "shoplifting"} == {None}


def test_extract_video_rejects_non_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="cannot read video"):
        ex.extract_video(bad, "normal", tmp_path / "out", tmp_path, ex.Sampling(), [], "x")


# --- manifest reader ----------------------------------------------------------------------------


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for label, count in (("shoplifting", 10), ("fighting", 10), ("normal", 3)):
        for i in range(count):
            row = {
                "frame": f"data/frames/{label}/v{i % 2}/{i:06d}.jpg",
                "video": f"v{i % 2}",
                "label": label,
                "t_sec": float(i),
                "width": 320,
                "height": 240,
                "source": "ucf-crime",
            }
            if label != "fighting":
                row["in_event"] = label == "shoplifting" and i < 5
            rows.append(row)
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_load_manifest_all(manifest):
    frames = load_manifest(manifest)
    assert len(frames) == 23
    assert isinstance(frames[0], Frame)
    assert frames[0].frame == "data/frames/shoplifting/v0/000000.jpg"
    assert (frames[0].width, frames[0].height, frames[0].in_event) == (320, 240, True)
    assert all(f.in_event is None for f in frames if f.label == "fighting")


def test_load_manifest_filters_labels(manifest):
    frames = load_manifest(manifest, labels=["normal", "fighting"])
    assert {f.label for f in frames} == {"normal", "fighting"}
    assert len(frames) == 13
    with pytest.raises(ValueError, match="robbery"):
        load_manifest(manifest, labels=["robbery"])


def test_load_manifest_limit_is_even_across_labels(manifest):
    frames = load_manifest(manifest, limit=6)
    assert (
        sorted(f.label for f in frames) == ["fighting"] * 2 + ["normal"] * 2 + ["shoplifting"] * 2
    )
    # normal has only 3 frames; the other labels share the rest
    counts = {}
    for f in load_manifest(manifest, limit=15):
        counts[f.label] = counts.get(f.label, 0) + 1
    assert counts == {"fighting": 6, "normal": 3, "shoplifting": 6}
    assert len(load_manifest(manifest, limit=1000)) == 23
    assert load_manifest(manifest, limit=0) == []


def test_load_manifest_limit_is_deterministic(manifest):
    assert load_manifest(manifest, limit=9, seed=1) == load_manifest(manifest, limit=9, seed=1)
    assert load_manifest(manifest, limit=9, seed=1) != load_manifest(manifest, limit=9, seed=2)


def test_load_manifest_missing_file_says_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_frames.py"):
        load_manifest(tmp_path / "nope.jsonl")


def test_load_manifest_bad_line_reports_line_number(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"frame": "a.jpg"}\n')
    with pytest.raises(ValueError, match="m.jsonl:1"):
        load_manifest(path)


def test_parse_temporal_annotations():
    text = (
        "Shoplifting007_x264.mp4  Shoplifting  550  760  4630  4920  \n"
        "Fighting018_x264.mp4  Fighting  80  420  -1  -1  \n"
        "Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1  \n"
        "\n"
    )
    assert parse_temporal_annotations(text) == {
        "Shoplifting007_x264": [(550, 760), (4630, 4920)],
        "Fighting018_x264": [(80, 420)],
        "Normal_Videos_003_x264": [],
    }


# --- download planning and the size cap ---------------------------------------------------------

MB = 10**6
LISTINGS = {
    "a.zip": [
        dl.Member("A/Shoplifting/Shoplifting003_x264.mp4", 900 * MB),
        dl.Member("A/Shoplifting/Shoplifting001_x264.mp4", 400 * MB),
        dl.Member("A/Shoplifting/Shoplifting002_x264.mp4", 300 * MB),
        dl.Member("A/Stealing/Stealing001_x264.mp4", 5000 * MB),
    ],
    "n.zip": [
        dl.Member("N/Normal_Videos_002_x264.mp4", 200 * MB),
        dl.Member("N/Normal_Videos_001_x264.mp4", 100 * MB),
    ],
}
DL_CFG = {
    "videos_per_label": 2,
    "labels": {
        "shoplifting": {"archive": "a.zip", "pattern": "A/Shoplifting/*.mp4"},
        "normal": {"archive": "n.zip", "pattern": "N/*.mp4"},
    },
}


def test_select_prefers_annotated_then_name():
    chosen = dl.select_videos(LISTINGS["a.zip"], "A/Shoplifting/*.mp4", 2, {"Shoplifting003_x264"})
    assert [Path(m.name).stem for m in chosen] == ["Shoplifting003_x264", "Shoplifting001_x264"]


def test_build_plan_is_deterministic_and_sized_from_zip_metadata():
    plan = dl.build_plan(DL_CFG, LISTINGS, annotated=set())
    assert [(i.label, i.video) for i in plan] == [
        ("shoplifting", "Shoplifting001_x264"),
        ("shoplifting", "Shoplifting002_x264"),
        ("normal", "Normal_Videos_001_x264"),
        ("normal", "Normal_Videos_002_x264"),
    ]
    assert plan[0].rel_path == "videos/shoplifting/Shoplifting001_x264.mp4"
    assert dl.plan_bytes(plan) == 1000 * MB
    assert plan == dl.build_plan(DL_CFG, LISTINGS, annotated=set())


def test_build_plan_fails_when_pattern_matches_nothing():
    cfg = {"labels": {"fighting": {"archive": "a.zip", "pattern": "A/Fighting/*.mp4"}}}
    with pytest.raises(ValueError, match="Fighting"):
        dl.build_plan(cfg, LISTINGS, set())


def test_cap_refuses_plan_over_limit():
    plan = dl.build_plan(DL_CFG, LISTINGS, set())
    assert dl.check_cap(plan, cap=1000 * MB) == 1000 * MB
    with pytest.raises(dl.CapError, match="over the cap"):
        dl.check_cap(plan, cap=999 * MB)
    with pytest.raises(dl.CapError):  # files already under data/ count too
        dl.check_cap(plan, cap=1000 * MB, other_bytes=1)


def test_cap_setting_can_never_exceed_5_gb():
    assert dl.cap_bytes(3.0) == 3 * 10**9
    assert dl.cap_bytes(5.0) == 5 * 10**9
    for bad in (5.01, 50, 0, -1):
        with pytest.raises(dl.CapError):
            dl.cap_bytes(bad)


def test_fit_trims_plan_and_keeps_labels_balanced():
    plan = dl.build_plan(DL_CFG, LISTINGS, set())
    fitted = dl.fit_plan(plan, budget=650 * MB)
    assert [i.video for i in fitted] == ["Shoplifting001_x264", "Normal_Videos_001_x264"]
    assert dl.plan_bytes(fitted) <= 650 * MB
    assert dl.fit_plan(plan, budget=10**12) == plan
    assert dl.fit_plan(plan, budget=0) == []


def test_resume_skips_only_files_with_the_right_size(tmp_path):
    item = dl.PlanItem("normal", "n.zip", "N/Normal_Videos_001_x264.mp4", 10, True)
    dest = tmp_path / item.rel_path
    assert not dl.is_present(tmp_path, item)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * 9)
    assert not dl.is_present(tmp_path, item)
    dest.write_bytes(b"x" * 10)
    assert dl.is_present(tmp_path, item)


def test_other_bytes_counts_everything_except_planned_videos(tmp_path):
    item = dl.PlanItem("normal", "n.zip", "N/Normal_Videos_001_x264.mp4", 10, True)
    (tmp_path / "videos" / "normal").mkdir(parents=True)
    (tmp_path / item.rel_path).write_bytes(b"x" * 10)
    (tmp_path / "videos" / "normal" / "old.mp4").write_bytes(b"x" * 7)
    (tmp_path / "manifest.jsonl").write_bytes(b"x" * 5)
    assert dl.other_bytes_on_disk(tmp_path, [item]) == 12
    assert dl.other_bytes_on_disk(tmp_path, []) == 22
    assert dl.other_bytes_on_disk(tmp_path / "missing", [item]) == 0
