# Design

This file is the contract between the parts of the project. If code and this file disagree, fix one of them.

## Goal

Measure how well SGLang serves a small vision-language model (VLM) for security-camera alerts, and how much
each optimisation saves. The workload is: one camera frame + one question in, one short JSON answer out.

Five measurements:

1. **JSON validity** — share of answers that parse and match the schema: with constrained decoding,
   without it on the same prompt, and with a plain request in words.
2. **Prefix caching** — time to first token when the shared instructions come first vs last in the prompt.
3. **Concurrency** — throughput and latency as 1, 2, 4, 8, 16, 32 simulated cameras call at once,
   one question per frame so every level sees the same cache state.
4. **Image size** — latency, prompt tokens and accuracy as the frame is scaled down.
5. **Quantization** — the same suite repeated per model variant (16-bit, 4-bit AWQ, others), plus GPU memory.

A cost model turns throughput into dollars per 1,000 alerts and compares it with a per-call cloud API price.

## Where things run

- **Laptop (Apple M1, no NVIDIA GPU):** data download, frame extraction, unit tests, and an end-to-end run
  against a local mock server. SGLang cannot run here.
- **RunPod (NVIDIA GPU):** SGLang server and the real benchmark runs.

## Layout and ownership

```
configs/data.yaml        which videos to fetch, size cap                    (data)
configs/models.yaml      model variants and their SGLang arguments          (serving)
configs/alerts.yaml      alert tasks: question, JSON schema, label mapping  (bench)
configs/bench.yaml       request counts, concurrency levels, prices         (bench)
scripts/download_data.py selective download from the remote zip files       (data)
scripts/extract_frames.py videos -> JPEG frames + data/manifest.jsonl       (data)
scripts/runpod_setup.sh  install SGLang and this package on the pod         (serving)
scripts/serve.py         launch SGLang for one variant from models.yaml     (serving)
scripts/gpu_mem.py       sample nvidia-smi while a benchmark runs           (serving)
scripts/run_all.sh       run every test against the running server          (bench)
scripts/run_matrix.sh    per variant: start server, sample GPU, run_all, stop (integration)
src/camalerts/data.py    manifest reader                                    (data)
src/camalerts/*.py       client, prompts, schema, benchmarks, cost, report  (bench)
src/camalerts/mock_server.py  OpenAI-compatible fake for local tests        (bench)
tests/                   pytest, runs on the laptop with no GPU             (each owner)
```

## Contracts

### Manifest: `data/manifest.jsonl`

One JSON object per line, one line per extracted frame:

```json
{"frame": "data/frames/shoplifting/Shoplifting001_x264/001587.jpg",
 "video": "Shoplifting001_x264", "label": "shoplifting", "t_sec": 52.9,
 "width": 320, "height": 240, "source": "ucf-crime", "in_event": true}
```

- `frame` is a path relative to the repository root. The file name is the frame's index in the video
  (zero-padded to six digits), so `t_sec` = index / fps. The videos fetched so far are 30 fps, 320x240.
- `label` is lower-case: `shoplifting`, `fighting`, `normal` (more may be added).
- The label is the video's class. It is a weak label: not every frame of a shoplifting video shows the act.
- `in_event` (optional, added by the data part) is the stronger per-frame label. It comes from UCF-Crime's
  `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`, which gives frame ranges of the anomaly for the
  290 test-set videos only:
  - `true`: the frame index is inside an annotated range of its video.
  - `false`: the video is annotated and the frame is outside every range, or the video is `normal`.
  - `null`: an anomaly video with no annotation (only 5 of the 50 Fighting videos have one;
    21 of the 50 Shoplifting videos do). Fall back to `label` for these, or drop them.

  The ranges are hand-marked and coarse. A `false` frame right next to a range can still show the scene
  just before or after the act.

Sampling (`configs/data.yaml`, `frames:`): one frame every 5 s starting at 2 s, at most 12 per video; a
longer video gets its 12 spread over its whole length. For an annotated anomaly video about half of the 12
are taken from inside the ranges (at least 1 s apart, so a short event gives fewer) and the rest from the
5 s grid outside them. The download script prefers annotated videos.

`camalerts.data.load_manifest(path="data/manifest.jsonl", labels=None, limit=None, seed=0) -> list[Frame]`
returns dataclass objects with the fields above (`in_event` is `None` when absent). `limit` samples evenly
across labels and is deterministic for a given `seed`; the result keeps manifest order.

### Model variants: `configs/models.yaml`

```yaml
variants:
  - id: bf16                # short name, used as results/<id>/
    hf_model: org/name      # Hugging Face repo id
    quantization: none      # none | awq-w4a16 | fp8 | ...
    status: expected        # expected | experimental
    sglang_args: ["--dtype", "bfloat16"]   # extra args for sglang.launch_server
    notes: "why this variant exists, with a source"
```

`scripts/serve.py --variant <id> [--port 30000]` starts the server and prints the served model name.

### Server API

OpenAI-compatible chat completions at `http://HOST:PORT/v1/chat/completions`. Images are sent as
base64 `data:image/jpeg;base64,...` URLs inside an `image_url` content part. Constrained output uses
`response_format: {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}`.
Streaming (`stream: true`) is used to time the first token.

### Results: `results/<variant_id>/<test>.json`

Each benchmark writes one JSON file with a `meta` block (variant, model, server URL, timestamp, config
used) and a `summary` block (the numbers that go in the report). Per-request records go to
`results/<variant_id>/raw/<test>.jsonl`, which is git-ignored.

Test names: `json_validity`, `prefix_cache`, `concurrency`, `image_size`, `gpu_mem`.

`meta.mock` is true for the built-in mock server. `meta.text_only` is true when a written scene
stood in for each frame (`--text-only`, for servers that cannot read images); the image-size test is
skipped in that mode. The report labels both kinds of rows.

`scripts/serve.py` also writes `results/<variant_id>/server.json` (command, load time,
`weights_mem_gb`, `max_total_num_tokens`) or, when the server does not start,
`server_error.json` (`error`, `log_tail`).

`python -m camalerts.report` reads every `results/*/` folder and writes `results/REPORT.md`.

## Rules

- Python 3.12 through `uv`. Run things with `uv run ...`. Do not edit `pyproject.toml` without saying so.
- Nothing under `data/` is committed. The dataset is for research use; ship the download script, not the files.
- No number goes into a document unless a run produced it. Mock-server numbers are labelled as mock.
- A variant that fails to start is recorded as a failure with its error, not silently skipped.
