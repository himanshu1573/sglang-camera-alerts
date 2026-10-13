# sglang-camera-alerts

A benchmark of [SGLang](https://docs.sglang.io/) serving a small vision-language model for
security-camera alerts. The workload is one camera frame and one question in, one short JSON answer
out:

```json
{"event": true, "confidence": "high", "people": 1, "reason": "Person at the right shelf pushes a boxed item into an open backpack."}
```

It measures what each serving optimisation is worth for this workload, and what each quantization
method costs in accuracy.

> **Status (2026-10-13):** all six variants ran once on a rented GPU (RTX PRO 4500 Blackwell,
> SGLang 0.5.21), 1,403 requests each with no failed request. The numbers are in
> [results/REPORT.md](results/REPORT.md), produced by the run described in [RUNPOD.md](RUNPOD.md).
> They are one run on one cloud GPU, and the model missed most real events, so they describe how
> SGLang serves this workload and not how well the model detects.

## What it measures

| Test | Question it answers | SGLang feature |
| --- | --- | --- |
| JSON validity | How often does the answer parse and match the schema, with constrained decoding and, for the same prompt, without it? | Grammar-constrained output (XGrammar) |
| Prefix caching | How much faster is the first token when the shared site instructions come first in the prompt? | RadixAttention prefix cache |
| Concurrency | How do throughput and latency change as 1 to 32 cameras call at once? | Continuous batching |
| Image size | Do smaller frames cut prompt tokens and latency, and what happens to accuracy? | Vision preprocessing |
| Quantization | What do 4-bit and 8-bit weights change in GPU memory, speed and accuracy? | AWQ, FP8 |

A cost model turns measured throughput into dollars per 1,000 alerts for a rented GPU and puts it
beside the per-call price of a cloud API.

## Model variants

All variants are the same model, `Qwen/Qwen3-VL-2B-Instruct`, so differences come from the weights
format and not from the model. Each repository was checked on Hugging Face on 2026-10-12.

| Variant | Weights | Size | Status |
| --- | --- | --- | --- |
| `bf16` | 16-bit baseline | 4.26 GB | expected to work |
| `bf16-noradix` | baseline with the prefix cache off | 4.26 GB | expected to work |
| `awq-w4a16` | 4-bit AWQ, pre-quantized | 2.23 GB | expected to work |
| `fp8` | 8-bit FP8, official Qwen checkpoint | 3.47 GB | expected to work |
| `fp8-online` | FP8 quantized by SGLang at load time | 4.26 GB download | experimental |
| `autoround-w4a16` | 4-bit AutoRound | 2.17 GB | experimental |

"Expected to work" is a reading of the SGLang documentation and source, not a test result. A
variant that fails to start is recorded as a failure and shown in the report with the last line
of its server log.
Sources for every choice are in [docs/serving-notes.md](docs/serving-notes.md).

## Data

Frames come from [UCF-Crime](https://www.crcv.ucf.edu/chenchen/datasets/), a public set of real
surveillance videos. `scripts/download_data.py` fetches only the selected videos from inside the
remote zip archives (about 0.95 GB for 36 videos) and `scripts/extract_frames.py` turns them into
363 labelled frames at 320x240.

The dataset's authors allow research use. No video or frame is stored in this repository; the
scripts download them.

The labels have limits that the report states with the numbers:

- Shoplifting frames use the dataset's per-frame event annotations.
- Only 5 of the 12 fighting videos have such annotations. The other fighting frames carry the class
  of the whole video, which is a weak label: not every frame of a fight video shows a fight.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

On any machine, with no GPU:

```bash
uv sync
uv run pytest -q                       # 178 tests, about 25 seconds
uv run scripts/download_data.py        # about 0.95 GB, capped at 3 GB
uv run scripts/extract_frames.py
```

A full rehearsal against the built-in mock server, which imitates the API and returns made-up
answers and timings:

```bash
uv run python -m camalerts.mock_server --port 30999 &
bash scripts/run_all.sh mock-demo http://127.0.0.1:30999
```

On an NVIDIA GPU machine, every variant in turn (see [RUNPOD.md](RUNPOD.md)):

```bash
bash scripts/runpod_setup.sh && source /workspace/camalerts_env.sh
bash scripts/run_matrix.sh
```

Against a server that cannot read images, a written scene can stand in for each frame. This checks
the client against a real server; it says nothing about how a model reads pictures:

```bash
TEXT_ONLY=1 bash scripts/run_all.sh ollama-qwen http://127.0.0.1:11434
```

One such run, against a local Ollama model, is recorded in
[docs/local-smoke-test.md](docs/local-smoke-test.md).

## Layout

```
configs/   data.yaml, models.yaml (variants), alerts.yaml (questions, schema, site brief), bench.yaml
scripts/   download_data.py, extract_frames.py, runpod_setup.sh, serve.py, gpu_mem.py,
           run_all.sh (one variant), run_matrix.sh (all variants)
src/camalerts/   client, prompts, schema, scoring, cost, the four benchmarks, report, mock_server
tests/     pytest; needs no GPU and no downloaded data
docs/      DESIGN.md (contracts between the parts), serving-notes.md (verified facts with sources),
           local-smoke-test.md (a text-only run against a local Ollama model)
results/   written by the runs; REPORT.md is the summary
```

## Limits

- Speed and memory numbers from a data-centre GPU do not transfer to an edge device such as a
  Jetson. This project measures the serving software and the method, not edge hardware.
- UCF-Crime frames are 320x240 and are not footage from any particular deployment.
- Rows produced by the mock server or in text-only mode are labelled as such in the report and are
  not measurements of SGLang.
- Most frames show no event, so accuracy alone is a weak measure here. The report prints recall and
  precision as counts beside it. The samples are small: one frame moves a figure by a few points.
- Cloud API prices in `configs/bench.yaml` are list prices on the date recorded there. The GPU
  price is left empty until it is filled in from a real bill.
