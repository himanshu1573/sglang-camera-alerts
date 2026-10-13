# sglang-camera-alerts

A benchmark of [SGLang](https://docs.sglang.io/) serving a small vision-language model for
security-camera alerts. The workload is one camera frame and one question in, one short JSON answer
out:

```json
{"event": true, "confidence": "high", "people": 1, "reason": "Person at the right shelf pushes a boxed item into an open backpack."}
```

## Aim

Before anybody plans to serve camera alerts on their own hardware, three questions need numbers:

1. Does the server return JSON that a program can always parse?
2. Is the long, fixed part of the prompt reused from one request to the next?
3. How do speed and memory change with load, and with smaller (quantized) weights?

This project answers them for SGLang with measurements on a rented GPU. It tests the serving
software. It does not show that this small model detects events well, and the results below say so.

## What was done

1. Download 36 public surveillance videos and cut them into 363 labelled frames.
2. Start SGLang with one variant of the model.
3. Send each frame with one question and a fixed site brief. The answer is one JSON object.
4. Run four tests and write every request and answer to a log.
5. Stop the server, start the next variant, and repeat.
6. Build one report from the logs: [results/REPORT.md](results/REPORT.md).

## Results

One run on 2026-10-13 on RunPod: one RTX PRO 4500 Blackwell (32 GB), driver 580.178.04, SGLang
0.5.21, torch 2.13.0+cu130. Five variants ran all four tests, 1,403 requests each with warm-ups
included. `bf16-noradix` has the prefix cache off and ran only the prefix test, 66 requests. No
request failed. The steps are in [RUNPOD.md](RUNPOD.md).

### Serving features, 16-bit baseline

| Question | Result |
| --- | --- |
| Is the JSON valid? | With constrained decoding, 100% of 180 answers matched the schema. The same prompt without it: 57.8%. A request in plain words: 27.2%. |
| Does the prefix cache help? | With the site brief first, 1,545 of 1,686 prompt tokens came from the cache and the first token took 37 ms. With a unique line first: 12 cached tokens, 75 ms. With the cache off: 0 cached tokens, 84 ms and 74 ms. |
| Do smaller frames save tokens? | No. 1,644.5 prompt tokens at 320x240, and 1,634.5 at 256, 192 and 128 pixels. The time for an answer stayed at 379 to 387 ms. |

Concurrency, one question per frame, closed loop:

| Callers | Requests per second | p50 total | p95 total |
| --- | --- | --- | --- |
| 1 | 2.6 | 391 ms | 441 ms |
| 8 | 14.0 | 539 ms | 672 ms |
| 32 | 27.1 | 1,147 ms | 1,651 ms |

### Variants

| Variant | Weights on the GPU | One caller, p50 total | Requests per second, 32 callers | GPU memory, idle | Schema-valid, constrained |
| --- | --- | --- | --- | --- | --- |
| `bf16` | 4.18 GB | 391 ms | 27.1 | 13,239 MiB | 100% |
| `awq-w4a16` | 2.26 GB | 234 ms | 28.4 | 11,173 MiB | 100% |
| `fp8` | 2.79 GB | 297 ms | 29.5 | 11,759 MiB | 99.4% |
| `fp8-online` | 2.80 GB | 381 ms | 27.3 | 11,771 MiB | 100% |
| `autoround-w4a16` | 2.15 GB | 447 ms | 25.4 | 11,207 MiB | 100% |

The 4-bit AWQ weights gave the fastest single answer. At 32 callers every variant ends between 25
and 30 requests per second, so the weights are not what limits throughput there. SGLang logged
that the AutoRound format "is not fully optimized yet", and it was slower than the baseline.

### Detection quality

Recall is the number of real event frames that the model marked as an event.

| Variant | Concealment, constrained | Concealment, schema in the prompt only | Aggression, constrained |
| --- | --- | --- | --- |
| `bf16` | 0 of 13 | 2 of 12 | 1 of 16 |
| `awq-w4a16` | 0 of 13 | 2 of 10 | 0 of 16 |
| `fp8` | 0 of 13 | 5 of 13 | 1 of 16 |
| `fp8-online` | 0 of 13 | 0 of 11 | 0 of 16 |
| `autoround-w4a16` | 0 of 13 | no usable answer | 4 of 16 |

The model answered "no event" for almost every real event. Accuracy of 73% to 78% in the report
only reflects that most frames show no event. So this run cannot say what quantization costs in
accuracy: the baseline already misses the events. Recall was also lower with constrained decoding
than for the same prompt without it. That difference has not been investigated.

### Found while running

- **Start-up.** SGLang did not start on the first three attempts. FlashInfer and DeepGEMM compile
  GPU code at start. The pod image had `nvcc` 12.8, which is too old for a Blackwell GPU, and
  `ninja` was not on `PATH`. `scripts/runpod_setup.sh` now handles both.
- **Start time.** A server needed 230 to 325 s to become ready. The server logs put weight loading
  at 0.7 to 8 s of that.
- **GPU memory.** 7.0 GB of the idle memory is the KV pool of 65,536 tokens that
  `configs/models.yaml` asks for. While requests ran, memory grew until the 32 GB card was full.
  PyTorch then logged one out-of-memory warning, memory dropped by about 16,000 to 19,000 MiB, and
  the run went on with no failed request. The cause has not been investigated.

The report also has the cost table. It assumes a GPU that is busy all the time.

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

| Variant | Weights | Download size | GPU run |
| --- | --- | --- | --- |
| `bf16` | 16-bit baseline | 4.26 GB | all four tests |
| `bf16-noradix` | baseline with the prefix cache off | 4.26 GB | prefix test only, by design |
| `awq-w4a16` | 4-bit AWQ, pre-quantized | 2.23 GB | all four tests |
| `fp8` | 8-bit FP8, official Qwen checkpoint | 3.47 GB | all four tests |
| `fp8-online` | FP8 quantized by SGLang at load time | 4.26 GB | all four tests |
| `autoround-w4a16` | 4-bit AutoRound | 2.17 GB | all four tests |

All six started on SGLang 0.5.21. A variant that fails to start is recorded as a failure and shown
in the report with the last line of its server log.
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
- The results are one run with no repeats, so they give no measure of run-to-run variation.
- Cloud API prices in `configs/bench.yaml` are list prices on the date recorded there. The GPU
  price in the results is 0.74 USD per hour, the price of the pod on the day of the run.

## Next

- Make the accuracy test usable: more frames with a per-frame label, and try `reason` before
  `event` in the schema, so that the model describes the frame before it decides.
- Find the memory budget for a small device: a smaller KV pool, a limit on running requests, and
  the cause of the memory growth.
- Repeat each test several times to see how much the numbers vary.
