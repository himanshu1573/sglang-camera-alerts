# Serving notes

Everything here was checked on **2026-10-12** against the linked pages.
Nothing has run on a GPU yet. The last section lists what is still unverified.

Three kinds of evidence are used and labelled:

- **docs** — stated on a documentation page or model card.
- **source** — read in the SGLang code at tag `v0.5.21`. True for that release, not promised by the docs.
- **inferred** — my reading of the two above. Treat as a guess until a run confirms it.

## 1. Quick start on the pod

```bash
cd /workspace && git clone <this repo> && cd sglang-camera-alerts
bash scripts/runpod_setup.sh            # add PREFETCH_MODELS=1 to download all models now
source /workspace/camalerts_env.sh
uv run python scripts/serve.py --variant bf16 --dry-run
uv run python scripts/serve.py --variant bf16 --detach     # returns when /health answers 200
uv run python scripts/gpu_mem.py --variant bf16 &          # start before the benchmark
# ... run the benchmark against http://127.0.0.1:30000 ...
kill -TERM %1                                              # writes results/bf16/gpu_mem.json
uv run python scripts/serve.py --variant bf16 --stop
```

`serve.py` writes `results/<id>/server.log` and then either `server.json` (ready) or
`server_error.json` (failed, with the last 60 log lines, exit code non-zero).

## 2. SGLang release and install

| Fact | Value | Source |
| - | - | - |
| Latest release | `0.5.21`, uploaded to PyPI 2026-10-01, GitHub release 2026-10-02 | https://pypi.org/pypi/sglang/json , https://github.com/sgl-project/sglang/releases/tag/v0.5.21 |
| Python | 3.10 or higher. The docs example uses 3.12. | https://docs.sglang.io/docs/get-started/install |
| CUDA | CUDA 13 only. CUDA 12 wheels and images are retired. | same page |
| NVIDIA driver | CUDA 13.x needs driver 580 or newer. | https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html |
| pip install | `uv venv --python 3.12` then `uv pip install --prerelease=allow sglang` | https://docs.sglang.io/docs/get-started/install |
| Pinned install | `uv pip install --prerelease=allow sglang==0.5.21` | release notes, link above |
| Docker image | `lmsysorg/sglang:v0.5.21` (same digest as `latest`, 15.2 GB compressed) | https://hub.docker.com/r/lmsysorg/sglang/tags |
| Smaller image | `lmsysorg/sglang:v0.5.21-runtime` (13.7 GB compressed, fewer tools) | same |
| Last CUDA 12 image | `lmsysorg/sglang:v0.5.19-cu129` | install page and Docker Hub |
| Launch | `python -m sglang.launch_server ...` still works. `sglang serve` is the newer name. | install page; source `launch_server.py` |
| Ready message in the log | `The server is fired up and ready to roll!` | https://docs.sglang.io/docs/get-started/quickstart |

SGLang 0.5.21 pins `torch==2.13.0`, `transformers==5.12.1`, `openai==2.6.1`, `xgrammar==0.2.7`
(PyPI metadata). These would fight with this repo's dependencies, so `runpod_setup.sh` puts
SGLang in its own environment (`/workspace/venvs/sglang`) and exports `SGLANG_PYTHON`.
`serve.py` starts the server with that interpreter.

If pip fails with `CUDA_HOME environment variable is not set`, the install page says to set
`CUDA_HOME` to the CUDA toolkit directory. If FlashInfer fails on the GPU, it says to add
`--attention-backend triton --sampling-backend pytorch`.

## 3. Base model

**`Qwen/Qwen3-VL-2B-Instruct`** — https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

- Exists. Not gated. Licence Apache-2.0. (Hugging Face API, `https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct`)
- 2,127,532,032 parameters stored as BF16. `model.safetensors` is 4.26 GB. (same API)
- Architecture `Qwen3VLForConditionalGeneration`, model type `qwen3_vl`. (its `config.json`)
- SGLang lists Qwen-VL as a supported multimodal family.
  (https://docs.sglang.io/docs/supported-models/multimodal_language_models)
- SGLang has a Qwen3-VL cookbook page whose command builder offers size **2B** with **BF16** or **FP8**.
  (https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-VL)
- It is the Instruct edition, so no reasoning parser is needed.

Not chosen: `Qwen/Qwen2.5-VL-3B-Instruct` exists and is not gated, but the API reports no
licence tag and it is larger (7.5 GB).

## 4. Quantized checkpoints of the same model

All rows were read from the Hugging Face API on 2026-10-12. Size is the weights file.

| Repo | Gated | Licence | Size | Publisher | Made with | Used |
| - | - | - | - | - | - | - |
| [cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit) | no | Apache-2.0 | 2.23 GB | cyankiwi (community) | LLM Compressor `AWQModifier`, 4-bit, group 32, compressed-tensors format | `awq-w4a16` |
| [Qwen/Qwen3-VL-2B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8) | no | Apache-2.0 | 3.47 GB | Qwen (official) | block-wise FP8, block size 128 | `fp8` |
| [Vishva007/Qwen3-VL-2B-Instruct-W4A16-AutoRound-AWQ](https://huggingface.co/Vishva007/Qwen3-VL-2B-Instruct-W4A16-AutoRound-AWQ) | no | Apache-2.0 | 2.17 GB | Vishva007 (community) | Intel AutoRound 0.10.0, 4-bit, group 128, AWQ export format | `autoround-w4a16` |
| [cyankiwi/Qwen3-VL-2B-Instruct-AWQ-8bit](https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-8bit) | no | Apache-2.0 | 2.93 GB | cyankiwi | LLM Compressor, 8-bit | no: 8-bit integer weights are not what the benchmark compares |
| [h2oai/Qwen3-VL-2B-Instruct-GPTQ-Int4](https://huggingface.co/h2oai/Qwen3-VL-2B-Instruct-GPTQ-Int4) | no | none stated | 2.17 GB | h2oai | GPTQModel 6.0.3 | no: SGLang docs say GPTQ fails on Qwen2.5-VL; no licence |
| [unsloth/Qwen3-VL-2B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-bnb-4bit) | no | Apache-2.0 | 2.16 GB | unsloth | bitsandbytes 4-bit | no: `bitsandbytes` is not installed with SGLang (only in its `test` extra) |
| [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) | no | Apache-2.0 | several files | Qwen (official) | llama.cpp GGUF | no: needs a separate vision projector file; not shown for VLMs in SGLang docs |
| [embedl/Cosmos-Reason2-2B-W4A16](https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16) | **yes (auto)** | other | 2.79 GB | embedl | W4A16 of `nvidia/Cosmos-Reason2-2B` | no: gated, different base model |

What the SGLang quantization page says (https://docs.sglang.io/docs/advanced_features/quantization):

- Offline (pre-quantized) checkpoints are recommended over online quantization.
- For a pre-quantized model, do **not** pass `--quantization`. The method is read from the model's config.
- "Limited support for quantized VLMs": on Qwen2.5-VL-7B, AWQ and `auto_round:auto_awq` work, GPTQ
  fails, `auto_round:auto_gptq` gives near-zero accuracy.
- On NVIDIA GPUs `gptq` was removed in favour of `gptq_marlin`.
- Online quantization example: `--quantization fp8` on a 16-bit checkpoint.

From the source (tag `v0.5.21`):

- SGLang builds the Qwen3-VL vision tower with no quantization at all; only the language model
  gets the quantization config. All three chosen checkpoints also leave the vision tower in
  16-bit, so they agree with that. (`python/sglang/srt/models/qwen3_vl.py`)
- A compressed-tensors checkpoint with 4-bit grouped weights and no activation quantization is
  served by the `CompressedTensorsWNA16` scheme on the Marlin kernel.
  (`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`)
- The fast FP8 kernel is only enabled on GPUs with compute capability 8.9 or 9.0 and above
  (Ada, Hopper, Blackwell). (`python/sglang/srt/layers/quantization/fp8_utils.py`)

## 5. Variants

Defined in `configs/models.yaml`. "Expected effect" is a prediction, not a measurement.

| id | What | Expected memory effect | Expected speed effect | Status | Main risk |
| - | - | - | - | - | - |
| `bf16` | Base model, 16-bit | Baseline. 4.26 GB of weights. | Baseline. | expected | CUDA 13 / driver mismatch on the pod. |
| `bf16-noradix` | Base model, `--disable-radix-cache` | Same as `bf16`. | Slower first token when prompts share a prefix. | expected | None beyond `bf16`. |
| `awq-w4a16` | 4-bit AWQ weights, 16-bit activations | About half the weight file (2.23 GB). | Unknown. A 2B model is small, so the gain may be small. | expected | Community checkpoint. The docs' AWQ evidence is for Qwen2.5-VL-7B, not this file. |
| `fp8` | Official block-wise FP8 | Weight file 3.47 GB. | Likely a gain only on H100-class GPUs. | expected | On RTX 4090 / L4 the block-wise FP8 matmul may use the slower Triton fallback. |
| `fp8-online` | Base model + `--quantization fp8` | Between `bf16` and `fp8`. Peak memory during load may be higher. | Unknown. Slower start-up. | experimental | Not documented for any VLM. |
| `autoround-w4a16` | 4-bit AutoRound weights in AWQ format | About half the weight file (2.17 GB). | Unknown. | experimental | Little-used community checkpoint. Its config has `zero_point: false`, so it cannot use the Marlin kernel and goes through the plain `awq` path, which SGLang logs as "not fully optimized" and possibly slower than 16-bit (source: `awq/awq.py`, `marlin_utils.py`, `model_config.py`). |

The embeddings and the vision tower stay 16-bit in every quantized checkpoint. That is why the
files are not a quarter or a half of the baseline.

`--torchao-config` and `torchao` are gone: neither appears on the current server-arguments or
quantization pages. So there is no torchao int4 variant.

## 6. Server arguments

All from https://docs.sglang.io/docs/advanced_features/server_arguments unless noted.

| Flag | What the page says | Used |
| - | - | - |
| `--model-path` (alias `--model`) | Local folder or Hugging Face repo id. | every variant |
| `--host` | Default `127.0.0.1`. | yes (`defaults.host`) |
| `--port` | Default `30000`. | yes (`defaults.port`) |
| `--dtype` | `auto`, `half`, `float16`, `bfloat16`, `float`, `float32`. "half" is recommended for AWQ. | `bf16` variants |
| `--quantization` | `awq`, `fp8`, `mxfp8`, `gptq`, `marlin`, `gptq_marlin`, `awq_marlin`, `bitsandbytes`, `gguf`, `modelopt`, `modelopt_fp8`, `modelopt_fp4`, `nvfp4_online`, `modelopt_mixed`, `petit_nvfp4`, `w8a8_int8`, `w8a8_fp8`, `moe_wna16`, `w4afp8`, `mxfp4`, `auto-round`, `compressed-tensors`, `modelslim`, `quark`, `quark_int4fp8_moe`, `quark_mxfp4`, `mlx_q4`, `mlx_q8`, `unquant` | `fp8-online` only |
| `--torchao-config` | **Not on the page.** | no |
| `--mem-fraction-static` | Share of GPU memory for weights plus KV-cache pool. Lower it on out-of-memory errors. | no (add it after `--`) |
| `--max-total-tokens` | Size of the token pool. "Typically used for development and debugging." | yes, `65536` |
| `--context-length` | Maximum context. Defaults to the model's config value. | yes, `8192` |
| `--max-running-requests` | Maximum number of running requests. | no |
| `--served-model-name` | Overrides the name returned by `/v1/models`. | no; the name is the repo id |
| `--grammar-backend` | `xgrammar`, `outlines`, `llguidance`, `none`. XGrammar is the default. | yes, `xgrammar` |
| `--enable-metrics` | Prometheus metrics. | yes |
| `--enable-cache-report` | "Return number of cached tokens in usage.prompt_tokens_details for each openai request." | yes |
| `--disable-radix-cache` | "Disable RadixAttention for prefix caching." | `bf16-noradix` |
| `--chat-template` | Built-in template name or file. Not needed: the model ships its own template. | no |
| `--enable-multimodal` | Turns multimodal on. Not used in the Qwen3-VL cookbook commands. | no |
| `--mm-process-config` | JSON with keys `image`, `video`, `audio` for preprocessing limits. | no (see 7e) |
| `--mm-attention-backend` | Attention backend for the vision encoder. | no |
| `--stream-response-default-include-usage` | Adds usage to every stream even without `stream_options`. | no |
| `--sampling-defaults` | `model` (default) reads `generation_config.json`; `openai` uses temperature 1.0. | no |

A fixed `--max-total-tokens` is a deliberate choice. Without it SGLang fills the GPU with KV
cache whatever the model size, and `nvidia-smi` would show almost the same number for every
variant. With the same pool everywhere, the difference in used memory is the weights. If the GPU
is too small for 65,536 tokens, SGLang logs a warning and uses its own smaller value (source:
`kv_cache_configurator.py`). `server.json` records `max_total_num_tokens` so this can be checked.
By my arithmetic from `config.json` (28 layers, 8 KV heads, head size 128, 2 bytes), 65,536 tokens
is about 7 GiB in 16-bit. This is not a measurement.

`serve.py` also records `weights_mem_gb`, parsed from SGLang's `Load weight end ... mem usage=X GB`
log line (source: `model_runner.py`). It is the most direct per-variant memory number.

## 7. Request format

### 7a. Image as a base64 data URL

- docs: the vision page shows the `image_url` content part.
  https://docs.sglang.io/docs/basic_usage/openai_api_vision
- docs: `data:` URLs are named as accepted input ("Local paths and `data:` URLs are not governed
  by the domain allowlist"). https://docs.sglang.io/docs/advanced_features/server_arguments
- source: `load_image` decodes strings that start with `data:` as base64.
  (`python/sglang/srt/utils/common.py`)

```json
{
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Is someone shoplifting in this frame? Answer in JSON."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64_JPEG>"}}
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 64
}
```

Send `temperature` yourself. The server's default comes from the model's `generation_config.json`
(`temperature` 0.7, `top_p` 0.8, `top_k` 20), because `--sampling-defaults` is `model`.

### 7b. JSON schema output

docs: https://docs.sglang.io/docs/advanced_features/structured_outputs

```json
"response_format": {
  "type": "json_schema",
  "json_schema": {
    "name": "alert",
    "schema": {
      "type": "object",
      "properties": {"alert": {"type": "boolean"}, "reason": {"type": "string"}},
      "required": ["alert", "reason"]
    }
  }
}
```

- The page says the output "will be guaranteed to follow the given constraints".
- It also advises describing the JSON format in the prompt for better quality.
- Only one constraint (`json_schema`, `regex` or `ebnf`) per request.
- source: `{"type": "json_object"}` is also accepted. `"strict": false` can leave the output
  unconstrained in some cases, so leave `strict` out. (`entrypoints/openai/protocol.py`)

### 7c. Streaming with usage at the end

- docs: `stream: true` is supported.
  https://docs.sglang.io/docs/basic_usage/openai_api_completions
- source: the request field is `stream_options` with `include_usage` (and
  `continuous_usage_stats`). The final SSE chunk has `"choices": []` and a `usage` object.
  (`entrypoints/openai/protocol.py`, `serving_chat.py`)
- docs: the server flag `--stream-response-default-include-usage` does the same for every request.

```json
{"stream": true, "stream_options": {"include_usage": true}}
```

### 7d. Cached prompt tokens in usage

- docs: with `--enable-cache-report` the server returns the number of cached tokens in
  `usage.prompt_tokens_details`. The flag is in `defaults`, so every variant has it.
- source: the field is `usage.prompt_tokens_details.cached_tokens`. It is only present when the
  count is above zero, so a missing field means 0. (`entrypoints/openai/usage_processor.py`)
- source: `usage.prompt_tokens_details.image_tokens` is reported when the prompt has an image.
- docs: `/metrics` also exposes `sglang:cache_hit_rate` and `sglang:time_to_first_token_seconds`.
  https://docs.sglang.io/docs/references/production_metrics

Shape of the last streamed chunk (placeholders, not measured values):

```json
{
  "choices": [],
  "usage": {
    "prompt_tokens": "<int>",
    "completion_tokens": "<int>",
    "total_tokens": "<int>",
    "prompt_tokens_details": {"cached_tokens": "<int>", "image_tokens": "<int>"}
  }
}
```

The flag that disables prefix caching is **`--disable-radix-cache`** (variant `bf16-noradix`).
`POST /flush_cache` empties the cache on a running server.
(https://docs.sglang.io/docs/basic_usage/native_api)

### 7e. Fewer vision tokens

- docs: `--mm-process-config '{"image":{"max_pixels":1048576}}'` limits image size. The entries
  are passed to the Hugging Face processor as `images_kwargs`.
  https://docs.sglang.io/docs/supported-models/multimodal_language_models
- I found no documented per-request pixel limit for Qwen-VL. The request schema has `detail`
  and `max_dynamic_patch` fields on `image_url` (source), but nothing documents their effect here.
- The simplest control is on the client: resize the JPEG before sending it.
- inferred: the model's `preprocessor_config.json` sets patch size 16 and merge size 2, so one
  image token covers about 32x32 pixels. It also sets a lower bound (`shortest_edge: 65536`
  pixels). Frames below that bound, roughly 256x256, may be scaled **up** to it. If so,
  shrinking a frame further would not reduce tokens unless the server is started with
  `-- --mm-process-config '{"image":{"min_pixels":<N>}}'` (after a bare `--`, `serve.py`
  passes arguments to SGLang unchanged). Check
  `usage.prompt_tokens_details.image_tokens` on the first run.

## 8. Health, readiness and version

| Endpoint | Meaning | Source |
| - | - | - |
| `GET /health` | Health check. Returns 503 while the server is still starting (source). | https://docs.sglang.io/docs/basic_usage/native_api |
| `GET /health_generate` | Health check that generates one token. | same |
| `GET /v1/models` | OpenAI model list. `data[0].id` is the served model name. | same; used by SGLang's own `wait_for_server` (source) |
| `GET /get_model_info` | Model path, tokenizer path, `has_image_understanding`. Deprecated in favour of `/model_info` (source). | same |
| `GET /server_info` | Server arguments and runtime state. Includes `version` and `max_total_num_tokens` (source). | same |
| `GET /metrics` | Prometheus metrics, needs `--enable-metrics`. | https://docs.sglang.io/docs/advanced_features/observability |

`serve.py` polls `/health` every 2 seconds, then reads `/v1/models`, `/server_info` and
`/health_generate`. The SGLang version comes from `/server_info`; if that is missing it uses the
installed package version (`importlib.metadata`), read without importing SGLang.

## 9. RunPod

Sources: https://docs.runpod.io/pods/choose-a-pod , https://docs.runpod.io/references/gpu-types ,
https://docs.runpod.io/pods/storage/types , https://docs.runpod.io/pods/configuration/expose-ports ,
https://docs.runpod.io/pods/manage-pods , https://docs.runpod.io/pods/configuration/use-ssh

**GPU.** The 16-bit model is 4.26 GB, so 16 GB is enough and 24 GB is comfortable.

- First choice: **RTX 4090 (24 GB)** or **L4 (24 GB)**. Both are Ada generation, which SGLang's
  fast FP8 kernel needs (source). RunPod's guide names this tier for 7B-13B LLM inference, so a
  2B model has room to spare.
- RTX 4000 Ada (20 GB) also fits. On 16 GB cards the fixed 65,536-token pool may not fit, and
  the memory comparison would then be uneven.
- Avoid RTX 3090 / A4000 / A5000 / A40 if FP8 matters: they are Ampere.
- To see block-wise FP8 at its best, a short extra run on an H100 is needed (inferred from the
  `--fp8-gemm-backend` table on the quantization page).
- **Driver:** SGLang 0.5.21 needs a CUDA 13 host driver (580+). When deploying, open
  *Additional filters -> CUDA Versions* and pick a CUDA 13 machine. `runpod_setup.sh` stops with
  a clear message if the driver is older.

**Image.** Two ways, both untested:

1. Use `lmsysorg/sglang:v0.5.21` as the pod's container image. SGLang is already installed, and
   `runpod_setup.sh` detects it. The pod must stay running, so set the container start command to
   `sleep infinity`. RunPod says the web terminal and basic SSH work on all pods.
2. Use a RunPod PyTorch template and let `runpod_setup.sh` install SGLang with
   `uv pip install --prerelease=allow sglang==0.5.21` into `/workspace/venvs/sglang`.

**Disk.** The four model repos total about 12.2 GB. `/workspace` is the persistent volume;
the container disk is erased when the pod stops. Suggested: 60 GB volume, 50 GB container disk.
`runpod_setup.sh` sets `HF_HOME=/workspace/hf` so models are downloaded once.
Storage costs $0.10 per GB per month while running (RunPod pricing page), which is small for a day.

**Ports.** Simplest: run the client on the same pod and call `http://127.0.0.1:30000`. Nothing
needs to be exposed and `serve.py` binds to `127.0.0.1`. To call it from outside, start with
`--host 0.0.0.0`, add `30000` under *Expose HTTP Ports*, and use
`https://<POD_ID>-30000.proxy.runpod.net`. That URL is public and the proxy cuts connections at
100 seconds, so add `-- --api-key <key>` and do not time latency through it.

**Price.** Check the RunPod pricing page (https://www.runpod.io/pricing) or the console when
deploying. On 2026-10-12 the page's cards and its embedded data disagreed (RTX 4090: $0.89/hr on
the card, $0.34 and $0.74 in the data), so no price is recorded here. Use the hourly rate the
console shows for the pod you rent in the cost model.

## 10. Notes for the benchmark

- `results/<id>/server.json` has `base_url`, `served_model_name`, `sglang_version`,
  `load_seconds`, `weights_mem_gb`, `max_total_num_tokens`, `gpu` and the exact `command`.
- `load_seconds` includes the download when `model_cached_before_start` is false. Run with
  `PREFETCH_MODELS=1` first, or start each variant twice, before comparing load times.
- One server at a time. Stop a variant before starting the next; they share port 30000.
- Stop `gpu_mem.py` with `kill -TERM`. Background jobs in a bash script ignore Ctrl-C.
- Start `gpu_mem.py` after the server is ready and before the first request, so that `idle_mb`
  is the idle server.
- A variant that fails leaves `server_error.json`. Record it as a failure; do not skip it.

## 11. Still unverified until it runs on a GPU

1. That SGLang 0.5.21 installs and starts on the rented pod (CUDA 13, driver 580+).
2. `scripts/runpod_setup.sh` end to end. It has only passed `bash -n`.
3. That the SGLang Docker image works as a RunPod template with `sleep infinity`.
4. That `cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit` loads and gives sensible answers in SGLang.
5. That `Qwen/Qwen3-VL-2B-Instruct-FP8` runs well on a GPU below H100 class.
6. `fp8-online` and `autoround-w4a16` at all. Both are experimental.
7. Whether any quantized variant is actually faster than `bf16` on a 2B model.
8. That `--max-total-tokens 65536` fits next to the 16-bit weights on the chosen GPU, and that
   every variant ends up with the same pool (`max_total_num_tokens` in `server.json`).
9. The log line used for `weights_mem_gb`. If the format differs the field is `null`.
10. That `/health` turns 200 only when the model is ready (read in source, not observed).
11. The exact `usage` fields in a live response: `cached_tokens`, `image_tokens`.
12. How many prompt tokens a 320x240 frame costs, and whether smaller frames cost fewer (7e).
13. Whether the prefix cache gives hits for text placed before a changing image.
14. JSON-schema decoding together with image input and streaming in one request.
