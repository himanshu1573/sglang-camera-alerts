# Local smoke test against Ollama (text only)

A check that the benchmark client works against a real server, run on 2026-10-12 before any GPU
was rented. **This is not an SGLang result and no image was sent.** Each frame was replaced by one
written sentence describing a scene (`--text-only`).

- Machine: Apple M1 laptop, Ollama 0.20.3 using the Apple GPU, context window 4,096 tokens.
- Model: `qwen2.5:3b`, a text-only model.
- Size: reduced (12 frames for JSON, 8 measured requests per ordering for prefix, 8 per level for
  concurrency), so these are small samples.
- Ollama has no `/flush_cache` endpoint and does not report cached tokens, so the cache effect below
  is seen in the timings only.

## What it confirmed about the client

- Streaming works, and time to first token is measured.
- Token counts arrive in a separate final chunk with empty `choices`, and the client reads them
  (`usage_source: server` for every request).
- `response_format` with a JSON schema is accepted. Whether it was needed is not shown: the
  unconstrained answers were valid too.
- 112 requests, 0 errors.

It also found a bug. The concurrency benchmark crashed in text-only mode when it computed the cloud
reference price, because a text-only request has no image size. The fix and a regression test are
in `src/camalerts/cost.py` and `tests/test_text_only.py`.

## Numbers

The summary files this run wrote are in [local-smoke-test-results/](local-smoke-test-results/).
They come from the code as it was at that moment. Since then the JSON test has gained a third mode,
and the concurrency test asks one question per frame where it then asked three.

JSON validity, 36 requests per mode:

| Mode | Strict JSON | Schema valid | Mean output tokens | Mean total time |
| --- | --- | --- | --- | --- |
| Constrained (`response_format`) | 100% | 100% | 39.9 | 3,788 ms |
| Prompt only | 100% | 100% | 40.2 | 2,734 ms |

This model wrote valid JSON every time without constrained decoding. The constrained mode was about
one second slower, but its prompt is also longer, because that mode pastes the JSON schema into the
prompt. The two effects are not separated here.

Prefix caching, 8 measured requests per ordering, about 1,625 prompt tokens:

| Ordering | Mean time to first token | p95 | Mean total time |
| --- | --- | --- | --- |
| Shared instructions first | 654 ms | 762 ms | 3,533 ms |
| Unique line first | 9,528 ms | 10,157 ms | 12,169 ms |

Putting the shared instructions first made the first token arrive 14.6 times sooner. On this
machine, reading 1,600 prompt tokens from scratch takes about nine seconds, so reusing them matters
a great deal. A fast GPU reads prompts far quicker, so the gain there will be smaller.

Concurrency, 8 requests per level:

| Simultaneous callers | Requests per second | p50 total time | p50 time to first token |
| --- | --- | --- | --- |
| 1 | 0.29 | 3,360 ms | 415 ms |
| 2 | 0.31 | 6,403 ms | 3,354 ms |
| 4 | 0.29 | 13,028 ms | 10,085 ms |

Throughput stayed flat and waiting time grew in step with the number of callers: this Ollama setup
answers one request at a time. Whether SGLang's batching does better is what the GPU run measures.

The accuracy figures from this run are left out on purpose. The model was reading a sentence that
already described the scene, so they say nothing about reading pictures.
