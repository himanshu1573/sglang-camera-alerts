# Camera-alert benchmark report

Generated 2026-10-13T13:53:00+00:00 from `results/`.

**Weak labels:** for autoround-w4a16, awq-w4a16, bf16, bf16-noradix, fp8, fp8-online, some or all frames had no per-frame `in_event` annotation, so the ground truth is the class of the whole video. Not every frame of a shoplifting or fighting video shows the act, so accuracy, precision and recall are rough.

## Variants

`load s` is the time from starting the server process until it answers `/health`: Python start-up, weight loading and CUDA graph capture together, so it is not the weight loading time alone. `GPU idle MB` is the first `nvidia-smi` sample after the server is ready and includes the KV pool; `GPU peak MB` is the highest sample during the tests and includes memory that PyTorch keeps reserved.

| variant | served model | HF model | quantization | load s | weights GB | KV pool tokens | GPU idle MB | GPU peak MB | GPU total MB | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autoround-w4a16 | Vishva007/Qwen3-VL-2B-Instruct-W4A16-AutoRound-AWQ | Vishva007/Qwen3-VL-2B-Instruct-W4A16-AutoRound-AWQ | autoround-w4a16 | 309.4 | 2.15 | 65536 | 11207 | 31783 | 32623 | 4/4 tests |
| awq-w4a16 | cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit | cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit | awq-w4a16 | 325.4 | 2.26 | 65536 | 11173 | 31649 | 32623 | 4/4 tests |
| bf16 | Qwen/Qwen3-VL-2B-Instruct | Qwen/Qwen3-VL-2B-Instruct | none | 297.5 | 4.18 | 65536 | 13239 | 31997 | 32623 | 4/4 tests |
| bf16-noradix | Qwen/Qwen3-VL-2B-Instruct | Qwen/Qwen3-VL-2B-Instruct | none | 235.5 | 4.18 | 65536 | 13175 | 15321 | 32623 | 1/4 tests |
| fp8 | Qwen/Qwen3-VL-2B-Instruct-FP8 | Qwen/Qwen3-VL-2B-Instruct-FP8 | fp8 | 295.4 | 2.79 | 65536 | 11759 | 31975 | 32623 | 4/4 tests |
| fp8-online | Qwen/Qwen3-VL-2B-Instruct | Qwen/Qwen3-VL-2B-Instruct | fp8-online | 233.4 | 2.80 | 65536 | 11771 | 31583 | 32623 | 4/4 tests |

## JSON validity (`json_validity`)

The same frames and questions in each mode. `constrained` sends `response_format`; the other modes do not. `prompt asks with` is the output instruction in the prompt: `schema` pastes the JSON schema, `unconstrained` only names the fields. Compare `constrained` with `schema_prompt_only` to see the decoder alone, since their prompts are identical. Rates are over answered requests. `truncated` counts answers cut off by the token limit. The modes are sent mixed together, several at a time, so `mean total ms` is a rough guide and not a clean per-mode latency. In the accuracy cell, recall and precision are counts, and `scored` is how many answers held a usable `event` value; answers that could not be scored are left out of accuracy, so compare modes only when their `scored` counts match.

| variant | mode | prompt asks with | n | errors | strict JSON | schema valid | loose JSON | mean out tokens | truncated | mean total ms (mixed load) | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autoround-w4a16 | constrained | schema | 180 | 0 | 100.0% | 100.0% | 100.0% | 52.0 | 0 | 474 | concealment: acc 76.7%, recall 0/13, precision 0/1, scored 60/60; aggression: acc 78.3%, recall 4/16, precision 4/5, scored 60/60 |
| autoround-w4a16 | schema_prompt_only | schema | 180 | 0 | 93.9% | 0.0% | 93.9% | 64.5 | 12 | 583 | concealment: acc n/a, recall 0/0, precision 0/0, scored 0/60; aggression: acc n/a, recall 0/0, precision 0/0, scored 0/60 |
| autoround-w4a16 | prompt_only | unconstrained | 180 | 0 | 47.8% | 18.9% | 47.8% | 58.2 | 18 | 541 | concealment: acc 73.9%, recall 0/12, precision 0/0, scored 46/60; aggression: acc 92.3%, recall 0/3, precision 0/0, scored 39/60 |
| awq-w4a16 | constrained | schema | 180 | 0 | 100.0% | 100.0% | 100.0% | 56.5 | 0 | 268 | concealment: acc 78.3%, recall 0/13, precision 0/0, scored 60/60; aggression: acc 73.3%, recall 0/16, precision 0/0, scored 60/60 |
| awq-w4a16 | schema_prompt_only | schema | 180 | 0 | 94.4% | 58.3% | 94.4% | 59.2 | 11 | 280 | concealment: acc 76.9%, recall 2/10, precision 2/6, scored 52/60; aggression: acc 77.2%, recall 4/16, precision 4/5, scored 57/60 |
| awq-w4a16 | prompt_only | unconstrained | 180 | 0 | 0.6% | 0.6% | 0.6% | 54.1 | 2 | 260 | concealment: acc 100.0%, recall 0/0, precision 0/0, scored 1/60; aggression: acc n/a, recall 0/0, precision 0/0, scored 0/60 |
| bf16 | constrained | schema | 180 | 0 | 100.0% | 100.0% | 100.0% | 57.8 | 0 | 448 | concealment: acc 78.3%, recall 0/13, precision 0/0, scored 60/60; aggression: acc 75.0%, recall 1/16, precision 1/1, scored 60/60 |
| bf16 | schema_prompt_only | schema | 180 | 0 | 96.7% | 57.8% | 96.7% | 60.9 | 7 | 466 | concealment: acc 74.5%, recall 2/12, precision 2/6, scored 55/60; aggression: acc 80.7%, recall 5/16, precision 5/5, scored 57/60 |
| bf16 | prompt_only | unconstrained | 180 | 0 | 43.9% | 27.2% | 43.9% | 56.7 | 3 | 440 | concealment: acc 78.0%, recall 0/11, precision 0/0, scored 50/60; aggression: acc 92.9%, recall 0/2, precision 0/0, scored 28/60 |
| bf16-noradix | not run |  |  |  |  |  |  |  |  |  |  |
| fp8 | constrained | schema | 180 | 0 | 99.4% | 99.4% | 99.4% | 57.3 | 1 | 332 | concealment: acc 78.0%, recall 0/13, precision 0/0, scored 59/60; aggression: acc 75.0%, recall 1/16, precision 1/1, scored 60/60 |
| fp8 | schema_prompt_only | schema | 180 | 0 | 98.3% | 65.6% | 98.3% | 57.5 | 4 | 332 | concealment: acc 75.0%, recall 5/13, precision 5/11, scored 56/60; aggression: acc 78.3%, recall 6/16, precision 6/9, scored 60/60 |
| fp8 | prompt_only | unconstrained | 180 | 0 | 17.8% | 12.2% | 17.8% | 54.3 | 2 | 313 | concealment: acc 79.3%, recall 0/6, precision 0/0, scored 29/60; aggression: acc 100.0%, recall 0/0, precision 0/0, scored 3/60 |
| fp8-online | constrained | schema | 180 | 0 | 100.0% | 100.0% | 100.0% | 58.2 | 0 | 429 | concealment: acc 78.3%, recall 0/13, precision 0/0, scored 60/60; aggression: acc 73.3%, recall 0/16, precision 0/0, scored 60/60 |
| fp8-online | schema_prompt_only | schema | 180 | 0 | 93.3% | 55.0% | 93.3% | 62.2 | 12 | 453 | concealment: acc 77.6%, recall 0/11, precision 0/0, scored 49/60; aggression: acc 79.7%, recall 5/16, precision 5/6, scored 59/60 |
| fp8-online | prompt_only | unconstrained | 180 | 0 | 3.9% | 1.7% | 3.9% | 57.4 | 6 | 420 | concealment: acc 33.3%, recall 0/2, precision 0/0, scored 3/60; aggression: acc 100.0%, recall 0/0, precision 0/0, scored 3/60 |

## Prefix cache (`prefix_cache`)

Sequential requests. `shared_first` starts with the site brief; `unique_first` starts with a per-request line, so nothing can be reused. The ratio is mean TTFT of unique_first over shared_first. `cached tokens` is the mean the server reported, counting a request with no report as 0, and `cache hits` is how many requests reported any. A server that never reports the field shows 0 hits even if it does cache.

| variant | ordering | n | errors | mean TTFT ms | p50 TTFT ms | p95 TTFT ms | mean total ms | prompt tokens | cached tokens | cache hits | TTFT ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autoround-w4a16 | shared_first | 30 | 0 | 40 | 39 | 60 | 434 | 1685.9 | 1545.0 | 30/30 |  |
| autoround-w4a16 | unique_first | 30 | 0 | 84 | 81 | 102 | 491 | 1686.8 | 11.9 | 30/30 | 2.10 |
| awq-w4a16 | shared_first | 30 | 0 | 28 | 28 | 32 | 230 | 1686.0 | 1545.0 | 30/30 |  |
| awq-w4a16 | unique_first | 30 | 0 | 66 | 65 | 75 | 269 | 1686.8 | 12.1 | 30/30 | 2.35 |
| bf16 | shared_first | 30 | 0 | 37 | 39 | 45 | 388 | 1685.7 | 1545.0 | 30/30 |  |
| bf16 | unique_first | 30 | 0 | 75 | 72 | 89 | 439 | 1686.8 | 12.1 | 30/30 | 2.00 |
| bf16-noradix | shared_first | 30 | 0 | 84 | 84 | 110 | 433 | 1686.0 | 0.0 | 0/30 |  |
| bf16-noradix | unique_first | 30 | 0 | 74 | 76 | 83 | 434 | 1686.6 | 0.0 | 0/30 | 0.87 |
| fp8 | shared_first | 30 | 0 | 36 | 33 | 46 | 301 | 1686.0 | 1545.0 | 30/30 |  |
| fp8 | unique_first | 30 | 0 | 57 | 59 | 63 | 329 | 1687.4 | 12.2 | 30/30 | 1.61 |
| fp8-online | shared_first | 30 | 0 | 37 | 36 | 47 | 404 | 1686.5 | 1545.0 | 30/30 |  |
| fp8-online | unique_first | 30 | 0 | 52 | 52 | 61 | 401 | 1687.1 | 12.2 | 30/30 | 1.40 |

## Concurrency (`concurrency`)

Closed loop: C workers send requests back to back, one question per frame. `repeats` is how many requests reused a prompt already sent at that level, which a prefix cache answers faster. `cached tokens` is the mean per request; it should be similar at every level, or the levels did not see the same cache state. USD per 1k alerts = GPU price per hour / (requests per second x 3600) x 1000, and reads `not set` until `prices.gpu_usd_per_hour` is filled in. It assumes the GPU is busy all the time at that level and counts one question as one alert, so it is a lower bound on real cost.

| variant | C | n | repeats | errors | req/s | out tok/s | p50 total ms | p95 total ms | p50 TTFT ms | p95 TTFT ms | cached tokens | USD per 1k alerts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autoround-w4a16 | 1 | 32 | 0 | 0 | 2.28 | 126.1 | 447 | 521 | 32 | 61 | 1545.0 | 0.0901 |
| autoround-w4a16 | 2 | 32 | 0 | 0 | 4.17 | 230.6 | 488 | 522 | 40 | 48 | 1545.0 | 0.0492 |
| autoround-w4a16 | 4 | 32 | 0 | 0 | 7.68 | 424.8 | 517 | 558 | 36 | 59 | 1545.0 | 0.0268 |
| autoround-w4a16 | 8 | 64 | 0 | 0 | 12.84 | 691.1 | 602 | 690 | 59 | 119 | 1545.0 | 0.0160 |
| autoround-w4a16 | 16 | 128 | 0 | 0 | 20.99 | 1108.5 | 762 | 936 | 62 | 292 | 1545.0 | 0.0098 |
| autoround-w4a16 | 32 | 256 | 0 | 0 | 25.42 | 1324.5 | 1245 | 1765 | 95 | 753 | 1545.4 | 0.0081 |
| awq-w4a16 | 1 | 32 | 0 | 0 | 4.35 | 260.6 | 234 | 258 | 30 | 38 | 1545.0 | 0.0472 |
| awq-w4a16 | 2 | 32 | 0 | 0 | 7.81 | 467.7 | 258 | 290 | 30 | 48 | 1545.0 | 0.0263 |
| awq-w4a16 | 4 | 32 | 0 | 0 | 13.70 | 819.4 | 290 | 313 | 30 | 51 | 1545.0 | 0.0150 |
| awq-w4a16 | 8 | 64 | 0 | 0 | 20.52 | 1174.8 | 368 | 475 | 44 | 161 | 1545.0 | 0.0100 |
| awq-w4a16 | 16 | 128 | 0 | 0 | 27.57 | 1589.3 | 563 | 725 | 67 | 271 | 1545.0 | 0.0075 |
| awq-w4a16 | 32 | 256 | 0 | 0 | 28.41 | 1612.3 | 1096 | 1626 | 115 | 710 | 1545.4 | 0.0072 |
| bf16 | 1 | 32 | 0 | 0 | 2.60 | 156.5 | 391 | 441 | 32 | 39 | 1545.0 | 0.0790 |
| bf16 | 2 | 32 | 0 | 0 | 4.69 | 282.3 | 428 | 482 | 33 | 51 | 1545.0 | 0.0438 |
| bf16 | 4 | 32 | 0 | 0 | 8.37 | 505.0 | 471 | 504 | 37 | 63 | 1545.0 | 0.0246 |
| bf16 | 8 | 64 | 0 | 0 | 14.01 | 816.5 | 539 | 672 | 49 | 222 | 1545.0 | 0.0147 |
| bf16 | 16 | 128 | 0 | 0 | 19.99 | 1166.6 | 769 | 1104 | 66 | 238 | 1545.0 | 0.0103 |
| bf16 | 32 | 256 | 0 | 0 | 27.14 | 1548.5 | 1147 | 1651 | 73 | 836 | 1545.4 | 0.0076 |
| bf16-noradix | not run |  |  |  |  |  |  |  |  |  |  |  |
| fp8 | 1 | 32 | 0 | 0 | 3.43 | 207.9 | 297 | 332 | 32 | 41 | 1545.0 | 0.0599 |
| fp8 | 2 | 32 | 0 | 0 | 6.08 | 368.8 | 329 | 368 | 29 | 52 | 1545.0 | 0.0338 |
| fp8 | 4 | 32 | 0 | 0 | 11.32 | 683.5 | 346 | 389 | 31 | 58 | 1545.0 | 0.0182 |
| fp8 | 8 | 64 | 0 | 0 | 18.37 | 1078.7 | 427 | 510 | 48 | 131 | 1545.0 | 0.0112 |
| fp8 | 16 | 128 | 0 | 0 | 25.62 | 1491.2 | 623 | 818 | 63 | 298 | 1545.0 | 0.0080 |
| fp8 | 32 | 256 | 0 | 0 | 29.52 | 1672.6 | 1103 | 1421 | 65 | 648 | 1545.4 | 0.0070 |
| fp8-online | 1 | 32 | 0 | 0 | 2.67 | 162.5 | 381 | 423 | 32 | 39 | 1545.0 | 0.0770 |
| fp8-online | 2 | 32 | 0 | 0 | 4.84 | 294.5 | 412 | 473 | 35 | 44 | 1545.0 | 0.0425 |
| fp8-online | 4 | 32 | 0 | 0 | 8.58 | 522.9 | 449 | 508 | 40 | 57 | 1545.0 | 0.0240 |
| fp8-online | 8 | 64 | 0 | 0 | 15.13 | 888.7 | 509 | 629 | 43 | 74 | 1545.0 | 0.0136 |
| fp8-online | 16 | 128 | 0 | 0 | 22.58 | 1321.9 | 699 | 875 | 58 | 233 | 1545.0 | 0.0091 |
| fp8-online | 32 | 256 | 0 | 0 | 27.33 | 1584.2 | 1164 | 1577 | 90 | 625 | 1545.4 | 0.0075 |

## Image size (`image_size`)

The same frames, downscaled so the longer side is at most `max side` (never upscaled). One question per frame.

| variant | max side | sent px | n | errors | prompt tokens | mean TTFT ms | mean total ms | schema valid | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autoround-w4a16 | native | 320x240 | 60 | 0 | 1644.5 | 35 | 419 | 100.0% | concealment: acc 83.3%, recall 0/4, precision 0/1, scored 30/30; aggression: acc 76.7%, recall 2/8, precision 2/3, scored 30/30 |
| autoround-w4a16 | 256 | 256x192 | 60 | 0 | 1634.5 | 37 | 430 | 100.0% | concealment: acc 83.3%, recall 0/4, precision 0/1, scored 30/30; aggression: acc 76.7%, recall 2/8, precision 2/3, scored 30/30 |
| autoround-w4a16 | 192 | 192x144 | 60 | 0 | 1634.5 | 38 | 432 | 100.0% | concealment: acc 83.3%, recall 0/4, precision 0/1, scored 30/30; aggression: acc 76.7%, recall 1/8, precision 1/1, scored 30/30 |
| autoround-w4a16 | 128 | 128x96 | 60 | 0 | 1634.5 | 35 | 420 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 76.7%, recall 1/8, precision 1/1, scored 30/30 |
| awq-w4a16 | native | 320x240 | 60 | 0 | 1644.5 | 32 | 228 | 98.3% | concealment: acc 86.2%, recall 0/4, precision 0/0, scored 29/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| awq-w4a16 | 256 | 256x192 | 60 | 0 | 1634.5 | 33 | 224 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| awq-w4a16 | 192 | 192x144 | 60 | 0 | 1634.5 | 34 | 226 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| awq-w4a16 | 128 | 128x96 | 60 | 0 | 1634.5 | 33 | 228 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| bf16 | native | 320x240 | 60 | 0 | 1644.5 | 39 | 387 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| bf16 | 256 | 256x192 | 60 | 0 | 1634.5 | 34 | 386 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| bf16 | 192 | 192x144 | 60 | 0 | 1634.5 | 31 | 380 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| bf16 | 128 | 128x96 | 60 | 0 | 1634.5 | 29 | 379 | 98.3% | concealment: acc 86.2%, recall 0/4, precision 0/0, scored 29/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| bf16-noradix | not run |  |  |  |  |  |  |  |  |
| fp8 | native | 320x240 | 60 | 0 | 1644.5 | 32 | 288 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| fp8 | 256 | 256x192 | 60 | 0 | 1634.5 | 34 | 292 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| fp8 | 192 | 192x144 | 60 | 0 | 1634.5 | 35 | 291 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| fp8 | 128 | 128x96 | 60 | 0 | 1634.5 | 35 | 285 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| fp8-online | native | 320x240 | 60 | 0 | 1644.5 | 33 | 379 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| fp8-online | 256 | 256x192 | 60 | 0 | 1634.5 | 33 | 377 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| fp8-online | 192 | 192x144 | 60 | 0 | 1634.5 | 38 | 387 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |
| fp8-online | 128 | 128x96 | 60 | 0 | 1634.5 | 36 | 378 | 100.0% | concealment: acc 86.7%, recall 0/4, precision 0/0, scored 30/30; aggression: acc 73.3%, recall 0/8, precision 0/0, scored 30/30 |

## Cloud API reference price

What one alert of the same shape would cost on a per-token cloud API. Image tokens follow the provider's published rule; prompt text tokens are estimated from character count; output tokens are the mean measured in the concurrency run. Prices are list prices on the date shown, with no caching or batch discounts.

| variant | cloud model | prices as of | image tokens | prompt text tokens (est.) | output tokens | input USD/1M | output USD/1M | USD per 1k alerts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autoround-w4a16 | gemini-3.8-flash | 2026-10-12 | 258 | 1778 | 53.0 | 0.75 | 3.75 | 1.7259 |
| autoround-w4a16 | gemini-3.1-flash-lite | 2026-10-12 | 258 | 1778 | 53.0 | 0.25 | 1.5 | 0.5885 |
| awq-w4a16 | gemini-3.8-flash | 2026-10-12 | 258 | 1778 | 57.6 | 0.75 | 3.75 | 1.7429 |
| awq-w4a16 | gemini-3.1-flash-lite | 2026-10-12 | 258 | 1778 | 57.6 | 0.25 | 1.5 | 0.5954 |
| bf16 | gemini-3.8-flash | 2026-10-12 | 258 | 1778 | 58.1 | 0.75 | 3.75 | 1.7447 |
| bf16 | gemini-3.1-flash-lite | 2026-10-12 | 258 | 1778 | 58.1 | 0.25 | 1.5 | 0.5961 |
| fp8 | gemini-3.8-flash | 2026-10-12 | 258 | 1778 | 58.0 | 0.75 | 3.75 | 1.7443 |
| fp8 | gemini-3.1-flash-lite | 2026-10-12 | 258 | 1778 | 58.0 | 0.25 | 1.5 | 0.5959 |
| fp8-online | gemini-3.8-flash | 2026-10-12 | 258 | 1778 | 58.7 | 0.75 | 3.75 | 1.7472 |
| fp8-online | gemini-3.1-flash-lite | 2026-10-12 | 258 | 1778 | 58.7 | 0.25 | 1.5 | 0.5971 |
