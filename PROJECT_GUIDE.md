# Project guide

This file explains the project in simple English: what it does, how each part works, what was done
on 2026-10-12, and what is left for the RunPod day. Read it before you run anything.

## 1. The project in one minute

A security camera sees something. We want a computer to answer a simple question about the picture,
for example "Is someone putting an item into a bag?". The answer must come back in a fixed form
that other software can read:

```json
{"event": true, "confidence": "high", "people": 1, "reason": "Person pushes a box into a backpack."}
```

A small AI model can do this. **SGLang** is the program that runs the model and answers the
requests. This project measures how well SGLang does that job, and which tricks make it faster and
cheaper.

One request looks like this:

```
picture + question  --->  SGLang (running the model)  --->  short JSON answer
```

## 2. Why this project exists

Many teams pay a cloud company for every AI request. Running a small model on your own hardware can
be cheaper and more private, and the answer can come faster. To do that you need a serving program,
and SGLang is one of them. Before trusting it with real work, you want numbers: how fast is it, how
much does each trick help, and how much does a smaller model file cost in wrong answers? This
project produces those numbers for one kind of work, security-camera alerts.

## 3. The five tests

Each test asks one question. The answers go into one report.

### Test 1: Is the answer always in the right form?

- **The problem:** sometimes a model writes broken JSON, or adds extra words. Then the software
  cannot read the answer and the alert is lost.
- **The SGLang feature:** *constrained decoding*. SGLang can force the model to follow the form.
- **What we do:** send the same pictures three ways and count how many answers are valid:
  1. with forcing, and the JSON rules written in the prompt;
  2. without forcing, with the same prompt. Comparing 1 and 2 shows what the forcing alone does;
  3. without forcing, only asking for JSON in words. This is what a simple request gets.

### Test 2: Is it faster when every request starts the same way?

- **The idea:** every request from one site starts with the same long instructions (about 1,100
  words). SGLang can remember this first part, so it does not read it again. This is called
  *prefix caching*.
- **What we do:** send requests with the shared instructions first, then with a different line
  first, so nothing can be reused. We compare the waiting time before the first word arrives.

### Test 3: How many cameras can one GPU serve?

- **The idea:** SGLang can work on many requests at the same moment. This is called *batching*.
- **What we do:** pretend to be 1, 2, 4, 8, 16 and 32 cameras asking at once, one question per
  picture. We measure how many answers come back per second and how long each camera waits.

### Test 4: Does a smaller picture help?

- **The idea:** a picture is turned into tokens, the small pieces a model reads. A smaller picture
  may mean fewer tokens and a faster answer, but the model may also see less.
- **What we do:** send the same pictures in four sizes and compare tokens, time and correct
  answers.

### Test 5: What does quantization change?

This is the main comparison of the project. It is explained in the next section.

## 4. Quantization, explained

A model is a very long list of numbers. Normally each number is stored with 16 bits.
**Quantization** means storing each number with fewer bits, for example 8 or 4. The model file
becomes smaller and needs less GPU memory. It can also become faster. The risk is that the model
becomes a little less correct, because the numbers are less exact.

We test the same model, `Qwen3-VL-2B`, in six forms. All six files were checked on Hugging Face
on 2026-10-12.

| Name in this project | What it is | File size | Will it work? |
| --- | --- | --- | --- |
| `bf16` | Normal, 16 bits. This is the baseline we compare with. | 4.26 GB | Expected |
| `bf16-noradix` | The same, with prefix caching switched off. Only used for Test 2. | 4.26 GB | Expected |
| `awq-w4a16` | 4 bits, made with a method called AWQ. | 2.23 GB | Expected |
| `fp8` | 8 bits, made by the Qwen team. | 3.47 GB | Expected |
| `fp8-online` | SGLang makes the 8-bit version itself when it loads the model. | 4.26 GB download | Experimental |
| `autoround-w4a16` | 4 bits, made with a different method called AutoRound. | 2.17 GB | Experimental |

For each form, the project runs Tests 1 to 4 again and also records how much GPU memory is used.
So at the end you can say, with numbers: "4-bit used this much less memory, was this much faster or
slower, and gave this many more wrong answers".

Three things to keep in mind:

- **"Expected" is not "tested".** It means the SGLang documentation says it should work. Nobody has
  run these exact files with this setup.
- **A failure is also a result.** If one form does not start, the project writes down the error and
  goes on. The report shows it as failed.
- **Smaller is not always faster.** This model is small already. A 4-bit version may save memory
  and still not be faster. We report what we measure, even if it is not what we hoped.

## 5. How this makes inference cheaper

A rented GPU costs a fixed price per hour. So the more answers it gives per hour, the cheaper each
answer is:

```
cost of 1,000 alerts = GPU price per hour / (answers per second x 3600) x 1000
```

Everything in this project is about making "answers per second" bigger or the request smaller:

| Trick | What it reduces |
| --- | --- |
| Forced short JSON answer | The words the model must write |
| Shared instructions first (prefix caching) | The words the model must read again |
| Many requests at once (batching) | Idle time on the GPU |
| Smaller picture | The tokens for the picture |
| Quantization | GPU memory, and maybe time |

The report also shows what the same request would cost at a cloud company's list price, so the two
can be compared.

## 6. The data

- The pictures come from **UCF-Crime**, a public set of real security camera videos.
- We use 36 videos: 12 shoplifting, 12 fighting, 12 normal. That makes 363 pictures.
- The download is about 0.95 GB. The script takes single videos from inside the big zip files, so
  it never downloads a whole 6 GB file.
- Each picture has a label, so we can check if the model's answer is right.
- The labels are good for shoplifting. For fighting they are weaker: only 5 of the 12 videos say
  exactly when the fight happens. The report says this next to the numbers.
- The videos are for research use. They are **not** uploaded to GitHub. Only the download script is.

## 7. What is in each folder

| Folder or file | What it does |
| --- | --- |
| `configs/models.yaml` | The six model forms and how to start each one |
| `configs/alerts.yaml` | The questions, the JSON form, and the long site instructions |
| `configs/bench.yaml` | Test sizes and prices. You put your GPU price here. |
| `configs/data.yaml` | Which videos to download |
| `scripts/download_data.py` | Downloads the videos |
| `scripts/extract_frames.py` | Cuts pictures out of the videos |
| `scripts/runpod_setup.sh` | Installs everything on the rented GPU machine |
| `scripts/serve.py` | Starts and stops SGLang for one model form |
| `scripts/gpu_mem.py` | Records GPU memory while a test runs |
| `scripts/run_all.sh` | Runs the four tests on one model form |
| `scripts/run_matrix.sh` | Runs everything for every model form. This is the main command. |
| `src/camalerts/` | The test code |
| `src/camalerts/mock_server.py` | A fake server, so the code can be tested without a GPU |
| `tests/` | Automatic checks of the code |
| `results/` | Where the results and `REPORT.md` are written |
| `RUNPOD.md` | The step-by-step list for the GPU day |
| `docs/serving-notes.md` | Every SGLang fact we rely on, with its source |

## 8. What was done on 2026-10-12

- The whole project was written, in three parts built at the same time and then joined.
- The data was downloaded and 363 pictures were made.
- The automatic tests pass on the laptop.
- The full run was practised on the laptop, with the fake server in place of SGLang. The script
  started each model form, ran the tests, handled one form that was broken on purpose, and stopped
  everything cleanly.
- The test code was run against a real server: the `qwen2.5:3b` model in Ollama on the laptop. This
  model cannot see pictures, so each picture was replaced by one sentence. The details are in
  `docs/local-smoke-test.md`. The main findings:
  - 112 requests, 0 errors.
  - Putting the shared instructions first made the first word arrive about 14 times sooner
    (0.65 seconds against 9.5 seconds).
  - Ollama answered one request at a time, so 4 cameras waited 4 times longer than 1 camera.
  - The small model wrote valid JSON every time, even without forcing.
  - The run found a real bug in the code, which was fixed.
- The project was reviewed for mistakes before the first commit. The review found real problems,
  and they were fixed. The most important ones:
  - A server that did not stop cleanly could have made every later model form fail.
  - Ctrl-C would not have stopped the run.
  - The report showed only accuracy, which looks good even for a model that never finds anything.
    It now also shows recall and precision.
  - The count of cached tokens could have been too high. It is now counted correctly.

## 9. What is not done and not known

- **Nothing has run on SGLang yet.** SGLang needs an NVIDIA GPU, and the laptop does not have one.
- **Nothing has been tested with real pictures going into a model.** The Ollama test used sentences.
- **The setup script for RunPod is untested.** It was written from the documentation.
- **We do not know the results.** Do not tell anyone what SGLang "will" show. Run it first.

## 10. The RunPod day, in short

The full list is in `RUNPOD.md`. In short:

1. Push the code to GitHub.
2. Rent a GPU with 24 GB (RTX 4090 or L4) on a machine with CUDA 13.
3. Download the code on the pod and run the setup script.
4. Download the pictures.
5. Put the GPU price into `configs/bench.yaml`.
6. Run `bash scripts/run_matrix.sh bf16` to try one form.
7. Run `bash scripts/run_matrix.sh` for all forms.
8. Read `results/REPORT.md`, save the results, and stop the pod.

## 11. Words to know

| Word | Meaning |
| --- | --- |
| Model | The AI "brain" that reads a picture and writes an answer |
| VLM | Vision-language model: a model that can read pictures and text |
| Inference | Using a finished model to answer a request |
| Serving | Running a model as a service that answers many requests |
| SGLang | A serving program. It runs the model and answers requests. |
| Token | A small piece of text or picture that the model reads or writes |
| TTFT | Time to first token: how long you wait before the answer starts |
| Latency | How long one request takes in total |
| Throughput | How many answers come back per second |
| p50 and p95 | p50 is the middle waiting time. p95 is the slow case: 95 of 100 requests are faster. |
| JSON | A fixed text form that programs can read |
| Schema | The rules for the JSON: which fields, and what type each one is |
| Constrained decoding | Forcing the model to follow the schema |
| Prefix caching | Remembering the start of a prompt so it is not read again |
| Batching | Working on several requests at the same moment |
| Quantization | Storing the model's numbers with fewer bits to save memory |
| AWQ, FP8 | Two quantization methods: AWQ uses 4 bits, FP8 uses 8 bits |
| Mock server | A fake server used only to test the code |

## 12. How to read the report

`results/REPORT.md` has one table per test. A few cells need explaining.

**The accuracy cell** looks like this: `concealment: acc 82.5%, recall 5/8, precision 5/7, scored 40/40`.

- `acc` is the share of answers that were right. Be careful with it: most pictures show no event,
  so a model that always says "no event" also gets a high number.
- `recall 5/8` means there were 8 real events and the model found 5 of them. This is the number
  that shows whether the model is useful.
- `precision 5/7` means the model raised 7 alerts and 5 of them were real.
- `scored 40/40` means all 40 answers could be read. If it says 36/40, four answers were broken and
  are not in the other numbers.

**Cache hits** in the prefix table, for example `28/30`, is how many requests the server said it
answered using remembered text. For the form with caching switched off it should be `0/30`.

**KV pool tokens** in the first table must be the same number for every model form. If it is not,
the GPU memory numbers cannot be compared with each other.

**USD per 1k alerts** is the cost of 1,000 questions if the GPU is busy all the time. Real cost is
higher, because a real GPU also waits.

The samples are small. One picture changes a number by a few points, so do not read a small
difference between two model forms as a real difference.
