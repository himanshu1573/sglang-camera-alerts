# Running on RunPod

This is the step-by-step list for the GPU run. It was followed once, on 2026-10-13, on an RTX PRO
4500 Blackwell pod (driver 580.178.04, RunPod PyTorch 2.8.0 template). The setup script was changed
after that run and has not run again since. If a step fails, the "If something goes wrong" part at
the end tells you what to try.

## Before you start (on the laptop)

1. Push the code to GitHub, because the pod will download it from there:

   ```bash
   git push -u origin main
   ```

2. Open the RunPod console and look at the price per hour of the GPU you will rent. Write it down.
   You need it in step 6.

## 1. Rent the pod

| Setting | What to choose | Why |
| --- | --- | --- |
| GPU | RTX 4090, L4 or RTX PRO 4500 | The model needs about 4 GB. 24 GB or more leaves room, and these cards support the fast FP8 path. |
| CUDA version | A machine with CUDA 13 | SGLang 0.5.21 needs NVIDIA driver 580 or newer. After you click a GPU, the right side of the deploy page shows "Available CUDA versions". It must list only 13.x. If 12.8 is also listed, you can get a machine with an older driver. |
| Template | A RunPod PyTorch template | The setup script installs SGLang for you. |
| Container disk | 50 GB | Room for the software. |
| Volume disk | 60 GB "Volume disk", mounted at `/workspace` | Keeps the models if the pod restarts. The four models are about 12 GB. |

Avoid the RTX 3090, A4000, A5000 and A40 if you want the FP8 result: they are an older generation.
Avoid the "MIG" entries too: they are one slice of a GPU that other people also use.

If the pod still says "Initializing" after 5 minutes and its system log is empty, terminate it and
deploy again. A stuck pod costs money.

## 2. Open a terminal and start tmux

Use the web terminal in the RunPod console. Then start `tmux`, a tool that keeps your commands
running if the browser tab closes or the connection drops:

```bash
apt-get update && apt-get install -y tmux
tmux new -s run
```

If you lose the connection, open the terminal again and type `tmux attach -t run`. Your run will
still be there.

## 3. Get the code

```bash
cd /workspace
git clone https://github.com/himanshu1573/sglang-camera-alerts.git
cd sglang-camera-alerts
```

## 4. Install everything

```bash
PREFETCH_MODELS=1 bash scripts/runpod_setup.sh
source /workspace/camalerts_env.sh
```

This installs the project, installs SGLang in its own environment, installs the CUDA compiler that
SGLang needs to build its GPU code, and downloads the models. It prints the versions at the end. `PREFETCH_MODELS=1` downloads the models now, so that download time
is not counted as model loading time later.

If you stop the pod and start it again later, run these two commands again first. They are quick the
second time.

## 5. Get the pictures

```bash
uv run scripts/download_data.py
uv run scripts/extract_frames.py
```

This downloads 36 videos (about 0.95 GB) and makes 363 pictures. You should see "363 frames" at the
end.

## 6. Put in the GPU price

Open `configs/bench.yaml` and change this line, using the price you wrote down:

```yaml
  gpu_usd_per_hour: null      # change null to your price, for example 0.69
```

If you leave it as `null`, everything still runs, and the cost column says "not set".

## 7. Try one variant first

```bash
bash scripts/run_matrix.sh bf16 2>&1 | tee /workspace/run_bf16.log
```

This starts SGLang with the normal 16-bit model, runs the four tests, and stops SGLang. The `tee`
part saves everything you see into a file. If you see tables with numbers and "All variants
finished", it works.

Then check that the server really stopped:

```bash
nvidia-smi
```

The memory in use should be close to zero. If several GB are still in use, the server is still
running. Do not go on to step 8; look at "If something goes wrong".

## 8. Run the other variants

```bash
bash scripts/run_matrix.sh bf16-noradix awq-w4a16 fp8 2>&1 | tee /workspace/run_rest.log
```

These are the other three variants that are expected to work. (Running `bash scripts/run_matrix.sh`
with no names runs all four, including `bf16` again, which costs time for no new result.)

To also try the two experimental ones afterwards:

```bash
bash scripts/run_matrix.sh fp8-online autoround-w4a16 2>&1 | tee /workspace/run_experimental.log
```

On the RTX PRO 4500 the four variants took about 32 minutes and all six about 52 minutes. A full
variant sends 1,403 requests and takes about 9 minutes, of which 4 to 5 minutes is the server
starting. While the server starts, the screen shows nothing new. That is normal.

Things to know while it runs:

- **A variant that fails to start is not a disaster.** The script writes the error to
  `results/<variant>/server_error.json` and goes on. A failed variant goes in the report as
  "failed", with the last line of its log.
- **Ctrl-C stops the whole run** and stops the server. You can start again from the variant where
  you stopped.
- **If the script cannot stop a server, it stops itself** and tells you. That is on purpose: a
  server that is still running would make every later variant fail.

## 9. Read the report

```bash
cat results/REPORT.md
```

## 10. Save the results

The results must leave the pod before you stop it. Make one file with everything:

```bash
tar czf /workspace/results-full.tgz --exclude '*.log' results
ls -lh /workspace/results-full.tgz
```

This includes the `raw` folders, which hold every single request and answer. Keep them: if a number
in the report looks wrong later, they are the only way to check it or count it again. They are
small, a few MB.

Then get the file off the pod. Two ways:

- Download `/workspace/results-full.tgz` with Jupyter: in the pod's *Connect* tab, open the port
  8888 link, right-click the file and choose *Download*. The PyTorch 2.8.0 template has it.
- Or commit the summary files from the pod (this does not include the `raw` folders):

  ```bash
  git config user.name "Your Name"
  git config user.email "you@example.com"
  git add results
  git commit -m "Add GPU benchmark results"
  git push
  ```

  `git push` asks you to log in. GitHub needs a personal access token here, not your password.

Also save the three log files from `/workspace/run_*.log` the same way.

## 11. Stop the pod

Stop or terminate the pod in the RunPod console. You pay for every hour it runs.

## What to check in the results

These are things the documentation suggests but nobody has tested with this setup:

- **Was the prefix cache used?** In the "Prefix cache" table, `cache hits` for `bf16` with
  `shared_first` should be most of the requests, and `cached tokens` should be large. For
  `bf16-noradix` both should be 0. If `bf16` also shows 0 hits, the server did not report cached
  tokens, and the timing difference is then the only evidence.
- **Does a smaller picture use fewer tokens?** The model may scale small pictures back up. If
  prompt tokens stay the same at 256, 192 and 128 pixels, that is the finding.
- **Is the quantized model really faster?** For a model this small, 4-bit or 8-bit weights may save
  memory without saving time. Report what you measure.
- **Is GPU memory comparable?** In the "Variants" table, `KV pool tokens` must be the same number
  for every variant. If it differs, the GPU memory columns are comparing different things.
- **Is the accuracy number meaningful?** Most pictures show no event, so a model that always says
  "no event" gets a high accuracy. Look at `recall` in the accuracy cell: it shows how many of the
  real events the model found.

## If something goes wrong

| Problem | What to try |
| --- | --- |
| The setup script stops and says the driver is too old | Rent a different machine with CUDA 13. Do not use `ALLOW_OLD_DRIVER=1` unless you want to see the real error. |
| The server log says `NVCC version must be at least 12.9` or `FlashInfer requires GPUs with sm75 or higher` | The CUDA compiler on the pod is older than SGLang needs. Run step 4 again; it installs the right one. Check with `nvcc --version`. |
| The server log says `FileNotFoundError: ... 'ninja'` | The SGLang environment is not on `PATH`. Run `source /workspace/camalerts_env.sh`, then `which ninja` must print a path. |
| A variant does not start | Read `results/<variant>/server_error.json`. It holds the last 60 lines of the server log. |
| The server is still running after a run | `uv run python scripts/serve.py --variant <variant> --stop`. If that fails, find it with `nvidia-smi` and stop it with `kill <pid>`. |
| You want to test one thing by hand | Start a server: `uv run python scripts/serve.py --variant bf16 --detach`. Check it: `curl http://127.0.0.1:30000/v1/models`. Run a small test under a different name, so the real results are not overwritten: `uv run python -m camalerts.bench_json --base-url http://127.0.0.1:30000 --variant debug --n 4`. Stop it: `uv run python scripts/serve.py --variant bf16 --stop`. Delete `results/debug` afterwards. |
| Constrained JSON requests all fail with HTTP 400 | The server rejected part of the JSON schema. Copy the error text; the schema is in `configs/alerts.yaml`. |
| Out of disk space | The models are in the folder that `echo $HF_HOME` prints. Check with `df -h /workspace`. |
| `uv` or `python` is "not found" after a pod restart | Run step 4 again. |

More detail, with the source for each fact, is in [docs/serving-notes.md](docs/serving-notes.md).
