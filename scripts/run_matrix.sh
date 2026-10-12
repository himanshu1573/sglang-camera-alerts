#!/usr/bin/env bash
# Run the whole comparison on a GPU machine: for each model variant, start SGLang, sample GPU
# memory, run the benchmarks, stop the server. Then rebuild results/REPORT.md.
#
#   scripts/run_matrix.sh                 # every variant with status "expected"
#   scripts/run_matrix.sh bf16 awq-w4a16  # only these
#
# Optional environment: INCLUDE_EXPERIMENTAL=1 (add the experimental variants),
# N (frames per benchmark), RESULTS_DIR (results), PORT (30000).
# A variant that fails to start is recorded in results/<id>/server_error.json and the run goes on.
set -uo pipefail
# `uv run` turns Ctrl-C into an ordinary non-zero exit, so without this the loop would carry on.
trap 'echo "Interrupted." >&2; exit 130' INT TERM

cd "$(dirname "$0")/.."
[ -f /workspace/camalerts_env.sh ] && source /workspace/camalerts_env.sh

RESULTS_DIR="${RESULTS_DIR:-results}"
PORT="${PORT:-30000}"
BASE_URL="http://127.0.0.1:${PORT}"

if [ "$#" -gt 0 ]; then
  variants=("$@")
else
  wanted="expected"
  [ "${INCLUDE_EXPERIMENTAL:-0}" = "1" ] && wanted="expected experimental"
  variants=()
  while IFS= read -r id; do
    variants+=("$id")
  done < <(WANTED="$wanted" uv run python - <<'PY'
import os
import yaml

wanted = os.environ["WANTED"].split()
for variant in yaml.safe_load(open("configs/models.yaml"))["variants"]:
    if variant.get("status", "expected") in wanted:
        print(variant["id"])
PY
  )
fi
if [ "${#variants[@]}" -eq 0 ]; then
  echo "No variants found in configs/models.yaml (is the environment set up? try: uv sync)" >&2
  exit 2
fi
echo "Variants to run: ${variants[*]}"

current=""
sampler=""
cleanup() {
  [ -n "$sampler" ] && kill -TERM "$sampler" 2>/dev/null
  [ -n "$current" ] && uv run python scripts/serve.py --variant "$current" --stop \
    --results-dir "$RESULTS_DIR" >/dev/null 2>&1
}
trap cleanup EXIT

failed=""
for variant in "${variants[@]}"; do
  echo
  echo "################ ${variant} ################"
  if ! uv run python scripts/serve.py --variant "$variant" --port "$PORT" \
      --results-dir "$RESULTS_DIR" --detach; then
    echo "!!! ${variant}: server did not start, see ${RESULTS_DIR}/${variant}/server_error.json" >&2
    failed="${failed} ${variant}(start)"
    continue
  fi
  current="$variant"

  uv run python scripts/gpu_mem.py --variant "$variant" --results-dir "$RESULTS_DIR" &
  sampler=$!

  # The no-prefix-cache variant exists only for the prefix-caching comparison.
  benches=""
  case "$variant" in *-noradix) benches="bench_prefix" ;; esac

  if ! BENCHES="$benches" RESULTS_DIR="$RESULTS_DIR" bash scripts/run_all.sh "$variant" "$BASE_URL"; then
    failed="${failed} ${variant}(bench)"
  fi

  kill -TERM "$sampler" 2>/dev/null
  wait "$sampler" 2>/dev/null
  sampler=""
  if ! uv run python scripts/serve.py --variant "$variant" --stop --results-dir "$RESULTS_DIR"; then
    # A server that is still up holds the port and the GPU, so every later variant would fail.
    echo "!!! could not stop the server for ${variant}. Stopping the run here." >&2
    echo "    Look at: nvidia-smi   and   cat ${RESULTS_DIR}/${variant}/server.pid" >&2
    exit 1
  fi
  current=""
done

echo
uv run python -m camalerts.report --results-dir "$RESULTS_DIR"
echo "Report: ${RESULTS_DIR}/REPORT.md"
if [ -n "$failed" ]; then
  echo "Finished with failures:${failed}" >&2
  exit 1
fi
echo "All variants finished."
