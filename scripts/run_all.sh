#!/usr/bin/env bash
# Run the four benchmarks against a running server, then rebuild the report.
#
#   scripts/run_all.sh <variant_id> [base_url]
#
# Optional environment: MANIFEST (data/manifest.jsonl), RESULTS_DIR (results),
# CONFIG (configs/bench.yaml), ALERTS (configs/alerts.yaml), N (frames per benchmark),
# TEXT_ONLY=1 (send a written scene in place of each frame, for servers with no vision),
# BENCHES (space-separated module names, to run only some of the four).
# A failed benchmark does not stop the rest, but the script then exits non-zero.
set -euo pipefail
# `uv run` turns Ctrl-C into an ordinary non-zero exit, so without this the loop would carry on.
trap 'exit 130' INT TERM

VARIANT="${1:?usage: scripts/run_all.sh <variant_id> [base_url]}"
BASE_URL="${2:-http://localhost:30000}"
MANIFEST="${MANIFEST:-data/manifest.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CONFIG="${CONFIG:-configs/bench.yaml}"
ALERTS="${ALERTS:-configs/alerts.yaml}"

extra=()
[ "${TEXT_ONLY:-0}" = "1" ] && extra+=(--text-only)
[ -n "${N:-}" ] && extra+=(--n "$N")

cd "$(dirname "$0")/.."

failed=""
for bench in ${BENCHES:-bench_json bench_prefix bench_concurrency bench_image_size}; do
  echo "=== ${bench} (${VARIANT} at ${BASE_URL}) ==="
  if ! uv run python -m "camalerts.${bench}" \
      --base-url "$BASE_URL" --variant "$VARIANT" --manifest "$MANIFEST" \
      --results-dir "$RESULTS_DIR" --config "$CONFIG" --alerts "$ALERTS" \
      ${extra[@]+"${extra[@]}"}; then
    echo "!!! ${bench} failed" >&2
    failed="${failed} ${bench}"
  fi
done

echo "=== report ==="
if ! uv run python -m camalerts.report --results-dir "$RESULTS_DIR"; then
  failed="${failed} report"
fi

if [ -n "$failed" ]; then
  echo "FAILED:${failed}" >&2
  exit 1
fi
echo "All benchmarks finished for ${VARIANT}."
