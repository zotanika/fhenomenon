#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GPU experiment runner + tracker for the Cheddar backend on the DGX Spark.
#
# Runs one experiment, tees its full output under experiments/runs/<id>/, and
# appends a structured record to experiments/log.jsonl plus a human-readable
# entry to experiments/JOURNAL.md. Every record carries the env fingerprint
# (GPU, CUDA, driver) and the fhenomenon + cheddar git SHAs, so results stay
# attributable months later.
#
# Usage:
#   experiments/run.sh gpu-test              # ctest FhnCheddarGpuTest (correctness + latency)
#   experiments/run.sh corpus [args...]      # fhn-corpus on the cheddar GPU backend
#   experiments/run.sh raw <label> -- cmd... # any command; recorded verbatim
#
# Env:
#   NOTE="free text"   attach a note to this run's record
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
CHEDDAR_LIB="$BUILD_DIR/lib/libcheddar_fhn.so"
PARAM_DIR="$REPO_ROOT/refs/cheddar-fhe/parameters"
RUNS_DIR="$REPO_ROOT/experiments/runs"
LOG_JSONL="$REPO_ROOT/experiments/log.jsonl"
JOURNAL="$REPO_ROOT/experiments/JOURNAL.md"

# Deterministic run id from an injectable UTC stamp (STAMP=YYYYmmdd-HHMMSS),
# falling back to `date` when unset. Kept injectable so runs are reproducible.
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
ISO="${ISO:-$(date -u +%FT%TZ)}"
NOTE="${NOTE:-}"

kind="${1:-}"; shift || true
[[ -z "$kind" ]] && { echo "usage: run.sh <gpu-test|corpus|raw> [args]"; exit 2; }

run_id="${STAMP}-${kind}"
run_dir="$RUNS_DIR/$run_id"
mkdir -p "$run_dir"
out_log="$run_dir/output.log"

env_json="$(bash "$REPO_ROOT/experiments/capture-env.sh")"

# --- Resolve command + metric extractor per experiment kind -----------------
declare -a CMD
metrics_json='{}'

case "$kind" in
  gpu-test)
    # -V so per-test stdout (context+keygen, per-op latency) is captured even
    # when the tests pass; --output-on-failure would swallow it on success.
    CMD=(ctest --test-dir "$BUILD_DIR" -R FhnCheddarGpuTest -V)
    ;;
  corpus)
    [[ -f "$CHEDDAR_LIB" ]] || { echo "error: $CHEDDAR_LIB not built"; exit 1; }
    local_bin="$(find "$BUILD_DIR" -name fhn-corpus -type f 2>/dev/null | head -1)"
    [[ -n "$local_bin" ]] || { echo "error: fhn-corpus binary not found in $BUILD_DIR"; exit 1; }
    CMD=("$local_bin" --backend "$CHEDDAR_LIB" "$@")
    ;;
  raw)
    label="${1:-raw}"; shift || true
    [[ "${1:-}" == "--" ]] && shift || true
    run_id="${STAMP}-raw-${label}"
    run_dir="$RUNS_DIR/$run_id"; mkdir -p "$run_dir"; out_log="$run_dir/output.log"
    CMD=("$@")
    ;;
  *) echo "unknown kind: $kind"; exit 2 ;;
esac

# --- Execute, timing the wall clock -----------------------------------------
echo "[$run_id] $ISO" | tee "$out_log"
echo "cmd: ${CMD[*]}" | tee -a "$out_log"
echo "env: $env_json" | tee -a "$out_log"
echo "---" | tee -a "$out_log"

t0="$(cat /proc/uptime | awk '{print $1}')"
"${CMD[@]}" >>"$out_log" 2>&1
status=$?
t1="$(cat /proc/uptime | awk '{print $1}')"
wall_s="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')"

outcome="pass"; [[ $status -ne 0 ]] && outcome="fail"

# --- Extract a few headline metrics from the captured log -------------------
# Pull the number adjacent to the us/ms unit, so labels like ROTATE(+1) don't
# leak their own digits into the reading.
num() { grep -oE "$1" "$out_log" | grep -oE '[0-9]+ (us|ms)' | grep -oE '[0-9]+' | head -1; }
setup_ms="$(num 'Context \+ keys: [0-9]+ ms')"
addcc_us="$(num 'ADD_CC pipeline: [0-9]+ us')"
hmult_us="$(num 'HMULT pipeline: [0-9]+ us')"
rot_us="$(num 'ROTATE\(\+1\)\+ROTATE\(-1\) pipeline: [0-9]+ us')"
tests_passed="$(grep -oE '\[  PASSED  \] [0-9]+ tests?' "$out_log" | grep -oE '[0-9]+' | head -1)"
belady_savings="$(grep -oE 'median per-shape savings @B_mid: [0-9.]+%' "$out_log" | grep -oE '[0-9.]+' | head -1)"

metrics_json="$(printf '{"setup_ms":%s,"add_cc_us":%s,"hmult_us":%s,"rotate_pair_us":%s,"tests_passed":%s,"belady_savings_pct":%s}' \
  "${setup_ms:-null}" "${addcc_us:-null}" "${hmult_us:-null}" "${rot_us:-null}" "${tests_passed:-null}" "${belady_savings:-null}")"

# --- Append machine-readable record -----------------------------------------
rec="$(printf '{"id":"%s","ts":"%s","kind":"%s","cmd":"%s","outcome":"%s","exit":%s,"wall_s":%s,"metrics":%s,"env":%s,"note":"%s","log":"experiments/runs/%s/output.log"}' \
  "$run_id" "$ISO" "$kind" "${CMD[*]//\"/\\\"}" "$outcome" "$status" "$wall_s" "$metrics_json" "$env_json" "${NOTE//\"/\\\"}" "$run_id")"
printf '%s\n' "$rec" >> "$LOG_JSONL"

# --- Append human-readable journal entry ------------------------------------
{
  echo ""
  echo "### $ISO — \`$kind\` — **${outcome^^}** (${wall_s}s)"
  echo ""
  echo "- run id: \`$run_id\`  ·  fhn \`$(echo "$env_json" | grep -oE '"fhn_sha":"[^"]*"' | cut -d'"' -f4)\`  ·  cheddar \`$(echo "$env_json" | grep -oE '"cheddar_sha":"[^"]*"' | cut -d'"' -f4)\`"
  echo "- cmd: \`${CMD[*]}\`"
  [[ -n "$setup_ms" ]]       && echo "- context+keygen: ${setup_ms} ms (one-time)"
  [[ -n "$addcc_us" ]]       && echo "- ADD_CC: ${addcc_us} µs · HMULT: ${hmult_us:-?} µs · rotate-pair: ${rot_us:-?} µs"
  [[ -n "$tests_passed" ]]   && echo "- tests passed: ${tests_passed}"
  [[ -n "$belady_savings" ]] && echo "- Belady median per-shape savings: ${belady_savings}%"
  [[ -n "$NOTE" ]]           && echo "- note: $NOTE"
  echo "- log: [\`experiments/runs/$run_id/output.log\`](runs/$run_id/output.log)"
} >> "$JOURNAL"

echo "---"
echo "[$run_id] outcome=$outcome exit=$status wall=${wall_s}s"
echo "recorded -> experiments/log.jsonl and experiments/JOURNAL.md"
exit $status
