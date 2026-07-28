#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Emit a one-line JSON fingerprint of the machine + toolchain + source state.
# Consumed by run.sh to stamp every experiment record. Safe to run anywhere;
# missing tools degrade to null rather than erroring.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

jstr() { # json-escape a string, or emit null for empty
  local s="${1:-}"
  if [[ -z "$s" ]]; then printf 'null'; return; fi
  s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; s="${s//$'\n'/ }"
  printf '"%s"' "$s"
}

host="$(hostname 2>/dev/null)"
kernel="$(uname -r 2>/dev/null)"
arch="$(uname -m 2>/dev/null)"
cpus="$(nproc 2>/dev/null)"
mem_gib="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null)"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
gpu_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
cuda_ver="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
gcc_ver="$(gcc -dumpfullversion 2>/dev/null)"
cmake_ver="$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"

fhn_sha="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)"
fhn_dirty="$(if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi)"
cheddar_sha="$(git -C "$REPO_ROOT/refs/cheddar-fhe" rev-parse --short HEAD 2>/dev/null)"
cheddar_dirty="$(if [[ -n "$(git -C "$REPO_ROOT/refs/cheddar-fhe" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi)"

printf '{'
printf '"host":%s,'         "$(jstr "$host")"
printf '"arch":%s,'         "$(jstr "$arch")"
printf '"kernel":%s,'       "$(jstr "$kernel")"
printf '"cpus":%s,'         "${cpus:-null}"
printf '"mem_gib":%s,'      "${mem_gib:-null}"
printf '"gpu":%s,'          "$(jstr "$gpu_name")"
printf '"gpu_cc":%s,'       "$(jstr "$gpu_cc")"
printf '"driver":%s,'       "$(jstr "$driver")"
printf '"cuda":%s,'         "$(jstr "$cuda_ver")"
printf '"gcc":%s,'          "$(jstr "$gcc_ver")"
printf '"cmake":%s,'        "$(jstr "$cmake_ver")"
printf '"fhn_sha":%s,'      "$(jstr "$fhn_sha")"
printf '"fhn_dirty":%s,'    "${fhn_dirty:-null}"
printf '"cheddar_sha":%s,'  "$(jstr "$cheddar_sha")"
printf '"cheddar_dirty":%s' "${cheddar_dirty:-null}"
printf '}\n'
