#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible Cheddar-FHE GPU build for the DGX Spark.
#
#   Target machine : NVIDIA GB10 (Grace-Blackwell), aarch64, sm_121
#   Toolchain      : CUDA 13.0, GCC 13, CMake 3.28
#
# Cheddar upstream assumes CUDA >= 11.8 and hardcodes an old SM list
# (60 61 70 75 80 86 89 90). Two things break that on this box:
#   1. CUDA 13 removed sm_60/61/70, so the hardcoded list won't even compile.
#   2. GB10 is sm_121, which is not in the list at all.
# So we patch the submodule's CMakeLists to target sm_121 only.
#
# Cheddar also needs libtommath (no apt/sudo here, so we build it from source,
# static + -fPIC so it links into the cheddar .so) and its RMM dependency pulls
# in spdlog headers, which the fhenomenon-side build expects at /tmp/spdlog.
#
# This script is idempotent: re-running skips work that is already done unless
# FORCE=1 is set. Everything machine-local lands in .gpu-deps/ (gitignored).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHEDDAR_DIR="$REPO_ROOT/refs/cheddar-fhe"
CHEDDAR_BUILD="$CHEDDAR_DIR/build"
DEPS_PREFIX="$REPO_ROOT/.gpu-deps"
TOMMATH_PREFIX="$DEPS_PREFIX/tommath"
SPDLOG_DIR="/tmp/spdlog"               # fhenomenon test CMake hardcodes this path
CUDA_ARCH="${CUDA_ARCH:-121}"          # GB10 Blackwell
JOBS="${JOBS:-$(nproc)}"
FORCE="${FORCE:-0}"

log() { printf '\n\033[1;36m[build-cheddar]\033[0m %s\n' "$*"; }

mkdir -p "$DEPS_PREFIX"

# ---------------------------------------------------------------------------
# 1. libtommath (static, -fPIC) — cheddar links it into libcheddar.so
# ---------------------------------------------------------------------------
if [[ "$FORCE" == "1" || ! -f "$TOMMATH_PREFIX/lib/libtommath.a" ]]; then
  log "Building libtommath from source -> $TOMMATH_PREFIX"
  rm -rf "$DEPS_PREFIX/libtommath-src"
  git clone --depth 1 --branch v1.3.0 https://github.com/libtom/libtommath \
    "$DEPS_PREFIX/libtommath-src"
  make -C "$DEPS_PREFIX/libtommath-src" -j"$JOBS" \
    CFLAGS="-fPIC -O2" libtommath.a
  mkdir -p "$TOMMATH_PREFIX/include" "$TOMMATH_PREFIX/lib"
  cp "$DEPS_PREFIX/libtommath-src/tommath.h" "$TOMMATH_PREFIX/include/"
  cp "$DEPS_PREFIX/libtommath-src/libtommath.a" "$TOMMATH_PREFIX/lib/"
else
  log "libtommath already built ($TOMMATH_PREFIX/lib/libtommath.a) — skipping"
fi

# ---------------------------------------------------------------------------
# 2. spdlog headers at /tmp/spdlog (RMM pulls these in transitively; the
#    fhenomenon-side cheddar_fhn target includes them from /tmp/spdlog/include)
# ---------------------------------------------------------------------------
if [[ "$FORCE" == "1" || ! -f "$SPDLOG_DIR/include/spdlog/spdlog.h" ]]; then
  log "Cloning spdlog headers -> $SPDLOG_DIR"
  rm -rf "$SPDLOG_DIR"
  git clone --depth 1 --branch v1.11.0 https://github.com/gabime/spdlog "$SPDLOG_DIR"
else
  log "spdlog headers already present ($SPDLOG_DIR) — skipping"
fi

# ---------------------------------------------------------------------------
# 3. Patch cheddar's hardcoded CUDA arch list -> sm_121 (idempotent)
# ---------------------------------------------------------------------------
CML="$CHEDDAR_DIR/CMakeLists.txt"
if grep -qE 'CMAKE_CUDA_ARCHITECTURES 60 61 70' "$CML"; then
  log "Patching CMAKE_CUDA_ARCHITECTURES -> $CUDA_ARCH in $CML"
  sed -i "s/set (CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90)/set (CMAKE_CUDA_ARCHITECTURES $CUDA_ARCH)/" "$CML"
else
  log "cheddar CMAKE_CUDA_ARCHITECTURES already patched — skipping"
fi

# ---------------------------------------------------------------------------
# 4. Configure + build cheddar
# ---------------------------------------------------------------------------
if [[ "$FORCE" == "1" ]]; then
  rm -rf "$CHEDDAR_BUILD"
fi

log "Configuring cheddar (arch=$CUDA_ARCH, jobs=$JOBS)"
cmake -S "$CHEDDAR_DIR" -B "$CHEDDAR_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_UNITTEST=OFF \
  -DENABLE_EXTENSION=ON \
  -DUSE_GMP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_PREFIX_PATH="$TOMMATH_PREFIX" \
  -DCMAKE_LIBRARY_PATH="$TOMMATH_PREFIX/lib" \
  -DCMAKE_CXX_FLAGS="-I$TOMMATH_PREFIX/include" \
  -DCMAKE_CUDA_FLAGS="-I$TOMMATH_PREFIX/include"

log "Building libcheddar.so"
cmake --build "$CHEDDAR_BUILD" --target cheddar -j"$JOBS"

log "DONE — libcheddar.so at: $CHEDDAR_BUILD/libcheddar.so"
ls -la "$CHEDDAR_BUILD/libcheddar.so"
