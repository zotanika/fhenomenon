# Lightweight runtime validation — 2026-10-31

Implemented entry points: `FHN/Runtime.h` and `FHN/NativeRuntime.h`.
The [runtime contract](lightweight-runtime.md) describes selection, lifetime,
accounting and the explicit compiler boundary. Existing `Session` and FHN
program paths remain separate; applications opt into this runtime API.

## Correctness and integration

- Debug: 37/37 CTest targets pass; 363 GoogleTest cases across 30 binaries.
- ASan + UBSan: 37/37 CTest targets pass with halt-on-error enabled, including
  installed-package tests. LeakSanitizer is unavailable on this host and was
  disabled; ODR checks remain enabled.
- Release with tests disabled and LTO disabled builds. A clean installed
  consumer passes both tests, including a separately compiled contribution
  using installed headers, two outputs sharing one rotation preparation, and
  an immediate result consumed by an explicitly compiled region.
- Core: 18 tests cover deterministic dispatch/declines, failure without retry,
  cross-runtime rejection, shared dependencies, resource lifetime past runtime
  destruction, pre-factory budget reservation/overflow/reentrancy, unpublished
  multi-output failure cleanup, provider cleanup lifetime and bounded trace.
- Native adapter: 11 tests cover compiler-free execution, stale preparation
  rejection, client-key expiry, surviving context leases, bounded explicit
  compilation, input rebinding, real scale checks, multi-output failure cleanup
  and restricted-provider fallback before execution.
- Prepared rotations: six tests include 108 exact differential cases against
  both previous rotation paths, all four tested levels with digit counts
  1/2/4, signed/extreme rotations, default-profile encrypted all-slot checking,
  immutable snapshots, context/key/scratch rejection and allocation failures.

The three benchmark workloads use the unchanged corpus graph and independent
oracle. Every output and logical slot is checked at warmup and on each measured
run with `abs(error) <= 1e-4 + 1e-6*abs(expected)`. `stats` has larger-magnitude
results; its measured maximum absolute error of 0.00021023 satisfies this mixed
tolerance. This is not a universal absolute-accuracy claim.

## Measured execution

macOS arm64, AppleClang 21.0.0, Release, no LTO, default native N8192 profile
(Q 50/40/40, P60, dnum3). Two warmups and seven measured repetitions per mode;
mode order rotates. No other agent build, test or timing ran concurrently.
Raw samples and accounting are preserved in the
[result JSON](../benchmarks/results/2026-10-31-lightweight-runtime-release.json).

Median milliseconds, including dispatch, allocation, arithmetic and temporary
release; excluding input import, context/key/encryption setup, preparation and
decryption:

| Existing workload | Immediate | Shared rotation preparation | Explicit compiled region |
|---|---:|---:|---:|
| `matvec` | 193.591 | 184.071 | 162.756 |
| `matvec-fused` | 79.267 | 79.390 (no reusable source) | 76.303 |
| `stats` | 75.535 | 75.569 (no reusable source) | 73.014 |

`matvec` is the existing 64-slot cyclic rotation/scalar workload. One prepared
source costs 0.148 ms and 589,824 bytes (576 KiB). Its separately submitted
rotations reuse the inverse transforms, reducing median evaluation by 9.520 ms
(4.9%, 1.052× throughput ratio in this run). Setup must still be added for
one-shot use. This is a modest measured improvement, not a claim of fully
hoisted key switching or a general speedup for unrelated workloads.

Explicit region preparation (compilation plus any additional trusted-host key
preparation) costs 2.925 ms, 0.357 ms and 0.412 ms respectively. Warm compiled
execution is 1.189×, 1.039× and 1.035× faster than this runtime's immediate path
in this sample. These compare project paths, not external FHE libraries.

Immediate peak tracked slab reservations are 4.5 MiB, 14.25 MiB and 6 MiB.
The prepared matvec peak is 5.0625 MiB. Compiled examples explicitly reserve a
32 MiB cap each; actual internal workspace is 23,724,864 / 4,850,496 /
4,719,424 bytes. Their reported peaks include the conservative full cap and
external ciphertexts. Keys, context scratch/tables, metadata and allocator
overhead are outside these figures. This benchmark does not choose the smallest
possible region reservation automatically.

## Review corrections and sanitizer investigation

Independent Astra review found two material issues; both are corrected and
covered by regressions:

1. A resource's dependencies were released before its payload destructor.
   Dependency storage now remains alive throughout provider cleanup.
2. A native recipe lacking an optional capability could be selected and fail
   before another capable recipe was considered. Capability and node-bound
   checks now decline before selection. Actual execution failures still do not
   trigger retries.

An initial full sanitizer run passed 36/37 tests; the older circuit benchmark
failed while unloading/reloading the same plugin between workload fixtures.
A separate three-line dylib with an unrelated virtual class reproduced the
same ASan duplicate-vtable registration, with no project code linked or crypto
functions called. Holding one outer image lease passed 20 reload iterations.
The precise toolchain defect was not established; the failure is independently
reproducible outside this repository.

Both benchmarks now retain their library image for the benchmark session while
recreating contexts/clients per workload. Production resource ownership and all
sanitizer checks are unchanged. The affected Debug tests and the complete strict
sanitizer suite passed after this correction.

## Scope still open

This delivers immediate recipe dispatch, retained intermediate state, a native
reuse example and explicit compiler regions. Automatic reuse discovery, global
optimization, asynchronous device execution and CGGI↔CKKS/BFV switching remain
outside this implementation. Recipe callbacks are trusted C++17 components;
an ABI-stable arbitrary-contribution C interface is not yet supplied.
