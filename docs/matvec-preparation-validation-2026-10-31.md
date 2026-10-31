# Matvec preparation validation — 2026-10-31

The unchanged encrypted 8×8 `matvec-fused` application takes **80.532 ms**
with immediate commands, **79.444 ms** with the previous contributed
packed-dot kernel, and **39.444 ms** with the new shared preparation.
That is **2.04× speedup / 51.02% less execution time** versus immediate
execution, or 50.35% less time versus the previous coarse kernel, in this one-host run.
The formula, eight outputs, levels, scales, key-switch/rescale rounding, Runtime
and NativeRuntime are unchanged.

## Why the original coarse kernel barely helped

The [baseline phase diagnostic](../benchmarks/results/2026-10-31-matvec-baseline-profile.json)
measured eight coarse dots at 74.343 ms versus 74.748 ms for eight ordinary
native HMULT plus 24 HROT_ADD calls. The instrumented arithmetic mirror matched
the public path exactly, including every live residue, layout and scale.

| Diagnostic phase, all eight rows | Median ms |
|---|---:|
| Tensor product, relinearization and rescale | 31.300 |
| Rotation/add distance 4 | 13.592 |
| Rotation/add distance 2 | 13.616 |
| Rotation/add distance 1 | 13.569 |
| Inputs and four keys validated per row | 1.709 |
| Native output copy | 0.028 |

About 98% of the instrumented outer median was spent in the arithmetic regions.
Their work survived the coarse boundary unchanged. Separately, eight checked
ADD_CS(0) publications took 0.175 ms and 32 buffer allocate/free pairs took
0.002 ms. These are diagnostic probes with different cache/access patterns,
not components that may be subtracted from an application total. Phase medians
are calculated separately and need not sum exactly.

The large repeated expense was general modular multiplication by immutable
key coefficients, as well as by the same source coefficients across rows.
An isolated exact-residue key-coefficient probe fell from 1.110 ms to 0.086 ms
using precomputed Shoup companions. This microprobe excluded preparation and
was only a candidate-selection experiment, not the application speedup claim.

## Complete application measurement

[Raw Release samples](../benchmarks/results/2026-10-31-matvec-preparation-release.json)
contain 24 measured executions and two warmups per mode. Measurement order
rotates, covering two complete cycles through the 12 modes. All modes use one
fixture's identical ciphertexts, context and keys. Every execution verifies all
8 outputs × 8 logical slots against the unchanged independent oracle; maximum
absolute error is 2.91260545e-07, identical across modes and below the unchanged
`1e-4 + 1e-6*abs(expected)` tolerance. Private/native differential tests also
require exact ciphertext residues, rather than tolerance-only equivalence.

Environment: macOS arm64, AppleClang 21.0.0, Release, LTO disabled,
`BUILD_TEST=OFF`. Parameters: N8192, Q50/40/40, P60, dnum3, scale 2^40;
8 logical slots repeated over the native physical packing. Input vector is
`[1,-2,3,0,-1,2,1,-3]`; matrix row j is `(j*3+k*2)%5-2`.
No other project builds/tests/timing jobs ran during this measurement.

| Mode | Warm median ms | P10–P90 ms | Preparation ms | Artifact MiB | Peak tracked MiB |
|---|---:|---:|---:|---:|---:|
| `immediate` | 80.532 | 80.183–81.828 | 0.000 | 0.00000 | 14.25000 |
| `prepared` | 80.627 | 80.041–81.058 | 0.003 | 0.00000 | 14.25000 |
| `region` | 77.592 | 77.079–78.069 | 0.367 | 0.00000 | 44.75000 |
| `contributed` | 79.444 | 79.108–80.116 | 0.000 | 0.00000 | 13.50000 |
| `split` | 80.349 | 79.876–81.082 | 0.000 | 0.00000 | 15.75000 |
| `matvec-bound` | 77.909 | 77.381–78.540 | 0.243 | 0.37500 | 13.87500 |
| `matvec-maps` | 76.003 | 75.703–76.656 | 0.487 | 0.46875 | 13.96875 |
| `matvec-keys` | 44.837 | 44.533–45.293 | 5.918 | 6.37500 | 19.87500 |
| `matvec-vector` | 74.251 | 73.738–75.259 | 0.616 | 0.75000 | 14.25000 |
| `matvec-full` | 39.444 | 39.122–40.170 | 6.448 | 6.84375 | 20.34375 |
| `matvec-split` | 47.999 | 47.719–48.402 | 6.647 | 6.84375 | 22.59375 |
| `matvec-ordinary-product` | 56.180 | 55.834–56.800 | 6.155 | 6.84375 | 21.09375 |

- `immediate`: the original 56 mathematical commands.
- `prepared`: the earlier rotation-source preparation strategy; this tree has
  no input reused by multiple rotations, so it creates no artifact.
- `region`: the existing explicitly requested compiler reference. Its 32 MiB
  reservation is included in peak tracked memory, not the artifact column.
- `contributed` and `split`: previous packed-dot contribution and its ordinary
  product/first-rotation boundary.
- `matvec-bound`: source snapshot plus once-validated weak key bindings, with
  original generic arithmetic and maps.
- `matvec-maps` / `matvec-vector`: add only cached maps or source companions.
- `matvec-keys`: retained evaluation-key companions **plus per-call fixed-scalar
  Shoup hoisting** for P-lift/mod-down/rescale. It does not isolate retained-key
  state from the scalar arithmetic improvement.
- `matvec-full`: all selected components; one artifact is shared by eight
  independent dot requests.
- `matvec-split`: prepared product → ordinary rotate/add → prepared partial
  reduction, sharing the same artifact across those calls.
- `matvec-ordinary-product`: ordinary product → prepared complete reduction.
  This control uses ALL preparation, including unused vector companions, so
  its setup and memory include that conservative extra cost.

Against `matvec-bound`, maps alone reduce the median by
1.907 ms (2.45%) and source companions alone by
3.658 ms (4.70%). These isolate retained reusable state beyond binding/validation.
The key-plus-scalar combination provides the largest reduction. Component
gains need not add; this experiment does not attribute all of that combination's
gain to retained keys alone. Avoiding output copies alone cannot explain the
measured improvement, and final publication still performs the original copy.

Intermediate interoperability preserves real gains: the mixed prepared/ordinary
path takes 47.999 ms (40.40% less time than immediate), while using an entirely ordinary
product before prepared reduction takes 56.180 ms (30.24% less). A whole-app compiled
region or a single all-rows kernel is unnecessary for these results.

## Setup, memory and amortization

The complete preparation retains 7,176,192 bytes (6.84375 MiB):
6 MiB of evaluation-key companions, 384 KiB of source snapshot, 384 KiB of
source companions and 96 KiB of maps. It takes 6.448 ms to create in the
measured fixture. The full path's peak tracked memory is
20.34375 MiB versus 14.25000 MiB for immediate execution. Accounting includes
retained ciphertext/artifact slabs and contribution staging plus publication;
it excludes object/allocator metadata, context scratch/tables and existing keys.
Companions are currently owned per source preparation, with no deduplication
across separate prepared sources.
Tracked memory is per-runtime accounting, not process RSS. All comparison Run
objects coexist in the benchmark process, each with its own retained inputs
and selected artifact; shared backend keys/tables remain outside these budgets.

Using the observed warm medians, preparation plus one complete eight-row app
is **45.892 ms**, below the old coarse kernel's 79.444 ms.
The estimated break-even is
`ceil(preparation / (old_coarse - prepared)) = 1` complete application. This is
an amortization estimate from separately measured setup and warm execution,
**not a measurement of cold first-request/service latency**. Registration,
initial input import, image load, graph import, context creation, key generation
and encryption are separately reported and excluded from these execution figures.

Warm timing includes dispatch, allocation, evaluation, native staging,
`importCopy` publication and temporary release. Decryption, initial import,
setup and final output destruction are outside it for every mode. No compiler
entrypoint is available to preparation/contribution/immediate modes.

All samples are retained, including outliers: full preparation execution ranges
39.095–67.242 ms and vector-only ranges
73.503–88.733 ms. P10/P90 are inclusive interpolated
sample quantiles, not confidence intervals. No statistical-significance claim
is made for the small coarse/binding/map differences. Different dates/runs
have different absolute baselines; comparisons above are within this one run.
These results do not establish performance on larger matrices, other parameter
sets, other CPUs, batching policies or distributed services.

## Verification

- Full Debug: **43/43 CTest targets passed**, including **410 GoogleTest cases
  across 35 binaries**.
- Full ASan+UBSan: **43/43 passed**. Halt-on-error and ODR checks enabled;
  LeakSanitizer disabled because unsupported by this macOS host.
- New coverage: 5 private arithmetic tests, 11 native preparation tests and
  9 contribution tests. Exact differential matrices cover levels, dnum1/2/4,
  all eight flags, multi-P and 62-bit arithmetic; signed/identity rotations;
  ordinary and prepared intermediate boundaries; source snapshot/identity;
  weak key expiry/replacement; budgets, no-allocation consumption, rollback,
  aliases, capacity and scratch failures; plugin compatibility and lifetimes.
- Full Release build with tests disabled and installed-package consumer passed.
  The consumer links `Fhenomenon::MatvecKernels` using installed public headers
  and exercises both full and split dots.
- Public C header passes C11 `-Werror` syntax validation.
- SHA-256 checks confirm the four Runtime/NativeRuntime files and corpus
  shapes/oracle are byte-identical to their pre-task versions.
- Independent Astra review found and verified fixes for width-one zero-stage
  accounting, non-finite scale admission and mismatched plugin services.
  No remaining implementation blocker was reported.

No Git commit/push was made. Existing unrelated workspace edits were preserved.

## Reproduce

With the repository's normal dependencies/configuration available:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DDISABLE_LTO=ON
cmake --build build-release -j 3
build-release/bin/fhn-matvec-profile > baseline-profile.json
build-release/bin/fhn-bench-runtime build-release/lib/libckks_fhn.dylib 24 matvec-reuse > reuse-release.json
```

Use the platform's shared-library filename on other operating systems.
Implementation and API usage: [prepared matvec](matvec-preparation.md).
