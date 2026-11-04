# Circuit execution validation — 2026-10-28

The native semantic circuit path executes **10 of 13 current corpus apps**, up
from six on the prior opcode path, under the same default encryption profile
and accuracy criterion. All six shared apps have lower median evaluation time
in both final paired rounds. This comparison is against this project's prior
native backend, not OpenFHE or another library.

## Measured evaluation time

Times below are milliseconds, medians from the second final round. The speedup
column spans both final rounds (11 timed repetitions each, after three warmups).
A ratio above one means faster. Individual min/median/max values, setup times,
accuracy, and memory figures for both rounds are retained in the
[machine-readable results](../benchmarks/results/2026-10-28-circuit-release.json).

| Application | Prior native ms | Compiled ms | Speedup across two rounds |
| --- | ---: | ---: | ---: |
| matvec | 373.309 | 256.543 | 1.46× |
| matvec-fused | 125.923 | 121.021 | 1.04× |
| matmul-tile | 304.707 | 251.475 | 1.20–1.21× |
| reduce-tree | 116.165 | 107.282 | 1.08× |
| weighted-sum | 26.925 | 5.530 | 4.74–4.87× |
| diamond | 3.721 | 0.250 | 14.74–14.89× |

Newly executable apps have no successful baseline timing and therefore no
speedup ratio:

| Application | Compiled median ms | Semantic nodes → regions | Internal slot bytes |
| --- | ---: | ---: | ---: |
| conv1d | 43.599 | 39 → 12 | 1,966,080 |
| iter-update | 0.254 | 32 → 1 | 393,216 |
| stats | 114.765 | 56 → 30 | 2,228,224 |
| wide-front | 55.099 | 71 → 1 | 262,144 |

`horner15` and `logreg` still exhaust the default modulus chain; `psi-eq`
requires exact Boolean operations. These remain explicit unsupported results.
The user-owned ResNet example currently evaluates plaintext vectors/ReLU and is
not included as encrypted evidence.

## What changed

- The separate `FhnCircuit` API expresses values, while the native compiler
  chooses arithmetic regions, level transitions, key requirements and storage.
  Existing physical opcode meanings are preserved for reference execution.
- Integer affine expressions preserve levels and combine before evaluation.
  Weighted sums, the repeated update, and the diamond collapse to one region.
  Wide-front accumulates 24 ciphertext products and its linear terms before one
  relinearization/rescale.
- Native multiplication performs one exact combined $Pq$ reduction. The exact
  residues match the old separate tensor/relinearize/rescale calculation with
  identical inputs and keys, including rounding boundaries and multi-prime $P$.
- Source digit NTT values are reused. Rotations calculate one permutation map
  per call and reuse it across components/limbs. The first measured version
  repeated that indexing and regressed two apps by about 4–5%; the final code
  corrects that regression. Pre-fix timings are not mixed into the final table.
- Executables copy the graph, bind immutable public-key references, reuse
  liveness-sized slots, rebind encrypted inputs, and allocate no heap during
  run. All public outputs are published only after evaluation succeeds.

## Measurement conditions and interpretation

Measured on macOS 26.6.2 arm64 with Apple Clang 21.0.0, Release `-O3 -DNDEBUG`,
LTO disabled. Both plugins use $N=8192$, $Q$ prime bit widths $(50,40,40)$,
$P$ prime bit width $60$, $\mathrm{dnum}=3$ and initial scale $2^{40}$.
Each run freshly encrypts the same application
inputs; encrypted matrix rows remain encrypted. All output slots are checked
before and after timing against plaintext execution of the original corpus
graph, independently of its imported circuit. This oracle is not a separate
specification of every application.

The acceptance rule is

$$
\lvert \mathrm{actual} - \mathrm{expected} \rvert \le 10^{-4} + 10^{-6}\lvert \mathrm{expected} \rvert;
$$

nonfinite results fail. Absolute errors can exceed $10^{-4}$ for large outputs, as
the recorded per-app maximum errors show. The criterion was not relaxed for
compiled execution.

The baseline shared library was copied before this work. Its SHA-256 is
`f5548384ad12aa234af3815f6ce45ea09ec2f1f1b61d4d58940e1c22cad10da5`.
The final candidate hash is
`fd7b253e642d8320924817cc836cf9dc6e0d049ee89fa6d863206ce7419536e6`.
Both were local builds, not published release tags, and neither binary is kept
in the repository. Loading
both simultaneously was checked: only the candidate exports the circuit API.

Evaluation timing excludes context/client creation, encryption, compilation,
evaluation-key preparation and verification decryption; those costs are
reported separately in the data. The baseline interpreter uses preallocated
SSA buffers. Execution order alternates per repetition, and no agent build,
test or other benchmark ran during measurement. Two rounds show repeatability
on this host, not a statistical guarantee or a cross-device claim.

Every compiled instance reserves 2,491,200 bytes of scratch in this profile,
plus the reported internal slot bytes. The rotation map uses 32 KiB inside that
existing reserve. Public I/O buffers, the existing backend context workspace,
keys, tables, graph metadata and allocator overhead are excluded. Baseline
`baseline_all_buffer_bytes` includes all its public SSA buffers; it is not
directly comparable to internal slots alone. No process-memory bound is claimed.

## Verification and review

After the last rotation change:

| Check | Result |
| --- | --- |
| Whole Debug build + CTest | 33/33 targets; 328 GoogleTests across 27 executables, no failures. |
| Whole ASan + UBSan build + CTest | Same 33/33 targets and 328 GoogleTests passed; halt-on-error enabled. LeakSanitizer is unsupported on this platform and was disabled. |
| Release with BUILD_TEST=OFF, LTO disabled | Build passed. |
| Installed package consumers | Original frontend plus native raw square and compiled scaled-square consumers passed in Debug, sanitizer and Release installation checks. |
| Final Release app runs | Both final paired rounds: executed=10, unsupported=3, failed=0. |
| Public header / formatting | C11 header syntax with warnings-as-errors, changed C++ formatting, and whitespace checks passed. |

The tests cover exact arithmetic, imported semantics, real encrypted outputs,
rebound and aliased buffers, source mutation, key expiry, allocation-failure
retry, malformed graphs, constant/scale bounds, cancelled incompatible packing,
noheap execution, and exact reserved-byte budget boundaries.

Independent Astra reviews covered the arithmetic and compiler/integration.
They identified and verified fixes for alignment bytes omitted from workspace
accounting and invalid packing hidden by affine cancellation. The final scoped
rotation-map review found no additional actionable issue. These reviews and
tests are engineering evidence, not a cryptographic security certification.

The native path remains CPU leveled CKKS. The existing `Fhenon`/`Session`
frontend still defaults to ToyFHE; this implementation is explicitly accessed
through the circuit API and `fhn-bench-circuit`. Bootstrapping, exact Boolean
operations, GPU execution and serialized cross-process transport are not added.

## Reproduce

```sh
build/bin/fhn-bench-circuit --backend build/lib/libckks_fhn.dylib --prefix ckks_ \
  --baseline /path/to/preserved-native-plugin --warmup 3 --reps 11
```

Use `.so` on Linux. Omit the baseline argument for correctness/compiled timing.
Use `--shape NAME --require-execution` to require a particular app. See
[circuit API and runner](circuit-execution.md) for lifecycle, options and limits.
The full logs and reviews were local to the validating host and are not kept in
the repository.
