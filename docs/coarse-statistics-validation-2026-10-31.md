# Coarse contribution validation — 2026-10-31

The [contribution](coarse-statistics-kernels.md) adds centered statistics,
packed dot and partial reduction kernels. Runtime and NativeRuntime headers
and implementations are byte-for-byte unchanged; the original SHA-256 values
were checked after implementation and are preserved in the
[raw result JSON](../benchmarks/results/2026-10-31-coarse-statistics-release.json).

## Performance result

Same encrypted inputs and keys, default N8192 Q50/40/40 P60 dnum3 profile,
AppleClang 21.0.0 on macOS arm64, Release with LTO/tests disabled. Two warmups
and seven measured runs per mode; measurement order rotates. All other agent
build/test/timing work was idle.

Median warm execution milliseconds:

| Path | `stats` | `matvec-fused` |
|---|---:|---:|
| Original immediate commands | 73.593 | 77.252 |
| Contributed energy, entirely ordinary reduction | 65.048 | — |
| Full contributed variance / packed-dot kernel | 64.986 | 76.473 |
| Contribution mixed with ordinary intermediate commands | 64.811 | 77.332 |
| Existing explicit compiled-region reference | 70.714 | 74.519 |

The statistics contribution reduces median execution latency by **11.7%**
against immediate execution and **8.1%** against the existing compiled reference.
Immediate samples span 73.117–74.977ms and contributed samples 64.449–66.063ms.
The split path retains the improvement when intermediate values cross back
through ordinary commands. Its 0.18ms difference from the full kernel is too
small to claim an advantage for that split.

The energy-only comparison is essential: the full kernel is just 0.095% faster
than the same centered energy followed by ordinary reduction. These samples do
**not** demonstrate an additional material benefit from grouping the remaining
reduction. The demonstrated gain comes primarily from the contributed
application-specific formulation, which eliminates one tensor product and one
relinearization/rescale per dataset. Keeping the intermediate state available
does not erase that gain.

Packed-dot aggregation in `matvec-fused` changes the median by only 1.0%, with
overlapping sample ranges; no robust performance improvement is established for
that case. Its existing compiled path remains faster. A larger kernel is not
automatically the best choice for every app.

These are comparisons between project paths on one host, not external FHE
library results or an end-to-end service-latency claim.

## Timing and memory scope

Timing includes complete ordinary sum construction on every stats run, recipe
dispatch, native staging allocation, arithmetic, result-publication copy and
temporary release. Initial input imports, provider registration, context/keys,
encryption/decryption and final output destruction are outside warm timing.
All modes register the same providers; compiler services are absent in every
non-region mode.

Setup is reported separately in JSON. In this run the statistics contribution
registration took 0.063ms, its two input imports 0.079ms, and shared
context/key/encryption setup 38.140ms. The explicit region's preparation took
0.396ms. The benchmark session's initial plugin-image load took 333.867ms once;
it is not included in the per-application warm results. The contribution has no
hidden one-time compilation or precomputed dataset sum.

Stats peak tracked reservations are 6.00MiB immediate, 5.25MiB full contribution,
7.50MiB split and 6.75MiB energy-only. Staging and publication buffers are both
charged during the copy. The split path trades additional live ciphertexts for
explicit intermediate access. These figures exclude keys, context scratch and
tables, metadata and allocator overhead; they are not process RSS. The compiled
reference reserves a 32 MiB cap, so its 36.50 MiB tracked peak cannot be interpreted
as actual used workspace.

## Correctness and boundaries

- The original corpus graphs, inputs, output IDs and independent plaintext
  oracle were not changed. All four stats outputs and eight matrix/vector
  outputs pass every logical slot check on every warmup and measured run.
- The original tolerance remains `1e-4 + 1e-6*abs(expected)`. Maximum stats
  absolute error was 0.000182663 on the original path and 0.000184993 on the
  contribution/split/energy-only paths; both satisfy the mixed tolerance.
- A sum made by ordinary commands is also an output and is passed into the
  contributed statistics kernel. Energy output returns to ordinary rotate/add
  commands and then to a contributed partial reduction. Matrix row kernels all
  share the original encrypted vector handle.
- Native tests check all 4096 physical slots for repeated corpus/nearconstant
  data; separate constant regressions check all 64 logical slots.
- Products/outputs use real native CKKS, with no decryption, plaintext fallback
  or client pointer in compute providers. Native coarse calls allocate no heap
  memory. They use existing context scratch and publish output only on success.

The initial reassociation `reduce(x*(n*x-s))` was rejected: on constant −8 data
it produced about 1.66e-4 error versus 1.92e-5 for the original expression, failing
the unchanged 1e-4 absolute bound. The stable `(n*x-s)^2/n` form passed the
retained regression, with an observed maximum 3.8e-10 in a focused run.

Its division is represented by actual output scale `n*input_scale^2/q_drop`.
It needs extra modulus headroom and is not interchangeable with an equal-level
ordinary product's scale. Statistics recipes require level >= 2; the tested
default-profile width 64 envelope has |x|<=8 and a correctly supplied period-64
sum. Other parameter/magnitude choices need their own accuracy validation.
Ordinary operations preserve this scale; incompatible joins are explicitly
rejected. No tolerance or value metadata was falsified to obtain a speedup.

## Verification and review

- Full Debug: **39/39 CTest targets passed**.
- Full ASan+UBSan: **39/39 passed**, halt-on-error enabled; 385 GoogleTest cases
  across 32 binaries. LeakSanitizer is unavailable on this host and was disabled;
  ODR checks remained enabled.
- After adding the energy-only measurement mode, its containing corpus target
  passed again in Debug and ASan+UBSan.
- New native suite: 8 tests. New public contribution suite: 14 tests, including
  metadata/width/level rejection, actual-scale interoperability, key expiry,
  no retry after execution failure, capability fallback, owned-context lifetime
  and counted allocation failures before staging/publication.
- Release/test-off builds and installed public consumers pass. The installed
  `Fhenomenon::StatsKernels` target exercises both shared-sum and split paths.
- The public coarse header passes a C11 syntax check.
- Independent Astra review found no blocking execution defect. Review tightened
  setup accounting, periodic-input wording and physical-slot coverage wording;
  those corrections are included. The reviewer independently checked medians,
  sample ranges, memory scope and the bounded performance conclusions above.
