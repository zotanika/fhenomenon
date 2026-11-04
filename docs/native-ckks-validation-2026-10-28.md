# Native CKKS implementation validation — 2026-10-28

This records the initial opcode backend. The subsequent semantic circuit path,
application coverage and performance measurements are documented in
[circuit execution validation](circuit-execution-validation-2026-10-28.md).

The genuine CPU leveled RNS-CKKS path is implemented and was checked before it
was committed. This is correctness and integration evidence, not independent
cryptographic certification.

## Final results

| Check | Result |
| --- | --- |
| Whole Debug build and CTest | 29/29 test targets passed; 293 GoogleTest cases across 24 executables, plus corpus/install checks. |
| Whole ASan + UBSan build, no LTO | 29/29 targets passed with halt-on-error; no diagnostics. |
| Release, BUILD_TEST=OFF, no LTO | Built all production targets including ckks_fhn and fhn-corpus. |
| Installed package | Both existing runtime consumer and genuine fractional encrypted-square consumer passed using installed headers/libraries. |
| Packed $8\times8$ matrix/vector | Every row/output slot verified; native fused and full primitive fallback agree. Debug, sanitizer and Release checks passed. |
| Additional native workloads | matvec (64 slots), matmul-tile (two scalar-multiply levels), reduce-tree, weighted-sum and diamond all required-execution checks passed. |
| Layer isolation | L4 cannot compile a Keys.h include; CMake rejects an upward dependency. |
| Formatting / diff whitespace | New native C++ formatting and git diff --check passed. |

Environment: macOS-26.6.2-arm64-arm-64bit-Mach-O; Apple clang version 21.0.0 (clang-2100.3.34.2).
Only this macOS host was executed locally. Linux is configured in CI, not
reported as run here. The macOS runtime does not support LeakSanitizer, so
ASan runs explicitly used detect_leaks=0; no leak-check claim is made.

The sanitizer run used address/undefined instrumentation on the whole build,
DISABLE_LTO=ON, UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 and
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1. Release install validation was
separate from the Debug and sanitizer installed-package checks.

## Independent Astra review

Three Astra workers implemented independent storage/CRT, evaluation and host
crypto layers, then reviewed other layers and the adapter. The reviews found
two actionable issues, both reproduced before correction and independently
rechecked afterward:

1. Encoding checked int64 overflow but not the centered active-$Q$ interval.
   Small/custom or lower-level bases could wrap a finite representable integer
   into another value. Exact checked-$Q$ bounds now reject both signs outside
   $\lfloor Q/2\rfloor$ before modifying output; one/two-limb boundaries and $Q$ products
   larger than uint64 are covered. Final host-crypto tests: 17/17.
2. Rotation-key preparation could retain a client-owned key but fail while
   allocating its evaluator observer. Retrying previously reported success
   without registering the observer. Preparation now reuses the key and always
   restores the observer. All 12 allocation failure sites were swept; every
   retry executes and decrypts the expected rotation. Final backend tests: 5/5.

No remaining actionable findings were identified in the reviewed scope.
Independent evidence also includes 2,007 large-integer CRT comparisons,
61,440 exact rescale coefficient comparisons, malformed-input transaction
checks, and delayed entropy-failure probes. Tests cover random/public-only
RLWE encryption, wrong-key mismatch, actual two-level multiplication,
positive/negative/identity/conjugate rotations, multi-$Q$/multi-$P$ hybrid digits,
secret wiping, key expiry, cross-context rejection, aliases, and zero heap
allocation during successful evaluation kernels. L4 tests: 15/15; storage:
6/6; CRT: 8/8.

## Footprint and scope

The default backend reserves 786432 bytes per ciphertext slab
and 3474240 bytes for its compute workspace. The latter
is a conservative bound, not observed peak use or an exact minimum. Buffer
metadata separately reports live payload and high water. Key, table and client
workspace allocations are separate; these numbers are not process RSS.

Native ciphertexts are computed without a plaintext mirror or secret-assisted
evaluation. FhnBackendCtx has no secret/client reference; the host client owns
keys and the evaluator borrows weak observers of evaluation keys. Preparation
occurs before public compute dispatch. No same-context native encrypt/decrypt
exports exist; ABI1 and the existing vtable layout remain unchanged.

The implementation is leveled CPU CKKS with two multiplication levels in the
default profile. Bootstrap, GPU, cross-process transport, high-level C++
frontend client integration, adversarial decryption hardening, side-channel
review and performance comparisons remain separate work. Fixed-capacity
buffers and conservative scratch bounds must not be described as exact total
program-memory accounting. The CPU movement model has no physical eviction.

## Evidence and handoff

The build directories, logs and review reports behind these results were local
to the validating host and are not kept in the repository. The commands in
[native usage](native-ckks.md) reproduce the checks.
