# Native CPU leveled RNS-CKKS

`ckks_fhn` performs genuine RLWE public-key encryption and homomorphic CKKS
arithmetic. Its evaluator operates on RNS polynomial ciphertexts and public
evaluation keys. It does not decrypt, retain a plaintext mirror, or
re-encrypt intermediate values. ToyFHE remains a separate insecure reference.

## Run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_DOCUMENTATION=OFF
cmake --build build
ctest --test-dir build --output-on-failure --no-tests=error
build/bin/fhn-corpus --backend build/lib/libckks_fhn.dylib --prefix ckks_ \
  --shape matvec-fused --require-execution --compare-fallback \
  --atol 1e-4 --rtol 1e-6 --budget-bytes min
```

Use `libckks_fhn.so` on Linux. This $8\times8$ workload encrypts both matrix rows and
vector, verifies every output slot against the oracle, and compares native
`HMULT`/`HROT_ADD` with complete primitive decomposition. Required execution
fails on a skip. `ckks_fhn` also builds and installs with `BUILD_TEST=OFF`.
The existing C++ `Fhenon`/`Session` frontend and `fhn-bench-matvec` still use
the ToyFHE default; use the native FHN client API or corpus command above.

## Compiled applications

The optional [circuit API](circuit-execution.md) accepts value graphs and chooses
larger execution regions. It preserves levels for exact integer affine work,
combines compatible sums of products before one key switch/rescale, and reuses
prepared storage. `fhn-bench-circuit` executes ten current corpus apps on the
same default profile and checks every output slot. The raw opcode contracts
below remain unchanged and are useful as the comparison path.

## What is implemented

| Layer | Implementation |
| --- | --- |
| L0 | Validated immutable parameter sets and prime chains. |
| L1 | Negacyclic NTT, approximate digit basis conversion, exact mixed-radix centered CRT. |
| L2 | Fixed 64-byte-aligned slabs, checked non-owning polynomial/ciphertext views, read-only evaluation-key views. |
| L3 | Caller-owned bounded workspace with scoped rewind. |
| L4 | Immutable setup tables; ciphertext arithmetic, tensor product, hybrid relinearization, exact divide-and-round rescale, Galois rotation/conjugation, scalar operations and level reduction. |
| L5 | Canonical real/complex CKKS packing, public-key encryption, decryption, secret/public/evaluation-key ownership, OS entropy. |
| L6 | FHN compute plugin, semantic circuit compiler/reusable executables, and separately owned trusted-client data plane. |

Each layer links only downward. L4 cannot include or link L5 key types.
Hybrid switching uses per-digit CRT projector messages
$P E_{\mathrm{digit}} s_{\mathrm{old}}$ under the destination secret.
Digit extension uses approximate BConv; removing
$P$ and reconstructing decoded coefficients use exact centered CRT. Lower-level
switches retain the full key's original $Q$-to-$P$ offsets.

The encoder supports positive power-of-two real or complex vector lengths up to $N/2$.
Short vectors repeat across physical slots, preserving signed cyclic rotations
of their logical length. Positive rotation steps rotate left. Decryption
requires the exact original logical count. Unsupported counts, nonfinite values,
coefficient-encoding overflow (int64 or centered active-$Q$ range), incompatible
buffers, missing keys and depleted
levels fail rather than substituting another computation.

Complex slots use a conjugate-symmetric canonical embedding: each slot $z$ is
paired with $\overline{z}$ at the reflected root. The existing ring arithmetic,
rotations and conjugation operate on both components. There is no universal
$\lvert z\rvert < 1$ input restriction: the scale, encoded coefficient range, modulus and
subsequent computation determine usable magnitude and precision.

On the raw opcode path, `MULT_CC` is a $2\times2$ tensor product with three output components. `RELINEARIZE`
returns two components; `RESCALE` removes one actual $Q$ limb and divides the
scale. `HMULT` performs all three. Scalar multiplication also consumes one
level, even for integer and zero constants. Add/subtract require equal levels,
component counts and scales within relative $10^{-12}$; there is no automatic scale
or level alignment. Plaintext-buffer and boolean/comparison opcodes are not
advertised. The circuit author remains responsible for numerical headroom.

## Client and evaluator lifetime

The additive `fhn_client_create/destroy/prepare/encrypt_vec_f64/decrypt_vec_f64`
exports keep ABI version 1 and the existing vtable layout intact. Native symbol
names have the `ckks_` prefix. It deliberately does not export the old
same-compute-context encryption/decryption functions.

Optional `ckks_fhn_client_encrypt_vec_c64` and
`ckks_fhn_client_decrypt_vec_c64` add packed complex I/O. Resolve them independently
using `FhnClientEncryptVectorComplex64Fn` and `FhnClientDecryptVectorComplex64Fn`;
each element is the C-compatible `FhnComplex64 { double real, imag; }` from
`FHN/fhn_backend_api.h`. Both components must be finite, and decryption requires
the exact logical count. Rejected calls leave the output unchanged. The real
and complex functions share the same ciphertext representation: real decryption
returns its real projection. ABI version 1 and the vtable layout remain unchanged.
See the [complex monitoring example](../examples/private_spectrum/README.md) for
scoped cross-spectrum evaluation through a separately linked contribution.

Create a compute context, then one client for it. `client_prepare(program)`
generates only the needed relinearization and Galois keys before execution.
The client owns the secret, public key, evaluation keys and its own workspace.
The compute context has no client or secret-key reference; it keeps weak
observers of evaluation keys, borrowed during one kernel call. Destroying the
client expires these keys: key-switching operations fail, while addition can
still execute. A compute context accepts only one client lifetime to prevent
accidental mixing of independent key identities. Buffers are checked against
their creating context. Free buffers and destroy the client before the context,
and unload the library last. Calls on one context must be serialized.

`CorpusBackend` resolves the complete optional client group, prepares the
program and preserves that lifetime order, including move operations. Old
scalar/packed backend APIs remain supported. The installed-package smoke test
uses only installed public C-ABI headers and the installed native plugin to
perform an encrypted fractional square and a compiled scaled-square circuit.

## Default profile and boundaries

The default profile is $N=8192$ (4096 physical slots), three distinct $Q$ primes
near $2^{50}, 2^{40}, 2^{40}$, one $P$ prime near $2^{60}$, $\mathrm{dnum}=3$,
and scale $2^{40}$. Every prime is $1$ modulo $2N$. Fresh ciphertexts are level 2;
two sequential `HMULT` operations
reach level 0. The prime search is deterministic; secret keys, errors, public
keys and encryption randomness come from the OS CSPRNG, never a fixed seed.
Secret/ephemeral polynomials are uniform ternary; errors use the difference
of two independent 21-bit population counts. Uniform residues use rejection
sampling. Entropy failure aborts the operation. Secret-key storage and key
operation sensitive scratch are wiped on release.

Optional JSON fields are `log_degree`, `main_primes`, `aux_primes`, `dnum`,
and `log_default_scale`; unknown fields and unsupported profiles are rejected.
An empty object selects the default. Explicit prime arrays are decimal unsigned
integers. The adapter rejects sparse/terminal profiles and empty full-level
digits, and conservatively bounds the total $QP$ bit length for degrees 1024
through 32768 using the public uniform-ternary 128-bit classical table in
[SEAL](https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/hestdparms.h).
The default's conservative total is 190 bits against that table's 218-bit
$N=8192$ bound. This is parameter screening, not an independent security audit
or certification of this implementation.

This is a leveled CPU implementation. Bootstrapping, general ciphertext
serialization and cross-process transport, GPU kernels, side-channel hardening,
sanitized/adversarial-result decryption and production security review remain
outside this version. No competitive latency or throughput claim is made.

## Memory accounting and validation

No successful evaluation kernel allocates from the heap or generates a key.
The caller supplies fixed output storage and an Arena. The adapter stages
results in its preallocated workspace so rejected operations and output aliases
do not corrupt existing buffers. `scratchBytes` figures are conservative
bounds including alignment, not exact minimum requirements. Setup explicitly
prepares tables for all configured levels; it does not lazily allocate during
execution.

Each native buffer reserves three full-$QP$ polynomials: 786432 bytes for the
default profile. Live two-component payload is 393216 bytes at level 2 and
shrinks to 131072 bytes at level 0, while reserved capacity stays unchanged.
`ckks_fhn_buffer_info` reports these separately along with workspace capacity
and observed high water. `fhn_level_bytes` reports the fixed reserve at every
valid level. The movement model excludes keys, tables and workspace; this CPU
backend does not physically evict memory, so modeled transfer counts/budgets
are not measured transfers or process-memory limits.

Tests cover independent large-integer CRT oracles; key equations and noise
consistency across RNS limbs; fractional packing and full-$Q$ decoding; randomized
public-key-only encryption and wrong-key mismatch; two sequential encrypted
multiplications; positive/negative/identity/conjugate rotations; multiple $Q$/$P$
primes per digit and lower levels; aliases, malformed metadata, scratch limits,
expired keys, entropy failure, secret wiping and zero successful-kernel heap
allocations. Native $8\times8$ output is checked against both a plaintext oracle and
primitive fallback. Whole-suite Debug, sanitizer and installed-package results
are recorded in the execution handoff rather than treated as a security proof.
