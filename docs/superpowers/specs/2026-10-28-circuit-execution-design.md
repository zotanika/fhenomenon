# Application-driven native circuit execution

## Authorization and objective

The user explicitly says the op table is convenience and authorizes disruptive
changes. Current test applications being more flexible and faster are the main
metric. Earlier intent rejects treating multiply/relinearize/rescale, or HMULT,
as mandatory execution atoms. This authorizes implementing a concrete internal
redesign; do not reopen a permission gate for the already authorized work.

Work continues in the original writable checkout, preserving the completed
native CKKS and audit changes. No commits, pushes, or unrelated user-file edits.
The unused attached worktree now has write permission, but does not contain the
current changes and is not the implementation source.

## Chosen direction and alternatives

A prepared opcode tape/pool removes dispatch/allocation but leaves repeated
residue/key checks and staging. Application-specific matvec entry points give
local speed but restrict graph composition. Choose a semantic circuit compiler
with reusable native execution regions. This is the implementer's choice under
the user's permission, grounded in current source and app inspection.

The current FhnProgram remains an explicit low-level reference interface.
Introduce a separate value-oriented FhnCircuit: Add/Subtract/Multiply/Scale/
AddConstant/Negate/Rotate/Conjugate. Multiply denotes the application product;
it does not promise an unrelinearized tensor representation. Integer Scale
preserves scale/level using exact modular multiplication. A deliberate importer
maps corpus mathematical semantics to this new contract, without changing old
MULT_CC or MULT_CS semantics. The legacy interpreter remains usable.

Compile a copied graph and actual encrypted input signatures to an opaque
executable. It owns reusable internal storage and scratch, never plaintexts or
secret keys. A host client prepares the executable's requested public keys;
run binds fresh input/output handles with matching signatures. Output is
published only after successful computation. Compiling is separate from
execution, and repeated run performs no heap allocation or key generation.

## First regions

1. Collect affine subgraphs with exact signed integer coefficients and finite
   constants. Merge repeated leaves, cancel zero terms and stop at expensive
   nonlinear/rotation boundaries. Control expansion; never expand a rotate-add
   reduction into exponentially many rotations. Hoist cheap shared affine
   expressions only when its compact representation remains bounded.
2. Lower sums of same-level/same-scale ciphertext products plus linear terms
   into a quadratic region, where profitable and precisely representable. Lift
   linear terms by the common integer scale, accumulate a three-component
   product, then relinearize/rescale once. This supports wide-front without
   pretending differently scaled branches match. General scale-join rewriting
   is rejected unless it is mathematically defined; no metadata relabel hack.
3. Managed standalone products align Q levels without altering value and use
   real multiplication/key switching/rescale. Linear joins require compatible
   actual scales, or a supported quadratic-region rewrite. Rotations can fuse
   their final addition with an affine consumer when the graph permits it.
4. Implement exact native fusion of key-switch P removal and rescale by q.
   For odd P,q, centered round division satisfies
   D_q(D_P(A)+t)=D_(Pq)(A+P*t). Reconstruct the combined centered remainder from
   P and dropped-q limbs once and emit only surviving Q limbs. Test bit-exact
   equality to separate tensor/relinearize/rescale on identical inputs/keys.
   Preserve source-digit NTT limbs instead of inverse/forward round-tripping
   them during switching. Do not assume unsigned approximate BConv commutes
   with signed Galois permutations; full QP hoisting is not part of this slice.

Private region kernels operate on validated views, known metadata, immutable
key bindings and supplied Arena. Public checked kernels retain validation and
transactional contracts. Trust is established at native input and key binding;
no arbitrary external memory view earns a permanent validation certificate.
Immutable public keys are validated once when binding, leased once per run and
expire on client destruction. Region outputs remain private until publication.

## Storage and boundaries

The compiler maps logical lifetimes to reusable physical slots. A destination
must not overlap a region's live input; release last-use slots only after that
region. Internal slots can use actual Q two-component capacity rather than
public buffers' three-full-QP capacity. Reserve all internal storage and scratch
before run. Report slot bytes and scratch separately; include external I/O and
key/table allocations as exclusions, never claim process-memory enforcement.

Input ports may bind the same immutable ciphertext. Distinct output values
cannot alias one physical destination. Output/input alias is supported through
late publication. Rebinding ciphertexts with the same signature is supported;
wrong owner/count/packing/level/scale, expired keys and insufficient budgets
fail before publication. Original circuit mutation/destruction cannot alter a
compiled executable. Context and executable calls remain serialized in this
slice; an instance owns its scratch, avoiding reliance on shared mutable eval
scratch. Destroy executables before their compute context.

## Metrics and scope

Baseline is the actual native Release plugin, copied before modifications.
Measure all-slot correctness under the same N8192 Q/P/default precision and
atol=1e-4,rtol=1e-6, without changing plaintext workloads, hiding key generation
inside evaluation, or making encrypted weights public. Timing uses completed
CPU evaluation, preallocated reference buffers, warmups and paired/interleaved
repetitions. Record distributions, setup/encryption/decryption separately and
actual reserved internal bytes. Do not benchmark concurrently with builds/tests.

Initial six executable applications: matvec-fused, matvec, matmul-tile,
reduce-tree, weighted-sum and diamond. Required flexibility targets are conv1d
and iter-update (integer scalar paths); managed products/quadratic joins target
stats and wide-front, yielding ten of thirteen original corpus shapes under
the same default profile. Horner15/logreg exceed the initial depth/profile;
psi-eq requires exact Boolean semantics. Report these limitations explicitly.
The user-owned resnet_circuit currently runs plaintext/ReLU and is unregistered;
do not count its output as encrypted evaluation or silently approximate ReLU.

## Verification

Meaningful RED/GREEN tests for circuit semantics, structural rejection, affine
coefficients/overflow, quadratic branch alignment, repeated rebind, aliasing,
key expiry and output preservation. Exact residue differential tests for Pq
fusion and old scalar/drop semantics; zero successful-run allocations; real
public-key input/output tests. All selected applications compare to the existing
independent plaintext oracle. Retain existing 293 tests and ABI1 behavior.
Run whole Debug/ASan+UBSan, Release/no-tests/install, and independent Astra
review. Accept speed claims only from actual app measurements; report regressions
and retain the reference/control path. No bootstrap, GPU, JIT, frontend rewrite,
external cryptographic certification or blanket performance claim in this slice.
