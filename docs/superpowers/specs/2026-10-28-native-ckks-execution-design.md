# Genuine native leveled RNS-CKKS execution

The user asks to implement the real backend after the audit correction clearly identified ToyFHE and the unfinished native layers. This continues the fixed native architecture and hybrid/dnum decisions in the July/August design and handoff. CPU leveled execution is the concrete first complete path; bootstrap and GPU remain separate. This is implementation authorization, not a request to reopen the settled scheme choice.

## Outcome

Implement actual RLWE public-key encryption and canonical CKKS packing, real polynomial evaluation, hybrid relinearization/Galois switching, and coefficient divide-and-round rescale. Execute the existing encrypted8x8 matvec through native and primitive fallback paths, with no plaintext arithmetic or secret-key-assisted evaluation. Preserve ToyFHE as a separate reference rather than presenting it as genuine crypto.

## Architecture and choices

Preserve L0-L6 downward dependencies, explicit fixed storage, caller Arena, independent Params coexistence and compute-only FHN IR. Complete L2 memory bindings; add exact mixed-radix CRT at L1; L4 contains free evaluation functions and immutable explicit tables; L5 owns secret/public/evaluation keys and injected randomness; L6 integrates the ABI.

Use hybrid CRT-projector keys: a digit pair encrypts P*E_digit*s_old under s_new over QP. Existing approximate digit BConv is valid in this path; ModDown and decoding use exact centered CRT. Rotation first applies sigma_g then switches sigma_g(s) to s; g=5^step implements positive-left rotation. Native inverse NTT is already normalized.

Packing repeats a shorter power-of-two logical vector across N/2 canonical slots; zero padding would violate current cyclic rotation semantics. Real input/output is the first supported portable data plane. Tensor2x2 produces3components, relin3→2, rescale removes a real RNS limb and divides scale. A second multiplication level is included in the default profile and tested. Scalar multiplication always performs encoded multiplication plus rescale and therefore consumes one level.

The existing same-context encrypt/decrypt exports cannot enforce secret separation structurally. Add optional host-only FhnClientCtx exports: create/destroy, prepare(program), encrypt/decrypt vector. Native FhnBackendCtx has no secret key or client backpointer. Client owns key material; evaluation context retains only weak evaluation-key observers, borrowed for one call. Corpus supports this optional group while preserving all existing backend APIs and ABI1 vtable layout. Preparing a program derives required rotation/relinearization keys before kernels execute.

Default profile: N8192, Q near50/40/40bits, P near60bits, dnum3, scale2^40; all primes distinct and1mod2N. Sample uniform ternary secrets/ephemeral polynomials, unbiased uniform residues, independent CBD21-minus-CBD21 errors. Production entropy comes from OS CSPRNG and failures stop key generation/encryption. Deterministic RNGs live only in tests. The adapter checks a conservative totalQP modulus-bit bound against the public ternary parameter table; this is not independent security certification.

All successful evaluation kernels use supplied output storage and Arena, with no heap allocation or key generation. Fixed buffers reserve enough for fresh-level3component results; live ciphertext payload, reserved capacity, scratch and key/table memory are distinct quantities. Do not report decreasing live size as decreasing reserved allocation. The first table bundle explicitly prepares the configured levels at setup; lazy subset preparation is a later optimization, not hidden execution-time caching.

## Alternatives

Wrapping an established CKKS library gives real crypto quickly but does not exercise the project's explicit-memory and layer goals. Finishing only storage or more Toy validation leaves no genuine executable circuit. A complete bootstrap implementation expands cryptographic and scheduling scope before establishing leveled correctness. Continue the already selected native hybrid design and deliver a small genuine end-to-end path, then measure before optimizing.

## Validation and limits

Each layer must link without higher layers. Check slab alignment/stride/constness/capacity; exact CRT against independent integer oracles; NTT Galois order; direct encryption randomness and public-key-only signature; independent plaintext arithmetic; three/two component transitions, real scale/level changes, two sequential multiplication levels, positive/negative/identity/conjugate rotations; in-place aliases; two parameter contexts; key expiry; insufficient scratch; zero successful-kernel heap allocations. End-to-end corpus must execute every selected slot and fail rather than skip required verification. Run full Debug/noLTO/ASan+UBSan and install regression checks.

No bootstrap, general ciphertext serialization protocol, GPU implementation, competitive performance claim, side-channel audit or production-security certification is included. Existing byte-residency model limitations remain explicit. Do not weaken tests, reuse plaintext in evaluation, or decrypt-reencrypt to make an operation pass.

## Reproducible references

- Existing native architecture/handoff under docs/superpowers, and local refs/cheddar-fhe/src/UserInterface.cu hybrid projector construction.
- SEAL's public ternary modulus bounds and RLWE sampling implementation: https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/hestdparms.h and https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/rlwe.cpp.
- OpenFHE hybrid switching reference: https://github.com/openfheorg/openfhe-development/blob/main/src/pke/lib/keyswitch/keyswitch-hybrid.cpp.

Implement original code from the verified formulas; no bulk copying of reference sources.
