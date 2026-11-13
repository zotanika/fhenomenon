# CUDA arithmetic source and contract

The reference implementation reuses parts of [Cheddar](https://github.com/zotanika/cheddar-fhe)
at submodule commit `c428a134bd871c57f73b286c695eb55a13f4f2e6`:

- `include/common/Basic.cuh`, `DoubleWord.h` and `Type.h`, included from the
  submodule for modular Montgomery arithmetic, word operations and index reversal.
- The forward and inverse butterfly functions from `include/ntt/NTTUtils.cuh`,
  extracted into `CheddarButterfly.cuh` to avoid unrelated allocator dependencies.
  The adjacent [MIT notice](LICENSE.cheddar) applies to the reused source.

Fhenomenon supplies canonical RNS buffers, table conversion, NTT launch/storage
layout, stream ownership, key-switch/rescale orchestration, packing, ciphertext
handles, representation transforms and target-level refresh. This is not a link
to the complete Cheddar context, RMM allocator, key generator or bootstrap code.

`CudaPrimitives` accepts odd moduli `3 <= q < 2^62`, canonical input residues and
native bit-reversed root tables. Public helper results are canonical. Montgomery
conversion is internal; ciphertext storage stays in the common representation.
Small NTTs use a single CTA; larger transforms use explicit stages on the supplied
stream. Buffers and tables must live until completion. No implicit host
synchronization or secret-key access belongs to these primitives.

`PrimitiveTest` compares modular results and NTT/INTT against independent CPU
oracles, including sizes 2 through 131072 and near-modulus inputs.
`EngineDifferential` compares complete encrypted operations. `UploadOrderingTest`
and the device-scope test check stream ordering and context ownership.
See [build and operation coverage](../../../../docs/kernel-providers.md).
