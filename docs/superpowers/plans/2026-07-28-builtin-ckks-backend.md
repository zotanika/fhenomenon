# Built-in RNS-CKKS Backend — Plan and Handoff

Date: 2026-07-28
Status: **design fixed, no code written.** Five decisions are open and two
of them block the first line of implementation. This document is the
handoff: it records what was decided, what was deliberately left open, and
what a fresh session needs in order to continue without re-deriving any of
it.

Read first: [`../specs/2026-07-26-builtin-ckks-backend-design.md`](../specs/2026-07-26-builtin-ckks-backend-design.md).
This file assumes it.

## Where things stand

Described by artifact rather than by commit hash, so this survives a rebase:

| Artifact | State |
|---|---|
| `experiments/` | GPU experiment track — build script, env fingerprint, runner, journal, machine-readable log, run evidence |
| `../specs/2026-07-26-builtin-ckks-backend-design.md` | The architecture spec — R1–R5, I1–I5, L0–L6 |
| `cmake/FhnCkksLayer.cmake` | Layer-target enforcement for invariant I5 |
| `src/CKKS/params/` | **L0 `Params`** — validated, immutable, hashable |
| `src/CKKS/arena/` | **L3 `Arena`** — caller-owned bump allocator with `Scope` and high-water tracking |
| `test/CKKS/` | One test executable per layer, each linking only its own layer target |

Everything above L0/L3 is unimplemented. The built-in backend is still ToyFHE
and remains fully functional; nothing here degrades it.

Two threads are live. **A** is the primary one; **B** is running and has one
recorded blocker that A happens to unblock.

## Thread A — the built-in RNS-CKKS backend

### Decisions locked

These were settled through discussion and evidence, not assumed. They should
not be reopened without new evidence.

1. **The goal is architectural, not cryptographic.** Every CKKS library
   exposes the same operation set; that interface is commodity. The
   justification for writing another implementation is that a backend
   executing *compiled programs it did not write* has structural
   requirements a monolithic library cannot meet. See the spec's Premise
   and R1–R5.

2. **Invariants are derived, not asserted.** R1–R5 (what the FHN model
   requires) → I1–I5 (what that forces) → L0–L6 (the layering). Every
   invariant in the spec names the requirement it comes from. If a future
   change makes one inconvenient, the question to ask is which requirement
   is being given up.

3. **Memory is a planned arena, not a pool.** Because an `FhnProgram`'s
   def-use chain is exact at plan time, so is its scratch requirement.
   This is the one component with no counterpart in a conventional library
   and it is the first falsifiable claim to measure: *this backend can
   report a program's exact memory footprint before running it.*

4. **ToyFHE stays.** It is the CI and backend-author reference. Being small
   enough to read in one sitting is its entire value; the CKKS backend does
   not replace it.

5. **Performance claims are measured or not made.** Any "competitive" or
   "SOTA" claim must be a recorded run in `experiments/`, against an
   established open-source library wrapped behind the same FHN ABI. Since
   the interface is commodity, wrapping a baseline is cheap — and without
   one, no performance claim is falsifiable.

6. **Documentation cites open-source only.** Design evidence in this
   repository must be reproducible by a reader. Cite open-source
   implementations at `file:line`; do not cite non-open-source code, even
   indirectly. Where a lesson came from somewhere unciteable, state it as a
   design decision justified on its own merits.

7. **The scheme is CKKS. TFHE was considered and declined** (2026-08-11).
   This confirms the standing decision already recorded in
   [`../specs/2026-07-13-level-aware-byte-budgets-design.md`](../specs/2026-07-13-level-aware-byte-budgets-design.md)
   ("CKKS-first (TFHE deferred) — the model targets leveled arithmetic
   schemes"), reached independently a second time.

   The decisive argument is not that TFHE is a crowded field. It is that
   **everything this runtime has built only has meaning on a leveled
   scheme.** Level-varying ciphertext size, the level/scale ledger,
   rotation-based data movement, slot layout, byte-budgeted residency, and
   fused-opcode decomposition are all properties CKKS has and TFHE does
   not: TFHE ciphertexts are fixed-size, programmable bootstrapping resets
   noise so there is no level to track, there are no slots to rotate, LWE
   samples are small enough that residency is not the bottleneck, and
   performance is dominated by one primitive rather than by composition.
   A runtime whose thesis is *planning composition, movement, and memory
   across a program* would not be validated by winning on TFHE, because
   TFHE offers almost nothing to plan.

   Two honest counterweights, recorded so this is not re-litigated from a
   half-memory: TFHE has exact integer semantics, which the corpus verify
   path (`fhn_encrypt_i64`/`fhn_decrypt_i64`) wants natively while CKKS
   needs a rounding adapter; and TFHE has no leveled-vs-bootstrapping
   dilemma, since PBS is in every operation. The second is the same fact
   as "nothing to plan", seen from the other side.

   Nothing is discarded. `TfheBackend.{h,cpp}` (284 lines) and the Rust
   crate stay quarantined behind `FHENOMENON_USE_TFHE`; the boolean
   opcodes `FHN_AND/OR/XOR/EQ/LT/LE` exist in the enum with no
   implementation anywhere, so the door to a TFHE-class *external* backend
   stays open at zero carrying cost.

8. **Key switching is hybrid, parameterised by `dnum`** (2026-08-11). Rather
   than choosing one of BV / GHS / hybrid, the implementation takes `dnum` —
   the number of digits the main chain is split into — as an explicit
   parameter, because the three are endpoints of one continuum: `dnum == 1`
   is GHS (a single digit, so `P` must cover all of `Q`), `dnum ==
   levelCount()` is a BV-style decomposition, everything between is hybrid.
   One implementation covers all three, which turns what the plan called the
   most expensive decision to reverse into a number to measure.

   `dnum` is an explicit field of `Params` with no default that hides it.
   What is genuinely expensive to reverse is not the value but four
   structural facts, and they are settled now: keys are stored per-digit
   rather than as one pair; precomputation tables are keyed by the
   (digit basis → `PQ` basis) pair; `Params` keeps the special primes `P` in
   a list separate from the main chain; and level accounting knows that key
   switching raises to `PQ` and comes back. Get those right and `dnum` is
   just a number.

   Honest cost: the generalised path carries slightly more index arithmetic
   than a hardcoded one, and `dnum == levelCount()` is representable without
   necessarily being an *efficient* BV implementation.

### Decisions pending

Ordered by how expensive they are to reverse. None of them now block the
first slice.

---

**(2) Cryptographic scope — blocking for parameter design only.**

Leveled-only first, or bootstrapping declared as the target from the start?
The invariants are identical either way; the parameter surface is not.
Non-goals in the spec keep bootstrapping's parameter surface (terminal
primes, sparse Hamming weight, sparse-secret encapsulation) *representable*
in `Params` without implementing anything, which is cheap insurance. If
bootstrapping is declared, `Params` also needs to express two secret
distributions coexisting.

---

**(3) The baseline backend.** Which open-source CKKS library gets wrapped
behind the FHN ABI as correctness oracle and speed reference. Not blocking —
but it should exist before the first performance number, or decision 5 above
cannot be honoured.

**(4) Prime width and modmul strategy.** 50–60-bit primes are conventional
on CPU; Cheddar uses ~30-bit for GPU. On this machine (aarch64, Cortex-X925
+ A725, SVE2, no AVX-512) the natural path is `umulh`-based Shoup/Barrett
with lazy reduction and a scalar kernel written first, vectorised later.
Affects L1 `NttTables` and L4 kernels; does not affect the layering.

**(5) Where arena sizing lives.** The spec proposes
`eval::scratch_bytes(op, params, level)` at L4, with the program-level
maximum computed by the caller. The alternative is to fold it into
`FhnMovementPlan`, which already walks the instruction stream and infers
levels. Deferrable until L4 exists.

**(6) Whether an empty key-switching digit should be rejected.** `Params`
currently accepts a `dnum` that leaves a trailing digit empty — 4 main primes
over `dnum = 3` gives `alpha = 2` and digits `[0,2) [2,4) [4,4)`, so the
caller's 3 behaves as 2. The partition stays exact and nothing miscomputes,
but a `dnum` that silently means something else is the kind of hidden
discrepancy invariant I4 exists to prevent. Rejecting it is a one-line
change; it was left permissive rather than decided quietly.

### First implementable slice

Three things are invariant across every open decision, so they were built
first.

1. **Layer skeleton — done.** `cmake/FhnCkksLayer.cmake` declares one static
   library per layer. Two mechanisms enforce I5, and both are verified rather
   than asserted: each layer owns a private include root, so a wrong
   `#include` does not compile; and a layer may only depend on a strictly
   lower index, so an upward or sideways edge is a configure-time error. The
   CKKS tree is filtered out of `add_fhenomenon_sources()`'s glob in the root
   `CMakeLists.txt` — without that filter the layers would be silently
   swallowed back into the monolithic library and the enforcement would
   evaporate. **Do not remove that filter.**
2. **`Params` (L0) and `Arena` (L3) — done.** `Params` validates the whole
   spec (NTT-friendliness `q ≡ 1 mod 2N`, prime width, duplicates, `dnum`
   range, and that `P` exceeds the widest key-switching digit) and reports
   the first reason as a string; `ciphertextBytes(level)` is the R5 level
   model reduced to a pure function. `Arena` refuses to grow when exhausted,
   tracks a high-water mark so a caller can check the size it planned was the
   size it needed, and offers `Scope` for nested temporaries.
3. **`NttTables` + negacyclic NTT (L1/L4) — next**, tested standalone: no
   keys, no allocator singleton, no layer above. If that test needs anything
   from L5, the layering has already failed and CI will say so.

`eval::scratch_bytes` was deliberately *not* added yet. A declared function
with no definition is a footgun, and the contract only becomes meaningful
once one opcode implements it — it lands with L4.

Then: RNS base conversion → `ModUp`/`ModDown` → hybrid key switching (per
locked decision 8) → the FHN kernel table.

## Thread B — GPU experiment track (Cheddar on the DGX Spark)

Full methodology and machine spec: [`../../../experiments/README.md`](../../../experiments/README.md).
Results log: `experiments/JOURNAL.md` and `experiments/log.jsonl`.

State as of the last recorded run:

- **build** — SUCCESS. `libcheddar.so` for `sm_121` on GB10.
- **gpu-test** — PASS 10/10. Context+keygen 1062 ms one-time; ADD_CC 10 µs,
  HMULT 163 µs, rotate-pair 152 µs.
- **corpus** — BLOCKED. A frontier, not a regression. The corpus verify path
  is exact-integer (`fhn_encrypt_i64`/`fhn_decrypt_i64`) because its oracle
  checks integers; Cheddar is approximate CKKS and exports only the `f64`
  plane, so `CorpusBackend` rejects the library up front. Two further
  blockers sit behind that one — `fhn-corpus` hardcodes `create(nullptr)`
  while Cheddar requires a param config, and the 12 shapes need a Spark
  param set covering depth 15 plus per-shape rotation keys.

**Interaction with Thread A worth noting:** a built-in CKKS backend with an
i64 adapter runs the 12-shape corpus on CPU with a *real* level model, with
no GPU dependency at all. Today ToyFHE declares a flat one-level model, which
makes the Belady-vs-LRU movement analysis vacuous. Thread A therefore
unblocks the corpus independently of the three Cheddar-side blockers above.

## Resuming in a fresh session

**Read in this order:** the spec → this file → `experiments/README.md`.

**Open branches as of 2026-08-11** (none merged to `main` yet; `main` is still
at the level-aware byte budgets commit):

| Branch | Contents | Note |
|---|---|---|
| `feat/gpu-experiment-track` | `experiments/` + the `.gpu-deps/` gitignore entry | **Merge this first.** This file links `experiments/README.md`; that link dangles until it lands. |
| `feat/real-builtin` | This file, the architecture spec, the scheme decision, and the L0/L3 layer skeleton | 4 commits |
| `docs/readme-scoped-execution` | Restores the README's Scoped Execution section and corrects the "legacy session execution path" claim | Independent of the other two |

Each branch is self-contained and rebased onto `origin/main`, so they can be
reviewed and merged in any order — the only ordering that matters is the link
noted above.

**Environment (already verified, do not re-derive):** host `spark-0faa`;
NVIDIA GB10, compute capability 12.1 → `sm_121`; CUDA 13.0; aarch64 Cortex-X925
+ Cortex-A725 with SVE2, 20 cores; 122 GiB unified memory; GCC 13.3, CMake 3.28.

**Do not:**

- commit the `refs/cheddar-fhe` working-tree change. The CUDA arch patch
  (`60 61 70 75 80 86 89 90` → `121`) is applied by `experiments/build-cheddar.sh`
  and is dirty *by design*. `git status` showing ` m refs/cheddar-fhe` is the
  expected steady state on this machine.
- add members to `FhnBackendCtx`. It is a composition root — a bundle of
  references. Growth there is exactly the failure mode the spec exists to
  prevent (see the spec's Cheddar evidence table).
- assert a performance comparison without a recorded run in `experiments/`.
- cite non-open-source code as design evidence.
- remove `list(FILTER FHENOMENON_SOURCES EXCLUDE REGEX "/CKKS/")` from the
  root `CMakeLists.txt`, or link `${PROJECT_LIB_NAME}` into a layer test.
  Either one silently folds the layers back into the monolith and the I5
  enforcement stops enforcing anything.

**Verifying the enforcement still works** (both should fail):

```bash
# 1. a layer must not see another layer's headers
echo '#include "CKKS/Params.h"' | g++ -std=c++17 -x c++ -fsyntax-only -I src/CKKS/arena/include -
# 2. temporarily give a layer an equal-or-higher DEPENDS, then configure
```

**Open the conversation with:** decision (4), prime width and modmul strategy.
It is the only pending decision that touches the next slice (L1 `NttTables` +
the negacyclic transform).

Note that (4) is narrower than it looks, and can largely be deferred rather
than answered. Prime *width* is already a per-parameter-set choice, not a
build-wide one: `Params` accepts any prime below `kMaxPrimeBits` (62) that is
`1 mod 2N`, so a 30-bit chain and a 60-bit chain are both representable
without touching code. What actually has to be chosen now is the modmul
*strategy*, and a scalar Shoup-with-lazy-reduction transform written against
a `< 2^62` modulus keeps every width on the table while staying portable —
vectorisation (SVE2 here, AVX2/AVX-512 elsewhere) is a later, local change.

Watch one landmine: the build is `-Wpedantic -Werror`, and GCC's `-Wpedantic`
objects to `__int128`. Any 128-bit intermediate needs its suppression confined
to a single small header rather than sprinkled through the arithmetic.

(This paragraph replaced a stale line pointing at decision (1), key switching,
which locked decision 8 the same day the pending list was renumbered.)
