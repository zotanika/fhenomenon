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

`main` currently contains:

| Commit | What |
|---|---|
| `2f725c5` | GPU experiment track (`experiments/`) — build script, env fingerprint, runner, journal, machine-readable log, run evidence |
| `4224600` | The architecture spec — requirements R1–R5, invariants I1–I5, layers L0–L6 |

Nothing of the CKKS backend is implemented. The built-in backend is still
ToyFHE and remains fully functional; nothing in this plan degrades it.

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

### Decisions pending

Ordered by how expensive they are to reverse. **(1) and (2) block the first
line of implementation; (3)–(5) do not.**

---

**(1) Key-switching variant — blocking, most expensive to reverse.**

Assumed in the spec: hybrid key switching (GHS + RNS digit decomposition,
special primes `P`, `dnum` digits). Starting with BV or GHS-only and
retrofitting hybrid means rewriting `ModUp`/`ModDown`, the key format, the
`Tables` layer's base-conversion identities, and every level-accounting
rule — effectively the whole backend.

Needs: explicit confirmation of hybrid, and a `dnum` policy (fixed, or
derived from the prime chain).

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
levels. Deferrable until L3 and L4 exist.

### First implementable slice

Independent of every open decision, three things are invariant across all
branches and can be built first:

1. **Layer skeleton** — one CMake target per layer (L0…L6), linked strictly
   downward, plus a test per layer that links *only* that layer's target.
   This is invariant I5, and neither this repository nor the libraries
   surveyed have it today (`src/CMakeLists.txt` gathers every source into a
   single `add_library`). Building it first means the layering cannot rot
   while the rest is written.
2. **`Params` (L0) and `Arena` (L3)** — the immutable parameter value type
   and the caller-owned bump allocator, plus the `eval::scratch_bytes`
   signature as a contract with no implementations yet.
3. **`NttTables` + negacyclic NTT (L1/L4)**, tested standalone — no keys, no
   allocator singleton, no layer above. If that test needs anything from L5,
   the layering has already failed and CI should say so.

Then, gated on decision (1): RNS base conversion → `ModUp`/`ModDown` →
hybrid key switching → the FHN kernel table.

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

**Open the conversation with:** decision (1), key switching. It is the most
expensive to reverse and gates everything after the first slice.
