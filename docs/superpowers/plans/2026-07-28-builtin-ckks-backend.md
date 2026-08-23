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
| `src/CKKS/tables/` | **L1 `NttTables` + `ntt::forward`/`ntt::inverse`** — negacyclic transform, Harvey lazy butterflies; **L1 `BConvTables` + `bconv::convert`** — fast RNS base conversion; `ModArith.h` is the one place that names `__int128` |
| `src/CKKS/storage/` | **L2 `ParamsId` / `RnsBasis` / `PolyLayout` / `CiphertextLayout`** — value arithmetic over index ranges, available before any memory exists |
| `src/CKKS/arena/` | **L3 `Arena`** — caller-owned bump allocator with `Scope` and high-water tracking |
| `test/CKKS/` | One test executable per layer, each linking only its own layer target |

Everything above L0/L1/L2/L3 is unimplemented. The built-in backend is still
ToyFHE and remains fully functional; nothing here degrades it.

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

**(4) Prime width and modmul strategy — settled 2026-08-12, and it split in
two.** The *strategy* half is decided and implemented: scalar Shoup
multiplication with Harvey lazy reduction, in `src/CKKS/tables/`. The *width*
half turned out not to be a decision at all. `Params` and `NttTables` both
accept any prime below 62 bits that is `1 mod 2N`, so a 30-bit chain and a
60-bit chain are already representable without touching code — width is a
property of a parameter set, not of the build. What remains is measurement
on real parameter sets, not a choice to make.

The 62-bit ceiling is Harvey's bound rather than a preference: lazy
butterflies carry intermediates up to `4q`, which fits a `uint64` only while
`q < 2^62`. `Params::kMaxPrimeBits` and `NttTables::kMaxModulusBits` are
therefore the same number derived twice, stated independently because L1 does
not see L0. `test/CKKS/NttTest.cpp` exercises a 62-bit modulus specifically so
that an off-by-one in a lazy window fails there.

Vectorisation (SVE2 on the Spark, AVX2/AVX-512 elsewhere) stays open and is a
local change behind `ModArith.h`, which is the only place in the tree that
names `__int128`.

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
3. **`NttTables` + negacyclic NTT (L1) — done**, tested standalone: no keys,
   no allocator singleton, no layer above. `CkksNttTest` links
   `fhn_ckks_tables` and nothing else, so if that test ever needs a layer
   above it, the link fails and CI says so.

   This slice was planned as "L1/L4" and built as L1 alone. Putting the
   transform at L4 was wrong: L4 is `eval::`, the layer of CKKS operations
   over ciphertexts and keys, and a negacyclic NTT over a single prime has
   none of those inputs — it is arithmetic on a span of residues given
   tables. It also needs no scratch, because in-place Cooley-Tukey and
   Gentleman-Sande butterflies allocate nothing, so it sits below L3 rather
   than above it. Placing it at L4 would have forced this layer's test to
   reach upward for something it does not use, which is the erosion I5
   exists to catch. `NttTables` is keyed by `(modulus, log_degree)` and not
   by `Params` for the same reason, which leaves L1 depending on nothing at
   all — the spec already said `NttTables(prime, logN)`; it was this plan's
   prose that drifted.

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

**Branch state as of 2026-08-24.** `main` is at the open-questions merge and
now carries the layer skeleton; two branches remain.

| Branch | PR base | State |
|---|---|---|
| `feat/gpu-experiment-track` | `main` | **merged** — `experiments/` and the `.gpu-deps/` gitignore entry are on `main` |
| `feat/real-builtin` | `main` | **merged as #14** (squash) — this file, the spec, the scheme decision, I5 enforcement, the L0/L3 skeleton |
| `docs/open-questions` | `main` | **merged as #17** — `docs/open-questions/` and question 001 |
| `feat/ckks-ntt` | `main` | **open as #18**, 8 commits, rebased onto `main`. L1 `NttTables`/`BConvTables`/`ModArith.h` and L2 layouts |
| `docs/readme-scoped-execution` | `main` | open — restores the README's Scoped Execution section; corrects the "legacy session execution path" claim |

### Opening the PRs

`gh` is deliberately not authenticated on the Spark box — it is a shared
office machine and a GitHub token should not live there. PRs are opened from
the WSL box instead.

Three things are easy to get wrong:

1. **`feat/ckks-ntt` must target `feat/real-builtin`, not `main`.** It is
   stacked. Based on `main` its PR shows seven commits and re-reviews the
   whole layer skeleton. `gh pr create --base feat/real-builtin`.

   **This repository squash-merges, and that changes what happens next.** An
   earlier revision of this file said GitHub retargets a stacked PR
   automatically once its base merges. That is true of a merge-commit
   repository and false here. A squash merge produces a *new* commit, so the
   base branch's commits are no longer ancestors of `main`; GitHub does not
   retarget, it **closes** the stacked PR when the base branch is deleted, and
   until it is rebased the PR's diff shows every already-merged file again —
   28 of them, the whole L0/L3 skeleton, on the first attempt here.

   So in this repository the procedure after a base PR merges is fixed, and
   both steps are required:

   ```bash
   git fetch origin
   git rebase --onto origin/main <old-base-tip> feat/ckks-ntt   # 8 commits, 0 conflicts
   git push --force-with-lease origin feat/ckks-ntt
   gh pr create --base main                                     # the old PR is gone, not retargeted
   ```

   Anyone with local work on top of the old tip moves it with
   `git rebase --onto origin/feat/ckks-ntt <old-tip> <their-branch>`. Check
   the result with `git diff --stat <old-head> origin/feat/ckks-ntt` rather
   than with `git log`: after a rebase every SHA differs, so a log comparison
   looks alarming and proves nothing, while an additions-only diff proves
   nothing was dropped.
2. **Merge `feat/gpu-experiment-track` before `feat/real-builtin`.** This
   file links `experiments/README.md`; that link dangles until it lands.
   Nothing else has an ordering constraint — no two branches touch the same
   file.
3. **Every code PR should say which compilers have actually seen it.** As of
   PR #18 that list is complete for the CKKS layers: GCC 13.3 on aarch64 and
   x86_64, Clang 21 locally, and **Apple Clang via the macOS CI leg — 3/3
   green**. `ModArith.h`'s `__int128` extension is the thing that was at risk
   and it passed. See "The Clang gap" below, which is now closed rather than
   narrowed.

Verification to state honestly in each PR body:

| Branch | Verified | Not verified |
|---|---|---|
| `feat/real-builtin` | aarch64/GCC 13.3 **19/19** (includes the Cheddar GPU test); x86_64/GCC 13.3 **18/18** (Cheddar absent, so that test never configures); zero warnings on CKKS targets under `-Werror`; clang-format clean under both 18.1.8 and 22; all three I5 checks verified by deliberate violation; `Params.cpp` and `Arena.cpp` also compile clean under Clang 21; **merged to `main` as PR #14** | — |
| `feat/ckks-ntt` | aarch64/GCC 13.3 **22/22** and x86_64/GCC 13.3 **21/21** at the rebased tip; 10 NTT tests and 13 base-conversion tests; **13 of 13 mutations caught** after an adversarial review round — including the accumulator-window mutation an earlier revision recorded as unkillable, whose impossibility argument was wrong (see `BConvTest.cpp`); layer isolation reconfirmed rather than taken on trust; **Clang 21 clean** locally and **Apple Clang clean on CI** (PR #18, 3/3 green) | — |
| `feat/gpu-experiment-track` | No compiled code — scripts, logs, markdown. Outside the clang-format path | — |
| `docs/*` | Markdown only | — |

The commit messages were written to be usable as PR bodies; prefer quoting
them over paraphrasing.

### The Clang gap — closed 2026-08-24 by the macOS CI leg

The narrow question was whether `#pragma GCC diagnostic ignored "-Wpedantic"`
around the `__int128` alias in `ModArith.h` suppresses Clang **at the alias**
or leaves a warning at every **use site**. The answer is neither: **Clang does
not diagnose `__int128` at all**, not under `-Wpedantic` and not under
`-pedantic-errors`. There is nothing for the pragma to suppress. It stays
because GCC does diagnose it — verified in the same run — but no redesign of
that header is needed, and nothing that stacks on it is at risk.

Measured, so that none of this rests on argument:

| Probe | Result |
|---|---|
| GCC 13.3, `unsigned __int128` alias, `-Wpedantic -Werror`, no pragma | **error** — this is why the pragma exists |
| Clang 21, same TU, no pragma, `-pedantic-errors` | **no diagnostic** |
| Clang 21, `-Wpedantic -Werror`, VLA probe | **error** — confirms `-Wpedantic` was actually live, so the line above is a real negative and not a dead flag |
| Clang 21, all four CKKS layer TUs, the project's exact `CLANG_WARNINGS` + `-Werror` | **0 warnings** |
| Clang 21, whole project | builds; the only 4 warnings are inside spdlog's bundled `fmt`, none in first-party code |
| Clang 21, `ctest` | 17/20 — the four CKKS layer tests are **43/43 green**; the three failures are discussed below |

**Apple Clang has since compiled it too**, on PR #18's macOS leg, 3/3 green —
so the ceiling this section describes is no longer a gap at all. The local
result below stands as the reason it was safe to keep stacking work on
`ModArith.h` before CI could say so, which was its whole purpose.

The compiler was Clang 21.1.0 obtained as `zig c++` (`pip install ziglang`),
because the WSL box has no `clang` package and no passwordless `sudo`. That is
a real Clang front end and the diagnostics above are front-end behaviour, so
the `__int128` answer is solid. What it is **not** is Apple Clang, and it is
not the CI's Clang 18 — the macOS `-Werror` leg is still the first exposure to
those. The gap is narrowed from "no Clang has ever seen this code" to "one
Clang has, and had nothing to say".

Reproducing without root:

```bash
python3 -m venv /tmp/zv && /tmp/zv/bin/pip install ziglang
printf '#!/bin/sh\nexec /tmp/zv/bin/python -m ziglang c++ "$@"\n' > /tmp/zv/zigcxx
printf '#!/bin/sh\nexec /tmp/zv/bin/python -m ziglang cc "$@"\n'  > /tmp/zv/zigcc
chmod +x /tmp/zv/zigcxx /tmp/zv/zigcc
cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=/tmp/zv/zigcxx \
      -DCMAKE_C_COMPILER=/tmp/zv/zigcc -DBUILD_TEST=ON -DBUILD_DOCUMENTATION=OFF
cmake --build build-clang -j
```

Do **not** pass `-DCMAKE_CXX_COMPILER_ID=Clang`; forcing it breaks CMake's
feature detection and googletest fails to configure. Let CMake detect it.

### A lead, not a finding: the three exit-time segfaults

Under that build, `CorpusUnitTest`, `FhnCorpusTest` and
`FhnExternalBackendTest` crash — and all three **pass every test first** and
then take SIGSEGV during process exit, after gtest's tear-down line. The same
binaries exit 0 under GCC. All three are the `dlopen` backend tests, which
points at a static destructor running after the backend handle is gone.

Two reasons not to call this a bug in the project yet. Zig's toolchain links
its own runtime, and mixing that with a `dlopen`ed shared object is exactly
where this class of crash appears spuriously. But it is worth writing down
because `FhnExternalBackendTest` is also the test that SIGTRAPs on macOS, and
"destructor ordering around a `dlopen`ed backend" would explain both. If
someone picks up the macOS SIGTRAP, start there. Nothing on the CKKS branches
is affected — those four tests link one static layer each and do not `dlopen`
anything.

### L2 Storage — what is forced, and the one thing that is not

Three designs were produced independently from different starting premises
(view-only, view-plus-owner, pointerless descriptor) and judged against the
tree rather than against each other. The useful result is how little of it
turned out to be open.

**Forced by the repository, and no longer worth discussing:**

| Question | Forced by |
|---|---|
| L2 cannot own arena memory; "arena-allocated RnsPoly" is not an L2 concept | `cmake/FhnCkksLayer.cmake`'s downward-only check — L3 is above L2 |
| L2 depends on `fhn_ckks_params` only, never on L1 | the spec's layering, plus the precedent that a `poly.toNtt(tables)` method is exactly how I3 erodes |
| The wire identity cannot be `Params::hash()` | `Params.cpp:165` — it is `std::hash`-derived, so implementation-defined, and `std::size_t` is not 64 bits everywhere. This is a correctness bug, not a trade-off |
| Capacity must be a separate, checked quantity from live size | `include/FHN/fhn_backend_api.h:76` — `FhnBufferAllocFn` takes **no size**, so a buffer is sized once while the live shape shrinks with level |
| Residues are limb-major | `NttTables.h` — `ntt::forward`/`inverse` need `degree()` contiguous residues for one prime, and the NTT dominates cost |
| Views never own; duplication is a distinctly named operation | invariant I4, verbatim |

**The one genuinely open decision: does L2 export an owning aligned slab type**
(`PolySlab` — owns bytes, knows no shape), or does durable ownership live
elsewhere? Nothing in the repo forces or forbids it. The argument that an
owning L2 type violates I2 does not survive contact with
`src/CKKS/arena/Arena.cpp:5`, where `Arena`'s own constructor calls
`::operator new` — a reading of I2 that condemns `PolySlab` condemns `Arena`.

**Settled by both sessions, 2026-08-23: yes, with a sharper reading of I2 than
either of us started with.** I2 is not about who calls `operator new` — it is
about who decides *how much, and when*. `Arena` satisfies it because its
capacity is an explicit constructor argument that never grows implicitly, not
because it avoids allocating. An L2 slab satisfies it the same way: explicit
size, no implicit growth, no default allocator, no ambient anything. Stated
that way the invariant is about the absence of hidden policy, which is what it
was always for, and the "must receive already-allocated storage" gloss that
seemed to follow from it does not.

The house had also decided this shape of question once already: the NTT moved
from L4 to L1 because placing it above "would have forced this layer's test to
reach upward for something it does not use". The same rule says an aligned
allocator that the L2 test needs to exercise L2's own contract belongs at L2.
Saying no would relax or drop the 64-byte alignment `Arena::kAlignment`'s own
comment promises to kernels, and push the durable owner somewhere needing a
*larger* amendment to the L3 line.

Still worth the user's sign-off, because it adds one clause to the spec's L2
line — but it is no longer an open technical question.

### The L1/L4 transposition — raised, then dissolved

`bconv::convert` consumes **one coefficient's residues across the whole source
basis** (`in[i]` indexes source primes), while storage must be limb-major
because `ntt::forward` needs `degree()` contiguous residues per prime. Read
naively those do not compose, and ModUp appears to need a gather of `k`
residues and a scatter of `l` per coefficient, `N` times.

That was wrong, and the fix is a loop order, not a buffer. Spark's shape:

```
for i in from:
    for n in 0..N:  t[n] = reduce(lazy(in_limb[i][n], hat_inv[i], q_i))   // contiguous
    for j in to:
        w = hat_res[i][j]                                                 // inner-loop invariant
        for n in 0..N:  out_limb[j][n] += lazy(t[n], w, p_j)              // contiguous
```

Verified bit-identical to the shipped per-coefficient path for
`(k,l,N)` in `{(3,2,17), (5,3,32), (8,4,9), (2,1,64)}`, with both agreeing
with an exact big-integer CRT oracle. Three things follow, and they matter
beyond tidiness:

- **Scratch is exactly one limb (`N` words), not `k*N`**, because `t` is reused
  for each `i`. So `eval::scratch_bytes(FHN_MOD_UP, ...)` counts one limb; it
  never needs a transposition buffer, and an estimate that budgets one is too
  large rather than too small.
- **The accumulation order over `i` is unchanged**, so the 2p window analysis
  in `BConvTables.cpp` carries over untouched. This is the reason to prefer
  this shape over any that reassociates the sum.
- **The Shoup multiplier `w` is invariant in the innermost loop**, which is the
  shape both SVE2 and AVX-512 want. The transposition was an artefact of the
  single-coefficient API, never intrinsic — which also means it was an API
  decision on the Cheddar side rather than a kernel one.

The batched entry point itself waits until L2 is real, because only the storage
layout fixes how a limb plane is addressed. Both sessions agree on that, and on
the shape to write when the time comes.

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

**Open the conversation with:** decision (6), whether an empty key-switching
digit should be rejected. The next slice after L1 is RNS base conversion →
`ModUp`/`ModDown` → hybrid key switching, and (6) is the one pending decision
that touches it: a `dnum` that silently means something smaller than the
caller asked for is exactly the hidden discrepancy I4 exists to prevent, and
rejecting it is a one-line change that gets harder to make once key switching
reads `digitRange()` in anger.

Decisions (2) and (3) — cryptographic scope, and which open-source library
gets wrapped as the baseline — remain open and neither blocks that slice.
(3) does block the first performance number, per locked decision 5.

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
