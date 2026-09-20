# Built-in RNS-CKKS Backend — Architecture Design

Date: 2026-07-26
Status: draft — the requirements (R1–R5) and the invariants they imply
(I1–I5) are fixed. Cryptographic scope (leveled vs bootstrapping),
parameter selection, and implementation staging are deliberately NOT
decided here.

## Premise

The built-in backend today is ToyFHE: a single-slot scalar toy scheme
(`include/Crypto/ToyFHE.h` — `Ciphertext{int64 c0, int64 c1}`), no ring, no
NTT, no packing, documented in its own header as "deliberately insecure".
It exists to make FHN semantics observable, and it does that job.

Replacing it with a real RNS-CKKS implementation cannot be justified by the
operation set. Every CKKS library exposes the same one — add, mult, tensor,
relin, rescale, rotate, conjugate, key-switch. That interface is commodity.
Reimplementing a commodity interface buys nothing.

The justification is structural, and it comes from the programming model.
An FHE library written to serve a fixed, hand-written workload — a
benchmark, a paper's evaluation circuit — can afford to bundle parameters,
precomputation, memory, and evaluation into one object, because there is
exactly one of each for the lifetime of the process and the call sequence
is known to the author. Fhenomenon does not have that luxury: it executes
*compiled programs* it did not write, plans their memory before running
them, decomposes and fuses their opcodes, and must hold more than one
backend and parameter set live at once.

**This document specifies a backend shaped by those requirements.** The
invariants below are not general software-engineering preference; each one
is derived from something the FHN model actually needs and cannot get
otherwise.

## Requirements the FHN model imposes

| | Requirement | Where it comes from |
|---|---|---|
| **R1** | Memory must be **plannable before execution**. | `FhnMovementPlan` — "FhnPrograms are straight-line and data-oblivious, so the full def-use chain of every buffer id is exact at plan time." A backend whose allocation is ambient cannot report what a program will need, so the byte-budget planner has nothing to budget against. |
| **R2** | Operations must **compose and decompose freely**. | `FhnDefaultExecutor` lowers `FHN_HMULT` into mult + relin + rescale when a backend exposes only primitives, and issues in-place calls (`result` may alias an operand, per the ABI). Operations carrying hidden state or requiring a specific call order cannot be rearranged by a compiler. |
| **R3** | Key material must live **outside the compute path**. | `fhn_backend_api.h` — key operations "are not kernel-table entries and cannot be reached from an FhnProgram: the instruction stream the executor dispatches is public and compute-only." A container that owns both keys and evaluation cannot express this split. |
| **R4** | Multiple parameter sets and backends must **coexist in one process**. | `ExternalBackend` `dlopen`s backends alongside the built-in one; the corpus compares backends in a single run; a future bootstrapping path needs sparse and dense secrets live together. Any process-global state — allocator, degree, table cache — makes this impossible. |
| **R5** | Level semantics must be **declarable as data**. | The `fhn_fresh_level` / `fhn_level_bytes` / `fhn_opcode_level_effect` trio. Byte-accurate planning needs ciphertext size as a function of level, which requires level to be a property of the ciphertext, not of the session. |

## Why the conventional shape cannot meet them

The failure is not that monolithic libraries are badly written. It is that
a single container owning parameters, precomputation, memory, and
evaluation *necessarily* violates R1 and R4, and makes R2 and R3 a matter
of discipline rather than construction.

### Measured: Cheddar (`refs/cheddar-fhe`, this repo's GPU backend)

Read directly on 2026-07-26. The container leaks into process-global state
on four distinct paths.

| Evidence | Consequence |
|---|---|
| `include/core/Context.h:88` — `param_`, `memory_pool_`, `elem_handler_`, `ntt_handler_`, `mod_switch_handlers_`, `encoder_` are public members under the comment `// The order matters here.` | The dependency graph is encoded in member declaration order. `encoder_(param_, ntt_handler_)` and `mod_switch_handlers_.emplace_back(param_, level, elem_handler_, ntt_handler_)` bind references to *sibling members*, making the object one non-relocatable knot. |
| `include/core/MemoryPool.h:11` — "After the creation of a MemoryPool object, **all** memory allocations on the current device uses binning_memory_resource." | Constructing the container mutates the process-global RMM resource. `DeviceVector : rmm::device_uvector`, so every ciphertext allocation implicitly depends on which `Context` was constructed last. **Violates R1 and R4.** |
| `src/core/Context.cpp:76` — `Container<word>::SetDegree(param_.degree_)` writing `static inline int degree_` (`include/core/Container.h:19`) | Data types do not know their own size. `Ciphertext::bx_(np.GetNumTotal() * Base::degree_)` reads a mutable global. **Two parameter sets are impossible in principle — R4 fails outright.** |
| `src/core/Context.cpp:77` — `MultiLevelCiphertext<word>::StaticInit(...)`, holding `static inline std::vector<Constant<word>> level_down_consts_{}` with the comment `// different from the one in Context` | The same precomputation is duplicated into a second global because there was no compositional way to share it. This is the terminal symptom, not the cause. |

The downstream consequences all take the same form — performance is
recovered by making the *caller* responsible for decisions the system should
own:

- `Ciphertext` is move-only and **cannot duplicate itself**. Copying requires
  `Context::Copy(Ct &res, const Ct &a)` — the container. Inevitable once size
  lives in a global and allocation lives in an ambient pool.
- `include/core/Context.h` — "**Unsafe** functions are used for performance
  reasons and they only work for specific cases. **Do not use them unless you
  know what you are doing.**" (`MultUnsafe`, `MadUnsafe`, `MultKeyNoModDown`,
  `AddLowerLevelsUntil`). A compiler cannot select a kernel whose
  precondition is "the author knows what they are doing".
- `include/extension/Hoist.h` — every method of `HoistHandler`, constructor
  included, threads `ConstContextPtr`, and performance depends on the caller
  setting `min_ks` / `suppress_bs_swap` / `inplace` correctly.
- `include/core/Parameter.h:11` —
  ```cpp
  // THESE ARE THE PARAMETERS FOR SENSITIVITY STUDIES
  // Best performance should be achieved with all true;
  constexpr bool kMergePMult = true;  // + kMergeCMult, kOptimizeAutomorphism,
  // kFuseMontgomery, kFuseModDownEpilogue, kFuseGSPAccum, kFuseBSKeyMult, kExtendedOT
  ```
  The optimisations are compile-time global flags for producing ablation
  tables — not strategies a runtime can select per program.

Cheddar gets three things right, and they are carried forward: keys are
passed explicitly as `const Evk &` and never stored in the container;
`DvView`/`DvConstView` separate non-owning access from ownership; and
`EvkRequest` lets a circuit *declare which keys it needs* before any are
generated.

### The general pattern

Two failure modes recur across libraries of this shape and are worth naming,
because avoiding one does not avoid the other:

1. **Removing a manual knob is not removing the cost.** Replacing a
   user-chosen copy depth with an unconditional deep copy is an improvement
   in safety and a regression in predictability if an innocuous-looking
   accessor now allocates. The goal is not "no knobs" — it is that expensive
   operations are *spelled differently* from cheap ones, so a caller (or a
   compiler pass) can tell them apart without reading the implementation.
2. **Layering that the build does not enforce erodes.** A clean, acyclic
   header dependency graph maintained by review discipline over a single
   build target has nothing to stop a layer-skipping include or a link-time
   cycle from appearing, and no way to test a layer alone.

This repository is currently in the same position on point 2:
`src/CMakeLists.txt` uses `add_fhenomenon_sources()` to gather every source
into one `add_library`. The enforcement described in Component 7 does not
exist here yet either — it has to be built, not assumed.

## What FHN already gets right

`include/FHN/fhn_backend_api.h` already draws the boundaries at the ABI edge:

- the kernel signature is `(ctx, result, operands, params, fparams)` with
  "Buffer ownership: … **Kernels never allocate or free buffers**";
- key material is unreachable from the kernel table (R3);
- movement (`prefetch`/`evict`) is "deliberately NOT kernel-table opcodes";
- the level model is a separate, declared, all-or-nothing export trio (R5).

The gap is that a backend can honour every one of these at the ABI edge and
still open a monolithic container immediately behind the opaque
`FhnBackendCtx *` — which is what `CheddarFhnBackend` necessarily does,
since it wraps `cheddar::Context`. The built-in CKKS backend exists to be
the first one where the split holds all the way down.

## Invariants

Each invariant is derived from a requirement above, and each is mechanically
checkable.

- **I1 — Operands are self-describing.** *(R2, R4, R5)* A ciphertext,
  plaintext, or key carries everything needed to interpret it: RNS basis,
  degree, level, scale, NTT state. No operation consults ambient state to
  understand its inputs. This is the invariant that makes a session container
  unnecessary; every other invariant is easier to hold once it is true.
- **I2 — Allocation is a parameter, never ambient.** *(R1, R4)* No global
  allocator, no singleton getter, no "current" anything, no process-wide
  install at construction. Any function that needs memory receives it. There
  is no API to set a default, because there is no default.
- **I3 — Evaluation is free functions over data.** *(R2)* No class owns the
  operations, so there is nowhere for hidden state to accumulate and no
  back-pointer to a container. `eval::` contains functions, not types.
- **I4 — Cost is visible in the type system.** *(R1, R2)* Ownership transfer,
  borrowing, and duplication are distinct types or distinctly named
  operations. An expensive operation may not share a spelling with a cheap
  one. A `View` never owns. Nothing named like a getter allocates.
- **I5 — Layer boundaries are link-time facts.** *(all)* One build target per
  layer, dependencies one-directional, each layer buildable *and testable*
  alone. A wrong include is a build failure, not a review comment.

## Layering

Dependencies point downward only. Each layer is constructible and testable in
isolation — testing the NTT must require no keys, no allocator singleton, and
no layer above it.

```
L0  Params      Immutable value type. N, prime chain, dnum, scale, level config.
                Hashable, serializable, copyable, no pointers, no cache, no
                allocation. Two Params coexist trivially.

L1  Tables      Pure derived precomputation, keyed by the identity of what it
                precomputes. Immutable after build, shared as const across
                threads. Split by concern, not bundled:
                  NttTables(prime, logN)     BConvTables(basis_from, basis_to)
                  EncodeTables(logN)         GaloisTable(step)
                Built lazily per what is actually requested.

L2  Storage     RnsPoly / Ciphertext / Plaintext = data. (level, scale, ntt
                state, RNS basis identity, coefficient span). No context
                pointer, no allocator handle, no global. Serializable as bytes.
                Handles never own. The one owner, a slab of words, takes its
                size explicitly and never grows: I2 is satisfied by the
                absence of hidden policy, not by refusing to allocate.

L3  Arena       Caller-owned scratch. Bump allocator over one planned slab.
                Passed by reference to anything that needs temporaries.

L4  eval::      Free functions. eval::hmult(out, a, b, key, tables, arena).
                No hidden allocation, no ambient lookup.

L5  Keys        SecretKey is host-only and never reachable from a kernel.
                EvaluationKey is passed explicitly to the ops that need it.

L6  Backend     Composition root behind FhnBackendCtx *. Params + Tables handle
                + per-thread Arena + key set. Cheap to construct and destroy.
```

## Component 1 — `Params` (L0)

A copyable, hashable, immutable value type. Not a base class, not a reference
held by others, not movable-only.

Rejected by evidence: Cheddar's `Parameter` is const-member-based and
movable-only, so `Context` must hold `const Parameter &param_` — a reference
the caller is obliged to outlive. A parameter set should be a value that can
be copied into whoever needs it, with no lifetime coupling in either
direction.

Level semantics live here, so the R5 export trio becomes a pure function of
`Params` rather than a hand-maintained table. ToyFHE currently declares a flat
one-level model; this is the first backend able to declare a real one on CPU.

## Component 2 — `Tables` (L1)

Precomputation is keyed by the identity of the precomputation — e.g.
`(algorithm, log_degree, modulus)` for an NTT table — not by "the session".
Tables are immutable once built and shared as `const` across threads, so a
table cache is a memo table, never mutable session state. Owned by the
composition root (L6) and passed down; not a process-wide static, because a
cache nobody owns can only be flushed globally and can never be freed
deterministically (R4).

Tables are built **per request, not per level range**. Cheddar builds
`mod_switch_handlers_` for every level `0..max_level` and `level_down_consts_`
for every level in its constructor, regardless of what the program uses.
Galois tables in particular must be built only for the steps a program
actually rotates by — Cheddar's `EvkRequest` idea, applied one layer lower.

## Component 3 — `Arena` (L3): planned, not pooled

This is the component that has no counterpart in a conventional library,
because it exploits something only the FHN model provides (R1).

`FhnMovementPlan` already establishes that an FHN program's full def-use chain
is exact before execution. The same reasoning extends to scratch: the temporary
memory an opcode needs is a pure function of its parameters and level.

```cpp
// L4, pure, no allocation, no I/O.
size_t eval::scratch_bytes(FhnOpCode op, const Params &p, int level);
```

Given that, a program's arena size is `max` over its instructions, computed
once. Then:

- zero allocation during execution — no pool, no fragmentation, no lock;
- the exact memory footprint is a **number reported before execution**, which
  is precisely what the byte-budget planner needs;
- thread scaling is one arena per thread, with no evaluator cloning;
- capacity reuse is the normal case rather than an unreachable fast path,
  because scratch is caller-held by construction;
- teardown is deterministic — no process-lifetime allocator whose destructor
  has to be disabled because it cannot safely free at static teardown.

The claim to measure first: this backend can answer "how many bytes will this
program need?" before running it. A backend built around an ambient allocator
cannot, and that is the falsifiable difference.

## Component 4 — `eval::` (L4)

Free functions, uniform shape:

```
eval::<op>(out, inputs..., keys..., tables, arena)
```

This is not a new convention here — the FHN kernel table is already
free-function-shaped (`src/FHN/ToyFheKernels.cpp` registers static functions
with a uniform signature). What changes is the layering underneath.

Multiplication is not hidden: `tensor`, `relin`, and `rescale` are separate
operations, and fused forms (`FHN_HMULT`) are explicit opcodes the executor
may select — not a default that silently manages levels behind the caller's
back. This is R2 restated: the compiler chooses the fusion, so the fusion must
be nameable.

Performance variants must not be a parallel universe of `*Unsafe` functions
the caller has to know about, nor compile-time global flags. A fast path with
narrower constraints is *a different kernel with declared constraints* —
which is exactly what the FHN kernel catalog is for.

## Component 5 — Keys (L5)

Three-tier ownership, enforced by type:

- the **caller owns** keys (owning handle);
- an evaluation path **observes** them (non-owning handle, checked, throwing
  on expiry);
- an operation **borrows** them for the duration of the call (raw reference).

An evaluation path must not be able to extend a key's lifetime — that is R3
expressed in the type system rather than in a comment.

Cheddar's `EvkRequest` — a circuit declaring which rotation keys and levels it
needs before any are generated — is adopted and generalised: since an
`FhnProgram` is fully known ahead of execution, the required key set is
*derivable from the program*, not requested by hand.

Rejected: a key collection that publicly derives from `std::unordered_map<int,
Key>` with magic sentinel indices (`11111111`, `-22222222`, …) sharing one
integer keyspace with rotation steps.

## Component 6 — the composition root (L6)

`FhnBackendCtx` is a bundle of references, not an owner of everything: a
`Params` value, a handle to shared `Tables`, per-thread `Arena`s, and a key
set. It should be short enough that nobody is tempted to add "just one more"
member, and cheap enough to construct per test.

The name is fixed by the ABI. The type behind the opaque pointer is ours to
keep small.

## Component 7 — mechanical enforcement (I5)

Every structural failure surveyed was a convention that eroded. Discipline
does not survive contact with a deadline; a link error does.

1. **One CMake target per layer**, linked strictly downward. A wrong include
   fails to build.
2. **`eval::` contains no classes** — free functions only. If there is nowhere
   to put state, state does not accumulate.
3. **No allocator-returning free function exists.** Not a default, not a
   getter. `Arena &` is a parameter or the code does not compile.
4. **Each layer's tests link only that layer's target.** If `NttTest` needs to
   link key generation, the layering has already failed and CI says so.
5. **RNG is injected**, never a `thread_local` global — reproducible tests and
   per-tenant randomness isolation both depend on it.

## Goals

- The built-in backend is a correct, secure, full-RNS CKKS implementation that
  honours I1–I5 internally, not only at the ABI edge.
- Ciphertext size varies with level for real, so the byte-budget movement
  planner and the Belady-vs-LRU corpus run on a genuine level model on CPU,
  with no GPU dependency. (ToyFHE declares a flat one-level model, which makes
  that analysis vacuous today.)
- An FHN program's exact memory footprint is reportable before execution.
- Any layer can be built and tested without the layers above it.
- "Competitive" is a measured claim recorded in `experiments/`, against an
  established open-source library wrapped behind the same FHN ABI — never an
  assertion.

## Non-goals (v1)

- Bootstrapping. Its parameter surface (terminal primes, sparse Hamming
  weight, sparse-secret encapsulation) is kept representable in `Params` so it
  can be added without re-architecting, but no bootstrapping component is in
  scope here.
- GPU execution. That is Cheddar's role; this backend is CPU.
- Replacing ToyFHE. It stays as the CI and backend-author reference — being
  small enough to read in one sitting is its entire value.
- Beating any specific library at any specific benchmark. Parity claims come
  after correctness, and only with numbers in the experiment track.

## Open questions

1. **Cryptographic scope** — leveled-only first, or bootstrapping declared as
   the target from the start? Affects parameter design, not the invariants.
2. **Key-switching variant.** Hybrid (GHS + RNS digit decomposition with
   special primes `P`, `dnum` digits) is the assumed choice; it is the most
   expensive decision to reverse and should be confirmed before any code.
3. **The baseline backend.** Which open-source library gets wrapped behind the
   FHN ABI as oracle and speed reference. Since the interface is commodity,
   this is cheap — and without it, any performance claim is unfalsifiable.
4. **Prime width.** 50–60-bit primes are conventional on CPU; Cheddar uses
   ~30-bit for GPU. This interacts with the modmul strategy on aarch64
   (`umulh`-based Shoup/Barrett; SVE2 is available on this machine).
5. **Where `Arena` sizing lives.** `eval::scratch_bytes` is proposed at L4, but
   the program-level maximum could be computed by `FhnMovementPlan` instead,
   which already walks the instruction stream and infers levels.
