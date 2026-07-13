# Level-Aware Byte Budgets — Design

Date: 2026-07-13
Status: approved (design discussion in session; slice choice and
level-rules placement each confirmed explicitly)

## Premise

`FhnMovementPlan` budgets are currently counted in buffer slots because
nothing tells the planner how many bytes a ciphertext occupies. In
CKKS-family schemes that size is a function of the ciphertext's *level*,
and levels change as instructions execute — so byte-accurate budgeting
needs (a) a way for backends to declare their size-versus-level model and
per-opcode level effects, and (b) level inference in the planner.

Two standing decisions shape this design:
- **CKKS-first (TFHE deferred)** — the model targets leveled arithmetic
  schemes; boolean/TFHE semantics are out of scope until that port lands.
- **Backends declare their own level semantics** (user decision): the
  per-opcode level-effect table is an ABI export, not a convention
  hardcoded in the pass. The planner stays backend-agnostic by consuming a
  caller-built model object, never calling the ABI itself.

## Goals

- Backends can declare fresh level, bytes-per-level, and per-opcode level
  effects through three new OPTIONAL ABI exports (all-or-nothing).
- `FhnMovementPlan::analyze` gains an optional level model; with a model,
  `device_budget` means BYTES; without one, existing slot-count behavior
  is bit-for-bit unchanged.
- Level inference doubles as validation: level underflow (too many
  CONSUME steps for the parameter chain) and invalid LEVEL_DOWN targets
  are caught at plan time instead of exploding inside a backend.
- ToyFHE exports a flat model so the byte path is CI-executable
  end-to-end; Cheddar declares its real RNS sizes (GPU-machine manual
  validation).
- The corpus grows a `--budget-bytes` mode — the first piece of the
  README's "memory estimates" bench vocabulary.

## Non-goals (v1)

- Pre-relinearization 3-polynomial ciphertext size (transient within
  fused/decomposed sequences; documented limit).
- Latency/throughput estimates, rotation-key metadata, batchability —
  later slices of the kernel catalog.
- TFHE semantics (deferred by standing decision).
- Session-path byte budgets: Session keeps `device_budget = 0`
  (unlimited) and passes no model; byte mode is exercised by tests and
  the corpus until real device pressure exists.

## Component 1 — ABI additions (`fhn_backend_api.h`)

```c
/* ── Optional level model (data plane) ──
   CKKS-family backends declare how ciphertext size varies with level and
   how each opcode moves levels, enabling byte-accurate movement planning.
   All three exports are optional but must appear TOGETHER: a backend
   exporting only a subset has the whole group ignored with a warning
   (mirroring the movement-hook half-pair rule). Additive and optional —
   no FHN_ABI_VERSION bump; every existing backend remains conformant. */

typedef enum FhnLevelEffect {
  FHN_LEVEL_PRESERVE = 0,   /* result level = min over operand levels   */
  FHN_LEVEL_CONSUME = 1,    /* result level = (min over operands) - 1   */
  FHN_LEVEL_SET_PARAM0 = 2, /* result level = params[0] (must not raise) */
} FhnLevelEffect;

/* Level assigned to freshly encrypted inputs. */
typedef int64_t (*FhnFreshLevelFn)(FhnBackendCtx *ctx);
/* Device bytes of a ciphertext at `level`; 0 = invalid level. */
typedef uint64_t (*FhnLevelBytesFn)(FhnBackendCtx *ctx, int64_t level);
/* Level effect of one compute opcode. */
typedef FhnLevelEffect (*FhnOpcodeLevelEffectFn)(FhnBackendCtx *ctx, FhnOpCode opcode);
```

Export names: `fhn_fresh_level`, `fhn_level_bytes`,
`fhn_opcode_level_effect` (prefix rules as usual).

`FhnRuntime` (Backend.h) and the corpus `CorpusBackend` loader gain the
three nullable members/getters, resolved with the all-or-nothing rule.

## Component 2 — `FhnLevelModel` and planner extension

`include/FHN/FhnMovementPlan.h`:

```cpp
// A backend's declared level semantics, pre-queried by the caller (the
// planner never touches the ABI). bytes_by_level is indexed by level,
// size fresh_level + 1; effects covers every opcode the program uses.
struct FhnLevelModel {
  int64_t fresh_level = 0;
  std::vector<uint64_t> bytes_by_level;
  std::unordered_map<int, FhnLevelEffect> effects; // key: FhnOpCode
};
```

`analyze` gains a trailing optional parameter:

```cpp
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                uint32_t device_budget = 0,
                                                FhnEvictionPolicy policy = FhnEvictionPolicy::Belady,
                                                const FhnLevelModel *model = nullptr);
```

Semantics with `model != nullptr`:

- **Level inference.** Input ids start at `model->fresh_level`. Each
  instruction's result level follows the declared effect (PRESERVE →
  min over operand levels; CONSUME → min − 1; SET_PARAM0 → `params[0]`).
- **Validation (nullopt):** an opcode missing from `effects`; a CONSUME
  result below level 0 (level underflow — the program exceeds the
  parameter chain); SET_PARAM0 with `params[0]` negative, above
  `fresh_level`, or above the min operand level (levels never rise);
  `bytes_by_level` smaller than `fresh_level + 1` or containing a 0.
- **Byte accounting.** `device_budget` is interpreted as bytes.
  Residency cost of id = `bytes_by_level[level(id)]`; the resident-set
  total, incoming-cost check, and infeasibility test (one instruction's
  working-set bytes > budget) all switch to byte sums. Victim selection
  (Belady/LRU) is unchanged — the policies choose *which* to evict; bytes
  only decide *how much* must go.
- **Stats:** new field `high_water_bytes` (0 in slot mode). Existing
  count-based `high_water` keeps its meaning in both modes.
- `model == nullptr` (all existing callers): behavior bit-for-bit
  unchanged, including `device_budget` as a slot count.

## Component 3 — Backend implementations

- **ToyFHE** (`src/FHN/ToyFheKernels.cpp`): exports a flat, honest model —
  `fresh_level = 0`, `level_bytes(0) = sizeof(toyfhe::Ciphertext)`,
  every compute opcode `PRESERVE` (its Engine handles rescale
  internally). Purpose: the byte path becomes CI-executable end-to-end,
  and the level-model exports get positive dlsym coverage (unlike the
  movement hooks, which still lack an exporting backend).
- **Cheddar** (`src/FHN/cheddar/CheddarFhnBackend.cpp`): declares
  `fresh_level` from its parameter chain, per-level RNS sizes from its
  own layout (formula implementation-defined by the backend — that is
  the point of the export), CONSUME on rescale-class ops per its kernel
  behavior. Compile-guarded like the rest of the file; validated manually
  on a GPU machine (no CI).
- **Partial-export fixture** (`test/`): a tiny stub library exporting the
  ABI handshake + required data plane + ONLY `fhn_fresh_level` — pins the
  all-or-nothing rule end-to-end in both loaders (this also retires part
  of the long-standing "negative-path ABI tests via stub fixtures"
  backlog item).

## Component 4 — Corpus integration

`fhn-corpus` gains `--budget-bytes N`:

- Requires a loaded backend that declares the level model (else: stderr
  error, exit 2).
- Per shape, one extra report line:
  `movement-bytes[<shape>]: hw_bytes=<unlimited-run bytes high-water> belady p/e=<..> lru p/e=<..> (budget-bytes N)`
  computed by running both policies at the byte budget with the queried
  model. The existing slot-count sweep and aggregate stay untouched.
- The FhnCorpusTest CI invocation appends a byte-budget exercise against
  ToyFHE (flat model) at `--budget-bytes` = `3 × level_bytes(0)`: corpus
  generators emit only unary/binary primitives, so every instruction's
  working set is at most 3 buffers, making this the minimum feasible
  budget — maximal eviction pressure on every shape whose liveness
  high-water exceeds 3, while the runner's infeasibility error keeps a
  wrong constant loud rather than silent. The runner computes it from the
  queried model at startup rather than hardcoding a byte number.

## Verification summary

- Plan unit tests with synthetic models: level inference across PRESERVE/
  CONSUME/SET_PARAM0 chains; underflow → nullopt; raise-attempt →
  nullopt; missing-effect → nullopt; byte-mode eviction under a byte
  budget (construct differing per-level sizes so a single big buffer
  eviction frees more than two small ones — assert byte accounting, not
  count, drives the loop); slot-mode regression: existing tests untouched
  and green; `high_water_bytes` correctness.
- Loader tests (both `ExternalBackend` and `CorpusBackend`): ToyFHE
  resolves all three (positive path); partial fixture → all three nulled
  with warning.
- Corpus: `--budget-bytes` line present and deterministic; error path
  exits 2 without a model.
- Full suite green, ASan/UBSan clean, both clang-format versions clean.

## Risks / honest limits

- Bootstrap, precisely: the enum could mechanically express it
  (SET_PARAM0 with a raised target, or an additive FHN_LEVEL_REFRESH
  value — no version bump either way), but this spec deliberately
  outlaws level raises as a plan-time bug-catcher, correct for the
  bootstrap-free CKKS-leveled scope. The real prerequisites for
  bootstrap support live elsewhere: (1) FhnOpCode has no BOOTSTRAP
  opcode (adding one renumbers → FHN_ABI_VERSION bump), and (2) this
  model accounts only operand/result buffer bytes — bootstrap's
  footprint is dominated by kernel-internal scratch peaks, which need a
  per-kernel scratch-bytes field in a later kernel-catalog slice.
  Multi-level drops (CONSUME by k) would be additive enum growth.
- PRESERVE-with-min quietly models cross-level operands that a strict
  backend might reject at execution; the planner's job is memory
  accounting, not scheme legality — execution remains the authority.
- ToyFHE's flat model exercises plumbing, not size variation; real
  variation is validated only when Cheddar runs on a GPU machine.
