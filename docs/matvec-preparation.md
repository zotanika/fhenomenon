# Prepared matvec without changing the computation

`Fhenomenon::MatvecKernels` is a separately linked contribution. It registers
four recipes through the existing public runtime API; `Runtime`, `NativeRuntime`
and the corpus computation remain unchanged.

```cpp
#include <FHN/contrib/MatvecKernels.h>
using namespace fhenomenon::runtime;
Runtime runtime;
auto services = loadNativeServices(plugin_path);
NativeRuntime native(runtime, services);
fhenomenon::contrib::registerMatvecKernels(runtime, services, plugin_path);
// The trusted host prepares evaluation keys and imports encrypted x and rows.
auto prepared = runtime.submit({"contrib.matvec.prepare", {x},
    fhenomenon::contrib::MatvecPrepareRequest{8}}).at(0);
auto y0 = runtime.submit({"contrib.matvec.dot", {x, rows[0], prepared}, {}}).at(0);
auto y1 = runtime.submit({"contrib.matvec.dot", {x, rows[1], prepared}, {}}).at(0);
```

Each dot performs the existing normalized ciphertext product and rotate-add
tree at distances 4, 2, 1. It consumes the same level and preserves the original
modular residues, scale and rounding decisions. No algebraic rewrite, compiler
request or hidden graph search is involved.

## What is reused

The immutable preparation owns a snapshot of the shared source and, according
to its flags, companion tables for fixed multipliers and rotation index maps.
A companion contains the quotient needed by the existing Shoup modular
multiplication primitive. Canonical normalization gives the same residue as
general modular multiplication, while avoiding division in the coefficient
loop. The prepared source and evaluation-key coefficients are reused by
multiple rows and separate requests.

| Flag | Retained preparation |
|---|---|
| `0` | Source snapshot and weak key identity bindings only |
| `FHN_MATVEC_PREPARE_KEYS` | Additionally, immutable evaluation-key companions |
| `FHN_MATVEC_PREPARE_VECTOR` | Additionally, shared-vector companions |
| `FHN_MATVEC_PREPARE_MAPS` | Additionally, rotation permutation maps |
| `FHN_MATVEC_PREPARE_ALL` | All three; default |

The key flag also selects fixed-scalar Shoup arithmetic for P-lift/mod-down
and fused rescale. Those few scalar quotients are computed per invocation,
outside coefficient loops, rather than retained in the artifact. Consequently
its benchmark is explicitly **key companions plus per-call scalar hoisting**;
it does not isolate the gain from retained key state alone. The vector-only
and map-only modes isolate those retained components.

## Intermediate values remain ordinary values

```cpp
auto product = runtime.submit({"contrib.matvec.product",
    {x, rows[0], prepared}, {}}).at(0);
auto rotated = native.rotate(product, 4);
auto partial = native.semantic(FHN_CIRCUIT_ADD, {product, rotated});
auto dot = runtime.submit({"contrib.matvec.reduce", {partial, prepared},
    std::uint32_t{4}}).at(0);
```

`product` is an ordinary native ciphertext, independently usable by other
commands. The final request reuses the preparation's key companions and maps
for the remaining distances 2, 1. A product produced entirely by the existing
ordinary multiply command can also enter `contrib.matvec.reduce` at width 8.
The caller specifies the reduction boundary; the runtime does not infer it.

Preparation requires a canonical two-component source at level >=1. Width must
be a power of two within its logical slots. A row must match source context,
basis, logical slots and scale. Reduction accepts the matching product level
and scale and preserves the actual intermediate metadata. This deliberately
bounded contract does not accept arbitrary mixed-scale matvec inputs.

## Ownership and limits

The contribution binds the source Value identity; a fresh same-shaped Value
cannot silently replace it. The native preparation is an independent snapshot,
so changing or freeing the original native source does not change the snapshot.
Weak key identities prevent preparation from retaining the client or its keys.
Every consumer obtains current matching key leases for its duration; expired
or replaced identities fail. Public wrappers retain context/library services.
Direct C callers must destroy preparation before its context and serialize
calls using that context, as for the other native APIs.

The exact coefficient/map slab estimate is reserved before allocation. Runtime
budgets count the retained artifact, source dependency and full staging/output
buffers. Final publication still uses unchanged `NativeRuntime::importCopy`;
both staging and copy cost are included in the timings. Native consumers use
existing context scratch and allocate no heap. Metadata, allocator overhead,
keys, tables and context scratch are outside the retained-slab accounting.

This first implementation owns companions per preparation. It does not dedupe
the same key tables across distinct prepared sources. The measured width-8
default artifact is several MiB; applications with many live prepared vectors
must account for that cost explicitly.

## Validation and measurement

The native/private tests compare every live residue against the unchanged
generic kernels across levels, key decomposition counts and preparation flags.
Provider tests check source identity, memory admission, lifecycle and ordinary
command boundaries. The installed consumer links the exported contribution
target using public headers only.

`fhn-bench-runtime NATIVE_LIBRARY REPS matvec-reuse` runs the unchanged encrypted
8×8 corpus app with the generic paths, preparation ablations and both split
variants. Every warmup and timed execution checks all 64 logical output values
against the original independent oracle. All modes share ciphertexts and keys;
only the explicit region reference has compiler entrypoints available. Setup,
preparation, retained bytes and warm complete-app samples are reported
separately. See the [measurement report](matvec-preparation-validation-2026-10-31.md).
