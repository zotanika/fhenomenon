# Contributed statistics and packed dot kernels

The separately linked `Fhenomenon::StatsKernels` target contributes four named
recipes. It uses the existing public runtime API and optional native CKKS
exports; neither `Runtime` nor `NativeRuntime` needs a new opcode, payload type,
dispatch branch, compiler pass or registry change.

```cpp
#include <FHN/contrib/StatsKernels.h>
using namespace fhenomenon::runtime;

Runtime runtime;
auto services = loadNativeServices(plugin_path);
NativeRuntime native(runtime, services);
fhenomenon::contrib::registerStatsKernels(runtime, services, plugin_path);
// Trusted host separately prepares keys and imports encrypted x in this context.
auto sum = x;
for (std::int64_t distance : {32, 16, 8, 4, 2, 1}) {
  auto rotated = native.rotate(sum, distance);
  sum = native.semantic(FHN_CIRCUIT_ADD, {sum, rotated});
}
auto variance = runtime.submit({"contrib.stats.variance", {x, sum},
                                std::uint32_t{64}}).at(0);
// sum remains independently available as the application's other output.
```

Let $n$ denote `width`, $x_i$ an input slot and $s=\sum_{i=0}^{n-1}x_i$
the corresponding replicated `sum`.

| Recipe | Inputs | Returned canonical ciphertext |
|---|---|---|
| `contrib.stats.energy` | `x`, corresponding `sum` | $(n x_i-s)^2/n$ per slot |
| `contrib.stats.variance` | `x`, corresponding `sum` | Rotate-add reduction of that energy |
| `contrib.packed.dot` | encrypted `a`, encrypted `b` | Rotate-add reduction of $a_i b_i$ |
| `contrib.packed.reduce` | encrypted intermediate | Rotate-add reduction, distances $n/2,\ldots,1$ |

Attributes contain exactly `std::uint32_t width`, a nonzero power of two no
greater than the logical slot count. Partial reductions retain that slot count.
The native C implementation resides in its own translation unit and reuses
existing private region arithmetic. It does not invoke the circuit compiler.
The contribution registration/publication code lives outside the core library
in `contributions/stats/StatsKernels.cpp` and uses public headers only.

## Shared intermediate boundaries

The variance kernel consumes a sum made by ordinary commands; it does not
recompute or privately cache that sum. The caller retains it as another output.
The same computation can expose an intermediate and cross boundaries again:

```cpp
auto energy = runtime.submit({"contrib.stats.energy", {x, sum},
                              std::uint32_t{64}}).at(0);
auto rotated = native.rotate(energy, 32);
auto partial = native.semantic(FHN_CIRCUIT_ADD, {energy, rotated});
auto variance = runtime.submit({"contrib.packed.reduce", {partial},
                                std::uint32_t{32}}).at(0);
```

Here `energy` has two ordinary consumers, and their result re-enters another
contributed kernel. It is an ordinary native ciphertext Value with its actual
level/scale, not an opaque completed-app result. Similarly the original shared
vector in `matvec-fused` feeds eight contributed dot kernels without reimporting
or recomputing the input. A split variant performs the product and first
rotate/add with ordinary commands, then uses the contributed partial reduction.

## Arithmetic and supported envelope

For an input of period $n$ with corresponding replicated sum $s$,

$$
\sum_{i=0}^{n-1}\frac{(n x_i-s)^2}{n}
= n\sum_{i=0}^{n-1}x_i^2-s^2.
$$

This is the existing `stats` variance proxy, with its original four outputs
and independent oracle. Per dataset the new expression requires one tensor
product and one relinearization/rescale, compared with two of each originally.
The total number of rotations is unchanged. Fusing the product/reduction also
removes repeated public buffer allocation/publication and operation boundaries.

The initially considered expression $\sum_{i=0}^{n-1}x_i(n x_i-s)$ has a first-order
cancellation problem under CKKS error: constant $-8$ at $n=64$ exceeded the
unchanged absolute tolerance. The centered-square form passed that regression
without changing tolerance. The benchmark still compares against the original
application oracle, not a newly defined encrypted-input oracle.

Division by $n$ is represented explicitly by the actual output scale

$$
\Delta_{\mathrm{out}}=\frac{n\Delta_{\mathrm{in}}^2}{q_{\mathrm{drop}}},
$$

where $\Delta_{\mathrm{in}}$ is the input scale and $q_{\mathrm{drop}}$ is the
dropped prime. It consumes no extra $Q$ level but uses
additional modulus headroom. The runtime does not force this value to have the
scale of an ordinary normalized product. Rotations and additions of matching
intermediates preserve the scale; incompatible joins are rejected.

Statistics recipes conservatively require level $\ge 2$ (three active $Q$ primes).
The tested statistics envelope is the default native profile: $N=8192$,
$Q$ prime bit widths $(50,40,40)$, $P$ prime bit width $60$, $\mathrm{dnum}=3$,
$n=64$ and input magnitudes $\le8$, with a correctly supplied sum. This admission
is not a precision guarantee for arbitrary parameters, magnitudes or malformed
sum provenance. Packed dot requires level $\ge1$; reduction also supports level 0,
subject to the usual CKKS value-range limits. Metadata cannot establish the
periodic plaintext or sum relationship.

## Ownership and accounting

Recipes retain their native context, adapter and library leases, never a client
or secret key. Applicability examines public capability and ciphertext metadata;
unavailable cases may fall through before execution. A native failure after
selection is reported without retrying another implementation.

Each invocation reserves a native staging ciphertext through `makeValue` before
allocation. After successful native execution, `importCopy` publishes the result
using the unchanged public native adapter. Both full allocations are charged
during this copy; staging is then released. This extra copy and allocation are
included in every timing. The native kernel itself allocates no heap memory and
uses the context's existing scratch. Scratch, evaluation keys and tables are
excluded from the runtime's retained-slab budget, as in the base runtime.

## Validation and measurement

See the [measured results and validation](coarse-statistics-validation-2026-10-31.md)
for the statistics speedup, the inconclusive packed-dot result and exact scope.
The subsequent [matvec investigation](matvec-preparation-validation-2026-10-31.md)
keeps the dot formula fixed and measures explicit reusable preparations through
a separate contribution, with the runtime core unchanged.

`fhn-bench-runtime` retains the original immediate, shared-rotation-preparation
and explicit-region modes. It adds contributed and split modes for `stats` and
`matvec-fused`, plus `energy-only` for statistics: the same stable energy kernel
followed by a fully ordinary reduction. That last comparison separates algebraic
savings from larger-kernel boundary savings.

Every timed iteration includes ordinary sum construction, contribution staging,
publication, dispatch, arithmetic and temporary release. Initial input import,
provider registration, optional preparation, encryption/decryption and final
output release are outside warm timing. Registration, input import, image load,
graph import and context/key/encryption setup are reported separately. All modes
register the same provider set and use identical ciphertexts/keys. Compiler
entrypoints are absent in every mode except the explicit region reference.

The benchmark checks every logical output slot against the unchanged corpus
oracle. Native tests additionally encrypt/check all 4096 physical slots with
period-64 corpus and nearconstant data; separate constant regressions check all
64 logical slots. Installed-package
tests link the separate contribution target and exercise both shared-sum and
split-intermediate execution using installed public interfaces.
