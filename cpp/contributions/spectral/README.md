# Complex cross-spectrum kernels

This contribution links as `Fhenomenon::SpectralKernels`, using only installed
public runtime interfaces. The [monitoring example](../../examples/private_spectrum/README.md)
supplies encrypted complex frequency bins from multiple sensors under one key.
It computes $X_i[k]\,\overline{R[k]}$ and carries the sums across finite frame scopes.
The runtime core does not know what a sensor or a spectrum is.

## Commands and reusable state

```cpp
contrib::registerSpectralKernels(runtime, computeServices, nativeLibraryPath);
contrib::SpectralRuntime spectral(runtime);

auto outputs = session->run([&](runtime::Scope &scope) {
  auto [conjugate, prepared] = spectral.prepareReference(scope, scope.input(reference));
  auto crossA = spectral.pair(scope, scope.input(sensorA), prepared);
  auto crossB = spectral.pair(scope, scope.input(sensorB), prepared);
  auto accumulated = native.op(scope, FHN_ADD_CC, {crossA, scope.input(previousSum)});
  return runtime::ScopeValues{prepared, conjugate, crossB, accumulated};
});
```

Here `contrib` and `runtime` abbreviate the `fhenomenon` namespaces. The
reference, sensor and previous-sum values are ciphertexts owned by the same
`Runtime`; the previous sum has the product's level and scale.

| Request | Inputs | Outputs |
| --- | --- | --- |
| `contrib.spectral.prepare-reference` | Reference ciphertext | Ordinary conjugate, sealed preparation artifact |
| `contrib.spectral.pair` | Sensor ciphertext, preparation | Per-slot complex cross-product |

These requests have no attributes. `SpectralRuntime` provides immediate and
scope-recording overloads. Preparation owns the exact conjugated source, its
provider and metadata, and optional Matvec multiplication companions. Copied
`Value` handles share it; copying its opaque payload into a new artifact does
not create a valid preparation. The preparation may be imported into another
scope on the same runtime. A new reference frame needs a new preparation.

Matvec preparation uses width **1** and `FHN_MATVEC_PREPARE_VECTOR |
FHN_MATVEC_PREPARE_KEYS`. Width controls reduction stages, not how many logical
slots receive products: every slot is processed. No rotation or reduction
companions are requested. The key contract is multiplication and conjugation,
available from `nativeSpectralKeyRequirements()`.

The exposed conjugate is a normal native ciphertext. Cross-products are normal
ciphertexts too: ordinary ADD can accumulate them or another contribution can
consume them. Result buffers do not depend on the preparation or the previous
accumulator, so replacing handles does not retain the entire stream history.
Retaining a preparation does retain its source and cached companions.

## Selection and contribution points

`contrib.spectral.pair.prepared` has priority 20 and uses the Matvec product
provider. `contrib.spectral.pair.ordinary` has priority 0 and uses native HMULT
when preparation has no Matvec companions. An empty library path explicitly
selects preparation without Matvec registration. A nonempty path requests the
prepared provider: all preparation failures, including unsupported requests and
memory-budget failures, propagate. This keeps a selected contribution's error
from being mistaken for permission to switch implementations.

`contrib.spectral.scope.pair-tile` covers two to eight adjacent compatible pair
commands that share the same preparation. `contrib.spectral.scope.pair` covers
one. Their executors dispatch each pair through `Runtime::submit`, preserving
higher-priority ordinary recipes and returning every live intermediate in the
scope's required order. This is an orchestration contribution, **not fused
cryptographic arithmetic**, and grouping alone implies no speedup.

To contribute another implementation, register a higher-priority pair recipe or
scope kernel with explicit input constraints. A scope implementation must return
all live outputs of its accepted span, including products consumed by subsequent
ordinary commands. Unsupported shapes should decline during matching; a selected
executor must report failure rather than silently retry another implementation.
See [the tests](../../test/SpectralContributionTest.cpp) for an ordinary override
that remains visible inside a tile, foreign/forged-state rejection, and exposed
complex intermediates reused across scopes.

`ScopeOptions::use_kernels = false` disables span selection while keeping the
same ordinary pair recipes and shared preparations. It is a useful comparison
against scope grouping; the example's separate native baseline also removes the
Matvec preparation while retaining a shared conjugated reference.

## Numerical and ownership contract

Reference and sensor ciphertexts need two components, equal positive levels,
matching logical slot counts and compatible scales (relative tolerance $10^{-12}$).
They must belong to the registered native context and runtime; that context's
single client lifetime supplies one key identity. HMULT consumes one level.
Accumulation requires compatible product metadata and a finite magnitude/error
budget even though each ADD consumes no further level.

This contribution neither decrypts nor holds a client capability. The caller
keeps the trusted client alive for evaluation-key leases and handles local FFT,
synchronization/calibration, encryption and final phase/coherence calculation.
It does not add ciphertext transport, independent-owner key composition,
bootstrapping or level replenishment.
