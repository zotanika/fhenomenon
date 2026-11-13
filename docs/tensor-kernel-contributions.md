# Tensor kernel contributions

`FHN/TensorKernels.h` defines a provider-independent registry for semantic tensor
operations. It belongs to the public C++ runtime. A contribution supplies a JSON
contract, pure public-metadata inference and an executor over immutable Runtime
values. The registry does not import a ciphertext implementation or client key.

```cpp
#include <FHN/TensorKernels.h>

using namespace fhenomenon::tensor;

registerTensorKernel({
    "tensor.example",
    {{"input_arity", 1}, {"output_arity", 2}},
    infer_public_metadata,
    execute_provider_operation,
});
```

The operation name is unique. Arity is a nonnegative integer; missing callbacks,
invalid contracts and duplicate names are rejected. Register contributions at
library initialization, then take an owning `tensorKernelDefinitions()` snapshot
when constructing an adapter. Later registrations cannot modify that snapshot.

`KernelEnvironment` provides the evaluator's public profile, scales and moduli.
`CipherMetadata` describes logical count, packing period, message representation,
level, scale and ciphertext payload bytes. `infer` validates the operation's
geometry and prepared-key requirements without executing provider arithmetic.
Its `KernelPlan` supplies ordered output descriptors and separate conservative
reservations for temporary ciphertext and public-buffer payloads.

`KernelExecution` gives the executor a Runtime, an opaque evaluator handle and
adapter-supplied borrow/produce callbacks. The adapter checks provider ownership,
reserves temporary storage before arithmetic, validates every produced output
against the inferred descriptors, and publishes the output group atomically.
Individual provider implementations may use their own native interfaces without
changing the registry or model-facing operator API.

An adapter can register each definition as an eager `Recipe` and a `ScopeKernel`.
It copies public attributes and metadata during capture, then executes the
operation as one scoped command. An executor may use composed provider
arithmetic or a hardware-specific fused implementation. Those choices do not
change the semantic contract and must not be confused in performance reports.

Catalogs should describe dtype, format, layout/packing, representation, geometry,
level consumption, prepared keys, temporary reservations, excluded memory and
measurement boundaries. Runtime payload budgets do not automatically cover
provider keys, caches, internal scratch or allocator overhead. These domains
need explicit accounting rather than an implied total-memory guarantee.

The standalone `TensorKernelsTest` exercises registration, invalid/duplicate
definitions, snapshot ownership and callback execution without a native HE
provider. Concrete adapters must additionally test numeric correctness, output
metadata, resource budgets, failed publication and scoped handle lifetimes.
