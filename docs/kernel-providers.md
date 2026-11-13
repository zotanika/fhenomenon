# Kernel providers and implementation replacement

The framework and the kernel implementation have different release boundaries.
Model capture, semantic graphs, compilation, Runtime/Scope, contracts, tests and
cost attribution belong to the common framework. Arithmetic and specialized
kernels belong to a provider. A provider may be built from source or distributed
as a compatible binary; its performance is measured under the same contracts.

This separation supports a kernel-supply business: a customer can retain model
code and evaluation criteria while obtaining an implementation for its measured
bottleneck. An engineering request carries public geometry, required operations,
accuracy and resource conditions. It does not need to contain model weights or
training examples, although geometry can disclose architecture. Commercial value
still requires measured improvement on the customer's workload and hardware.

## Interfaces and loading

| Interface | Implementation boundary |
| --- | --- |
| `fhenomenon.ml` | Shared capture, compilation, tensor contracts and conditional cost analysis |
| `FHN/TensorKernels.h` | C++ contribution inference, execution and checked output publication |
| `FHN/Scope.h` / Rust `scope.rs` | Bounded multi-command selection and resource contracts |
| `cpp/bindings/tensor_runtime/bridge.h` | C ABI for a complete scoped tensor program |
| `cpp/providers/ckks_tensor/backend.h` | C ABI for client, evaluator and immutable ciphertext handles |
| `cpp/providers/ckks_tensor/cuda/` | Device arithmetic behind the same provider boundary |

The tensor provider and tensor runtime expose ABI version 1. Python rejects
incompatible versions before constructing handles. It also checks that the
client and runtime use the same loaded provider instance. ABI version matching
alone does not establish compatible cryptographic parameters or numerical error.
Those conditions are carried by profiles, value descriptors and operation checks.

On Linux, select a build or installation directory before starting the process:

```sh
python -m pip install './python[ml]'
FHN_LIBRARY_DIR="$PWD/build/lib" python -m unittest \
  python.native_tests.test_backend \
  python.native_tests.test_runtime \
  python.native_tests.test_library_identity \
  python.native_tests.test_model_compiler
```

The directory contains `libfhn_ckks_tensor.so` and `libfhn_tensor_runtime.so`.
The latter links the former; keep them together when installing. The same
runtime binary can use a compatible replacement provider. Do not mix handles
from separately loaded provider instances. CPU and CUDA are build/device choices,
not different model APIs. The current Python loader selects one device mode per
process; use separate processes for comparisons.

```python
from fhenomenon.backends import create_runtime_context
from fhenomenon.ml import TensorEngine

# NativeTest is deliberately insecure, for correctness tests only.
client, backend = create_runtime_context(profile="NativeTest", device="cpu")
engine = TensorEngine(backend, client=client, slots=256)
# Capture and compile a model with this engine as in model-compiler.md.
```

Use `device="cuda"` with a CUDA-enabled provider. Client key generation,
encryption and terminal decryption remain trusted host operations. Evaluator
arithmetic must not fall back to plaintext. The sample runs both roles in one
process; it is not a network-isolated service.

Measurements bind to the actual loaded runtime/provider binaries, graph,
cryptographic parameters and device. A replaced binary requires a fresh process,
recompilation and a fresh measurement. Linux loaded-file identity is checked;
profiling does not trust a manually supplied implementation label. Capture time,
setup and unattributed output work are distinct from semantic kernel timing.

## Operation coverage

| Contract | CPU provider | CUDA provider |
| --- | --- | --- |
| `add`, `sub`, `neg`, `add_plain` | Native canonical RNS arithmetic | Canonical buffers with Cheddar modular arithmetic |
| `mul`, `mul_plain` | Native multiplication, key switching and rescale | Fhenomenon device orchestration with Cheddar arithmetic/NTT butterflies |
| `rotate` | Native automorphism and key switching | Device automorphism, key switching and NTT |
| `level_down` | Explicit modulus drop | Explicit device modulus drop |
| `mod_raise(target)` | Target-level encrypted refresh schedule | Same schedule using device arithmetic |
| `slot_to_coeff`, `coeff_to_slot` | Explicit linear transforms | Same transform semantics using device arithmetic |
| `tensor.public_linear` | Packed diagonal contribution | Same contribution over CUDA provider operations |

A primitive fallback is a declared decomposition, not a fixed upper limit on
kernel granularity. Tensor contributions use the public C++ interface; bounded
multi-command contributions use Scope. The Python registry currently replaces
one semantic operator at a time. General cross-operator fusion and a stable C
plugin ABI for arbitrary C++ contributions are not implemented.

ModRaise accepts the requested target level; it does not automatically refill to
the largest chain. Refresh keys are prepared explicitly. Slot/coefficient
transforms change the representation and consume levels; they do not substitute
for refresh. The client lowers ciphertexts to the minimum level before terminal
decryption. Metadata checks and tests cover these distinctions.

## Contract assurance

JSON is a serialization format, not a proof language. Current model and tensor
APIs construct typed descriptors and validated metadata; arbitrary matcher and
executor callbacks remain trusted code. Three different obligations matter:

1. **Contract consistency:** types, shapes, finite bounds, supported operations,
   levels, representation and memory declarations must be valid. Current checks
   cover defined fields and supported relations, not arbitrary logical formulas.
2. **Composition:** inferred outputs must satisfy consumer requirements and
   belong to the same runtime. Scope checks descriptors and live-outs before
   publishing results. Parameter changes can require recompilation.
3. **Implementation conformance:** a kernel must compute what it promises.
   Differential tests, boundary cases and runtime output checks provide evidence;
   output shape and metadata checks cannot prove arbitrary arithmetic correct.

Public approximation ranges and some memory estimates are caller/provider
requirements, not mechanically proved properties of all secret inputs. A custom
matcher cannot serve as a proof merely because it returns true. The framework
currently has no SMT solver, general contract DSL or formal correctness claim.
Future proofs can attach to normalized, versioned contracts without changing the
model API; they would need explicit assumptions and independently checked scope.

## Verification and limits

The maintained tests cover modular edge cases, NTT/INTT sizes, CPU/CUDA encrypted
agreement, nondefault-stream ordering, device ownership, representation changes,
target-level refresh, whole-Scope publication, binary identity and model costs.
The OpenVLA fixture adds the full forward/backward/update graph. See its
[reproduction instructions](../examples/openvla_training/README.md).

`NativeTest` uses insecure small parameters. `Native128` enforces an internal
modulus-budget policy, not independent security certification. No full-size
OpenVLA encrypted training, convergence, production security or performance
advantage is claimed by these tests. Retained-resource accounting excludes some
provider caches, keys and internal scratch; it is not a process/GPU memory cap.
