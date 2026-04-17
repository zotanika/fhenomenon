# Fhenomenon Vision Archive

This document preserves ideas from earlier README drafts that are no longer accurate as present-tense project claims.

Most of these ideas are not abandoned. They are long-horizon capabilities that should become possible only after the lower-level FHN model is strong enough to support them honestly. Keeping them here lets the project keep its memory without asking newcomers to confuse aspiration with implementation.

## Current Center Of Gravity

The codebase has moved toward a compute-device-style FHE backend interface:

- `FhnProgram` is a flat, public, topologically sorted instruction stream.
- Backends expose a C ABI with four required lifecycle/table functions.
- Kernels live in a simple opcode table.
- `FhnDefaultExecutor` dispatches supported kernels and decomposes selected fused opcodes.
- ToyFHE is the reference backend.
- `ExternalBackend` loads compatible shared libraries.
- Cheddar-FHE is the first GPU CKKS integration target.

This direction comes before the older high-level frontend story. The long-term ambition is still broad freedom for FHE programming, but the project is currently building the substrate that such freedom would need.

## Archived Or Deferred Ideas

| Earlier README claim | Current status | Reason |
| --- | --- | --- |
| `Compuon<T>` should transparently support `int`, `float`, `double`, `std::complex`, and POD structs. | Long-horizon. | The most exercised frontend path is `Compuon<int>`. ToyFHE has fixed-point internals, but broad public template support and POD packing need a real layout story first. |
| Repeated `belong()` calls should transcrypt between schemes. | Long-horizon. | The code does not implement scheme switching or bootstrapped transciphering/transcryption. This can return later as an explicit operation with cost and security semantics, not as magic. |
| POD struct fields should map directly into CKKS slots. | Long-horizon. | Slot layout should probably be handled by an explicit packing/layout layer or compiler frontend, not by magical struct reflection in `Compuon<T>`. |
| The scheduler should automatically infer FHE parameters for arbitrary programs. | Long-horizon. | Parameter selection is scheme- and backend-specific. The current FHN ABI leaves such policy closer to the backend and future planning layers. |
| Fhenomenon should expose a general hardware abstraction layer for GPUs, ASICs, and FPGAs. | Reframed. | Hardware control is backend-owned today. A future cross-backend capability model may emerge after backend-owned kernels are real. |
| External libraries should mainly implement a C++ virtual `Backend` interface. | Replaced for now. | The newer direction uses a C ABI and kernel table so FHE libraries can integrate through stable shared-library exports. |
| `ExternalBackend::add()` and `ExternalBackend::multiply()` should behave like the legacy built-in C++ backend. | Deferred/legacy. | External backends are expected to run through FHN programs and the executor path. The direct virtual methods are currently compatibility surface. |
| Full key management, secure enclave support, key rotation, and audit logging are core README commitments. | Long-horizon. | These are important, but not implemented. The README should not imply production security posture before the project earns it. |
| Fhenomenon already supports SEAL, OpenFHE, HElib, HEAAN, CKKS, BGV, BFV, and TFHE transparently. | Long-horizon ecosystem goal. | The actual default path is ToyFHE. There is an optional TFHE-rs experiment and an optional Cheddar-FHE CKKS backend. |
| The strategy framework should be the main extension point for optimization passes. | Narrowed. | Passes still exist, but the current contribution path is clearer at the FHN opcode, lowering, decomposition, and backend-kernel levels. |
| Matrix multiplication recognition and fused matmul kernels are implemented end to end. | Long-horizon, partially scaffolded. | There is scaffolding and design intent, but the active implementation is not yet a complete fused matmul pipeline. |

## North Star Beyond The Backend

The backend-first direction is not an argument against high-level programming. It is the route toward it.

Eventually, Fhenomenon should make it possible to build ML-style end-to-end computation graphs over encrypted data, closer in spirit to TensorFlow than to a pile of isolated arithmetic calls. That future needs:

- a kernel catalog rich enough to express real workloads;
- packing and layout planners that understand CKKS/TFHE-style constraints;
- cost models that make secret control flow visibly expensive;
- graph-level APIs that naturally decompose work into independent encrypted subproblems;
- backend capability discovery across CPU, GPU, FPGA, ASIC, and library targets;
- enough tests and benchmarks to keep high-level abstractions honest.

The reason to move high-level claims out of the README is not that they are contradictory. It is that Fhenomenon should earn those claims from the kernel layer upward.

## Still Worth Exploring

These ideas remain interesting once the FHN boundary is stable:

- A layout system for CKKS packing and rotation planning.
- A formal model of what public host scheduling may reveal.
- A frontend that rejects or rewrites secret-dependent control flow before lowering.
- FHN as an output target for MLIR/HEIR experiments.
- Domain kernel libraries for linear algebra, encrypted inference, approximate nearest neighbor baselines, sorting networks, and database-style scans.
- Benchmarks that compare primitive-by-primitive execution against fused-kernel residency.

## Principle

The README should describe what a new contributor can touch today.

The archive should preserve what the project may become tomorrow.
