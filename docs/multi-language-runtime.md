# C++ and Rust runtimes

## Intent and boundaries

C++ and Rust are independent implementations of the same synchronous runtime
contracts. Rust does not call the C++ Runtime or Scope. The existing native CKKS
engine and CUDA kernels remain shared providers behind their C ABIs; this change
does not port arithmetic or the older scheduler/frontend to Rust.

The Rust implementation covers Runtime, Scope and their contracts. Native
providers remain separate from language runtimes, so extending arithmetic does
not require maintaining two copies of the cryptographic engine.

## Repository layout

- `cpp/`: existing C++ source, headers, tests, examples, contributions and CMake
  implementation. A root CMake entry point preserves root-based builds.
- `rust/`: Cargo workspace, independent runtime crate, native provider adapter,
  tests and examples. The core uses Rust's standard library only.
- `contracts/`: language-neutral behavior fixtures used by both test suites.
- `python/`: shared model frontend, semantic contracts and kernel request analysis.
- `docs/`, `tools/`: shared documentation and tooling.

## Common behavior

Both implementations provide immutable resource identities, retained-byte
accounting and dependencies, ordered recipe selection, bounded diagnostics,
complete scope capture, descriptor inference/validation, inspection, bounded
contiguous kernel selection, ordinary recipe fallback and last-use release.

Capture, matching, inspection and metadata callbacks cannot perform evaluator
work. Strict contracted scopes freeze registration. Captured symbols belong to
one scope and are not concrete values. Failed output validation prevents that
span's publication. Only completed spans count as executed. Deadline checks
occur between spans, including the last one; they cannot preempt provider code.

Caps, ordering, errors and graph results are specified by common fixtures and
focused tests. ABI-compatible provider objects retain their library, compute
context and allocation owners until the last resource dies. Client/key-owning
operations remain separate from compute recipes.

## Rust API decisions

The Rust API uses `Result`, owned `Rc` resources and borrowed scope views rather
than reproducing C++ syntax. Runtime and resources are deliberately not Send or
Sync. Attribute and metadata payloads are owned type-erased values; providers
must treat them as immutable. Rust lifetimes prevent retaining views and many
forms of recording reentrancy. Runtime guards still reject invalid calls made
through shared references. Panic unwinding restores guards and becomes an
execution error at callback boundaries.

No fixed opcode vocabulary is imposed on the generic runtime. The native
adapter translates its own typed attributes to the existing kernel ABI. Kernel
availability is reported explicitly; Rust runtime support does not imply that
every optional native contribution or tensor training adapter has been ported.
The initial native Rust adapter and encrypted integration tests cover the CPU
CKKS C ABI. Its low-level client preserves that ABI's decryption behavior.
The tensor/model provider's mandatory minimum-level egress policy is separate
and is not yet exposed by this Rust adapter. CUDA-specific adapters, compiled regions and prepared-artifact APIs
are not covered by this port. Operations that need modulus-chain information
unavailable through the current public ABI remain unsupported rather than
inferring unverifiable descriptor metadata.

## Validation

The C++ suite includes an installed-package consumer that builds outside the
source tree. Rust tests cover ownership, budgets, failure cleanup, callback
restrictions, descriptor validation, fused spans, inspection and large graphs.
Both suites consume [the common conformance cases](../contracts/runtime-v1.json).
The native Rust integration test encrypts at the client, evaluates a captured
graph through the shared native library, and decrypts only exported results.
CI builds both languages on Linux and macOS.

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DOCUMENTATION=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure

# Replace .so with .dylib on macOS.
FHN_NATIVE_TEST_LIBRARY="$PWD/build/lib/libckks_fhn.so" \
  cargo test --manifest-path rust/Cargo.toml --workspace
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
```

The standalone C++ entry point uses `cmake -S cpp -B build-cpp`. Its executable,
provider and installed-package paths have the same shape as a root build:
`BUILD_DIR/bin`, `BUILD_DIR/lib` and the selected install prefix. Internal CKKS
layer archives live under `BUILD_DIR/cpp/src/CKKS` for a root build and
`BUILD_DIR/src/CKKS` for a standalone build. A source directory change requires
a fresh build directory; existing generated files are not a migration layer.

To check only the dependency-free Rust runtime, run
`cargo test --manifest-path rust/Cargo.toml -p fhenomenon`. Workspace native
tests require `FHN_NATIVE_TEST_LIBRARY`; absence of the provider must not turn
the encrypted integration test into a successful skip.

The [OpenVLA example](../examples/openvla_training/README.md) uses the common
Python package and native tensor provider. Set `FHN_LIBRARY_DIR` to the desired
provider build or installation directory.
