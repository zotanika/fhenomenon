# Rust runtime

This workspace contains an independent implementation of the Fhenomenon
Runtime, Scope, resource and contract APIs. It does not call the C++ Runtime or
Scope. The core crate uses only the standard library; `serde_json` is a test
dependency. The separate `fhenomenon-native` crate loads the shared CKKS C ABI
with `libloading`.

Rust 1.85 or later is required. The runtime is synchronous and deliberately
single-threaded. `Rc` handles are neither `Send` nor `Sync`. Native kernels may
parallelize internally; moving one context between concurrent calls is not part
of this contract.

## Build and test

From the repository root:

```sh
cargo test --manifest-path rust/Cargo.toml -p fhenomenon
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ckks_fhn -j 4
FHN_NATIVE_TEST_LIBRARY="$PWD/build/lib/libckks_fhn.so" \
  cargo test --manifest-path rust/Cargo.toml --workspace
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
```

Native tests require an explicit library path and fail when it is unavailable.
There is no automatic native-test skip. CPU CKKS integration is tested; a Rust
CUDA provider adapter has not been validated. The C++ arithmetic implementation
and its CUDA implementations remain outside this Rust port.

Both language test suites consume [`runtime-v1.json`](../contracts/runtime-v1.json).
Its hand-written cases specify graph results, exported aliases, exact progress
counters, capture caps, fallback selection and cleanup after failure. Focused
Rust tests add registry snapshots, panic recovery, metadata guards, payload and
dependency destruction, fused validation, deadlines and a 100,000-command graph.

## Runtime usage

[`scoped_integer.rs`](fhenomenon/examples/scoped_integer.rs) is a complete example
of resource creation, recipe registration and one scoped graph without native
dependencies:

```sh
cargo run --manifest-path rust/Cargo.toml -p fhenomenon --example scoped_integer
```

`ScopeContract` describes imported values, infers descriptors during capture and
validates produced values before publishing each span. Strict scopes freeze
recipe/kernel registration. Callbacks must treat `Data` payloads as immutable;
type erasure does not prevent a provider from placing an interior-mutable object
inside one. Panic unwinding is converted to `ExecutionFailed` at callback
boundaries; process aborts and native memory faults are not recoverable errors.

A `ScopeKernel` matches a bounded contiguous command view and returns exactly
the span's live outputs, in their symbolic order. The same descriptor validation
applies to ordinary recipes and coarse kernels. An accepted kernel failure
stops execution; it does not silently rerun another implementation. Declines
permit a later kernel or ordinary recipe to handle the span.

Handles share immutable resource identities and accounting charges. Payloads
are destroyed before their dependencies, and resources may outlive their
runtime. `max_live_bytes` bounds charged retained resources, not graph metadata,
keys, provider scratch or temporary allocations. A scope deadline is checked at
capture completion and span boundaries; it cannot interrupt provider work.

## Native CKKS usage

[`scoped_ckks.rs`](fhenomenon-native/examples/scoped_ckks.rs) encrypts a vector,
captures a rotation and addition, executes them using one native coarse span,
and decrypts only the exported result:

```sh
FHN_NATIVE_TEST_LIBRARY="$PWD/build/lib/libckks_fhn.so" \
  cargo run --manifest-path rust/Cargo.toml -p fhenomenon-native \
  --example scoped_ckks
```

`NativeProvider::load` is an explicit unsafe boundary: callers must trust the
library and its ABI implementation. The loader checks the version before using
other exports, but a version number cannot prove arbitrary native code safe.
Buffer owners retain their compute context and library. `NativeClient` owns the
secret-key context separately, and encryption/decryption are never registered as
recipes. Dropping the client expires native evaluation keys; key-dependent
operations then fail while keyless addition remains available. This is a
single-process integration, not a serialized client/server transport.

The adapter exposes physical copy, add/subtract, negate, scalar add/subtract,
raw tensor product, relinearization, rotation/conjugation, level-down and the
native rotate/conjugate-add operations. The optional `register_scope_kernels`
registration fuses a rotation followed by addition when the rotation's result
does not need to escape that span. Exporting the intermediate forces ordinary
recipe fallback. This is an explicit pattern-based capability, not an automatic
cost optimizer.

Native contracts check context, operand count, reserved bytes, logical slots,
components, level and scale. Evaluation-key availability is checked by the
native operation at execution; ABI 1 has no pure key-capability query. The
contract does not infer rescale, normalized multiplication or scalar multiply
because ABI 1 does not expose the modulus-chain values needed to derive their
actual output scales. Compiled regions, prepared artifacts, plaintext vectors,
bootstrap/ModRaise and tensor training adapters are not ported here. No new
cryptographic arithmetic is implemented in Rust.
