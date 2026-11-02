# Lightweight execution runtime

The public C++17 API in `FHN/Runtime.h` executes a named request when submitted.
Optional `FHN/Scope.h` records bounded commands before kernel selection; neither
entry point invokes a compiler automatically. Contributions register
recipes with public-metadata applicability checks, deterministic priorities and
an execution callback. One recipe can produce several outputs, allocate a
preparation artifact, or compose other requests. The runtime has no built-in
arithmetic instruction vocabulary.

This replaces the earlier (2026-10-28) mandatory semantic-graph/provider-search
proposal.
Existing backend-bound `Fhenon` sessions and physical FHN programs remain
available. `Session::create(Runtime&)` explicitly selects the native scope route;
it does not silently redirect the old frontend.

## Execution and resource contract

`Runtime::submit(Request)` selects the first applicable recipe, ordered by
descending priority and then registration order. An applicability decline may
try another recipe. Once execution begins, an exception is reported without
retrying a different implementation. Callers observe returned outputs only on
success. Providers must preserve escaped inputs; the generic core cannot police
mutability hidden inside arbitrary `std::any` payloads or roll back external
side effects.

`Value` is a shared immutable resource handle with an identity, format tag,
provider payload and strong dependencies. Copies share one charge. Resource
payloads are destroyed before their dependencies, so provider cleanup may use
the dependency storage. Handles can outlive their `Runtime`; native handles
also retain their backend context and loaded library. Releasing the last handle
releases its charge. A new encryption/import has a new identity even when its
shape and parameters match an existing input.

`makeValue` reserves the declared bytes **before** invoking its factory and
rolls back reservations on failure. Reentrant factories count their pending
reservation. The optional retained-memory budget is not a process-memory
limit: keys, context tables/scratch, callback temporaries, graph metadata and
allocator overhead are excluded. Native ciphertexts charge their actual fixed
reserved slab size, including spare components. Rotation artifacts charge
their snapshot and coefficient slabs. An explicit region reserves its entire
chosen workspace cap; executable statistics separately expose used workspace.

Configure these runtime-wide settings when constructing `Runtime`:

| `RuntimeOptions` field | Default | Effect |
| --- | ---: | --- |
| `max_live_bytes` | `0` | Maximum bytes charged to retained Values/artifacts and pending reservations. Zero means unlimited. A new reservation that would exceed the limit fails with `BudgetExceeded` before its factory runs. Copies of a Value share its charge; nested recipe allocations use the same budget. |
| `max_trace_events` | `256` | Maximum retained diagnostic events across requests and scopes. Zero disables trace retention. When full, the oldest events are dropped; execution continues. |

This memory budget applies across scopes sharing the Runtime, including resources
exported by earlier scopes. It is separate from `ScopeOptions`, which limits the
recorded command structure and matching work rather than ciphertext bytes.
For example, retaining many exported preparation artifacts can exhaust the
memory budget even when each scope is small.

The trace is a bounded history, not a complete execution log: recipe and scope
selection, declines, completions and failures share it. Increase
`max_trace_events` when diagnosing a long run, or set it to zero when retained
diagnostics are unnecessary. Trace capacity does not change kernel selection.
Calls on a runtime and on a native context are synchronous and must be serialized.
Scope matching is opt-in and bounded; no background queue, automatic cache,
autotuning or global representation search is hidden in dispatch.

## Native execution and reusable state

`FHN/NativeRuntime.h` supplies a compute-only adapter. `NativeServices` can be
injected, or `loadNativeServices(path)` loads the genuine CPU CKKS plugin.
Neither services nor recipes hold a client or secret key. Trusted application
code separately encrypts inputs, prepares public evaluation keys and decrypts
outputs. Keys remain client-owned; expired keys cause evaluation to fail.

```cpp
Runtime runtime({64 * 1024 * 1024, 256});
NativeRuntime native(runtime, loadNativeServices(plugin_path));
// encrypted_input belongs to native.context(); trusted host prepared keys.
auto x = native.importCopy(encrypted_input);
auto prepared = native.prepareRotation(x);
auto left = native.rotate(x, 1, prepared);
auto right = native.rotate(x, -1, prepared);
// prepared remains useful across later commands; release it when no longer useful.
```

Import copies the ciphertext. Subsequent mutation/freeing of the original
buffer cannot invalidate the runtime value. A rotation preparation owns a
canonical source snapshot and inverse-NTT c1 limbs. Each consumer applies its
own coefficient automorphism and basis conversion/key switch. It reuses inverse
transforms, not the entire hoisted key-switch decomposition. Its exact reserve
is `3 * active_q_limbs * polynomial_degree * sizeof(uint64_t)`.
The adapter checks source identity before reuse, including when the supplied
replacement has the same ciphertext layout. The raw optional C exports in
`FHN/fhn_prepared_api.h` instead operate on the owned snapshot directly.

`op()` preserves physical FHN opcode semantics. `semantic()` is an explicitly
chosen application adapter for the eight circuit operations: multiplication
uses normalized HMULT; integer scale uses additions/negation without consuming
a level; fractional scale uses native scalar multiplication. Binary operands
can be lowered to a common level, but incompatible actual scales still fail.
This policy belongs to the native adapter, not the generic runtime.

## Native scopes and kernel selection

```cpp
#include "Session/Session.h"
#include "FHN/NativeRuntime.h"
// runtime, native and x are the same concrete objects as above.
auto session = Session::create(runtime);
auto outputs = session->run([&](Scope &scope) {
  auto source = scope.input(x);
  auto preparation = native.prepareRotation(scope, source);
  auto left = native.rotate(scope, source, 1, preparation);
  auto sum = native.op(scope, FHN_ADD_CC, {source, left});
  return ScopeValues{preparation, left, sum};
});
// outputs[0] can be imported into another scope and reused with the same x.
```

The callback executes once using ordinary public C++ control flow. `Scope::input`
retains a concrete Value from the same Runtime and deduplicates exact identity.
`Scope::submit(operation, inputs, owning_attributes, output_count)` records a
request, including zero/multiple outputs. It returns opaque `ScopeValue` symbols,
which cannot be inspected, decrypted or converted to boolean. Symbols are valid
only in that recording; concrete exported Values can cross scopes. Capture
requires owning immutable attributes, never references to temporary buffers.
The Runtime must outlive a native Session.

After capture, the runtime greedily covers the command sequence from left to
right. `Runtime::addScopeKernel` registers a name, priority, maximum span,
read-only matcher and executor. The matcher receives a bounded `ScopeView` and
returns a prefix length, or zero to decline. Higher priorities win; registration
order resolves ties. This is a local selection policy, not an optimal global
plan. The scope registry is snapshotted before capture. Ordinary recipe selection
still snapshots per submission. Missing span matches fall back to one ordinary
`Runtime::submit` request.

`ScopeView` exposes commands and any currently concrete values without requiring
an arithmetic vocabulary. `liveIns(length)` lists distinct external inputs;
`liveOuts(length)` lists outputs defined inside the span that are exported or used
later. The executor receives a view clipped to its selected span and must return
**exactly those live-outs, in that order**, all owned by the same Runtime. A
matcher must decline if it cannot produce an observable intermediate. Internal
results can stay private to a fused implementation. The core releases concrete
results after their last use, while explicit exports and Value dependencies
retain their resources. Callback export order and duplicates are preserved.
Views are borrowed for one callback and must not be retained.

Scope callbacks, matchers and executors are trusted host extensions. Matchers
must be read-only; immediate `submit` and `makeValue` on this Runtime are rejected
during capture and matching. Executors may compose ordinary requests. Nested
scopes on one Runtime are rejected; Session additionally rejects nested sessions
and mixed Fhenon recording. Recording failures execute no captured commands.
Executor failures release unpublished results and never retry another kernel;
they do not undo an extension's external side effects. `scope.span` trace events
show the chosen kernel, command start/count and live-in/live-out counts.

### Scope options: limits, selection and fallback

`ScopeOptions` is copied into a native Session at `Session::create(runtime,
options)`, or supplied to an individual `runtime.runScope(capture, options)`
call. Every run starts fresh counters; the values are not a lifetime quota for
the Session or Runtime. Changing the original options object later does not
change an already-created Session.

| `ScopeOptions` field | Default | What it controls |
| --- | ---: | --- |
| `max_commands` | `256` | Number of `Scope::submit` calls recorded, including zero-output commands. Requests composed internally by an executing kernel do not count as additional recorded commands. |
| `max_values` | `2048` | Distinct imported Value identities plus every declared command output symbol. Repeated `scope.input(value)` calls share one symbol. Output symbols count even when never used or exported; this is not a peak-live-value limit. |
| `max_input_edges` | `4096` | Sum of input entries across recorded commands. Using one symbol twice in a command counts twice, as does using it again in another command. This bounds dependency bookkeeping independently of the number of distinct symbols. |
| `max_exports` | `2048` | Entries returned by the capture callback, including duplicates and imported values returned without any computation. Exports need not be distinct. |
| `max_lookahead` | `32` | Maximum number of consecutive commands shown to one matcher. The actual window is the minimum of this value, that kernel's `max_commands`, and the number of commands remaining. A smaller window limits eligible spans by truncating the view rather than raising an overflow error. |
| `max_match_attempts` | `4096` | Total `ScopeKernel::match` calls permitted across the entire run, including declines. The counter does not reset at each command. It does not count ordinary recipe applicability checks or requests composed inside an executor. |
| `use_kernels` | `true` | Enables contributed scope-span matching. With `false`, each recorded command goes through ordinary `Runtime::submit` while capture, dependency tracking, resource release and exports remain active. |

All six numeric fields must be positive, even with `use_kernels = false`.
A zero field raises `InvalidRequest` before the capture callback runs. Exceeding
`max_commands`, `max_values` or `max_input_edges` raises `BudgetExceeded` during
recording. `max_exports` is checked after the callback returns, before any
recorded command executes. Recording errors that escape the callback, and
export-limit errors, abort the scope before any recorded command executes. The
runtime never silently truncates the program or splits it into smaller scopes.

The matching budget is different: when another matcher call would exceed
`max_match_attempts`, execution fails with `BudgetExceeded` instead of switching
to ordinary fallback. Exactly that many calls are allowed if no further call is
needed. Earlier spans may already have executed; unpublished results are
released, but extension side effects are not rolled back. With span matching
disabled, no matcher attempts are consumed. `max_lookahead` only clips a window
and therefore has no corresponding quota-exhaustion failure.

`use_kernels = false` does **not** disable prepared or fused ordinary recipes,
remove imported preparation, or change the plaintext meaning of the program.
For example, a prepared search block still selects its ordinary prepared recipe;
only the layer grouping several recorded commands is bypassed. This makes it a
useful comparison when checking the effect of scope selection. It does not make
the run an unprepared or primitive-only baseline. If a recorded operation has no
applicable ordinary recipe, fallback reports `Unsupported`; it does not synthesize
an implementation for a command supported only by a scope kernel.

```cpp
// runtime, native, x and prepared are the concrete objects created above.
ScopeOptions limits;
limits.max_commands = 128;
limits.max_lookahead = 8;
limits.max_match_attempts = 1024;
auto capture = [&](Scope &scope) {
  auto source = scope.input(x);
  auto preparation = scope.input(prepared);
  return ScopeValues{native.rotate(scope, source, 1, preparation)};
};
auto bounded_session = Session::create(runtime, limits);
auto selected = bounded_session->run(capture);

// The same capture and preparation, with only scope-span selection disabled.
limits.use_kernels = false;
auto ordinary = runtime.runScope(capture, limits);
```

Choose command/value/edge/export caps for the largest scope the application
intends to record. Raise only the limit reported by an expected larger workload,
or explicitly divide the work into scopes and carry exported state between them.
Lowering `max_lookahead` can exclude larger kernels; raising it only helps when a
registered kernel accepts a larger span. The built-in search collection kernel
has its own eight-command cap, so lookahead above eight does not enlarge it.
A larger kernel registry or many declines may need more matcher attempts; inspect
`scope.span` diagnostics before increasing that budget.

These settings bound structural counts, not callback time, arbitrary
`std::any` allocations, bytes held by captured attributes, or total process
memory. They do not guarantee a latency bound or the fastest kernel choice.
There is no speculative replay, encrypted branch exploration, automatic key
discovery or automatic reuse discovery.

The [search scope contribution](../contributions/search/README.md#native-session-scopes)
selects multi-block and single-block implementations over separately recorded
requests. Its query preparation survives both ordinary commands and later scopes.
`NativeRuntime` also provides scoped physical op/rotation adapters and public
`NativeOpAttributes` for matchers. Scoped `semantic` and `compileRegion` adapters
are not provided; explicit immediate compilation remains available outside capture.

## Optional compilation

```cpp
auto shifted = native.semantic(FHN_CIRCUIT_ADD_CONSTANT, {x}, 0, 1);
auto region = native.compileRegion(circuit, {shifted},
                                  {256, 8 * 1024 * 1024});
// Trusted host: client_prepare_executable(client, native.executable(region)).
auto outputs = native.runRegion(region, {shifted});
auto next = native.semantic(FHN_CIRCUIT_ADD_CONSTANT, {outputs[0]}, 0, 2);
```

Compilation is an explicit request with a node limit and finite workspace cap.
Prototype inputs contribute signatures; the executable does not retain their
data and may be rebound to fresh compatible inputs. Oversized or unsupported
regions leave immediate execution usable. There is no implicit “try the
compiler” step on ordinary requests. Compile-time graph metadata is bounded by
the node/input/output limits of the compiler, not charged as ciphertext memory.

## Contribution and validation

The separately linked [statistics contribution](coarse-statistics-kernels.md)
adds centered energy/variance, packed dot and partial reduction recipes. Its
intermediates pass through existing commands and back into contributed kernels
using the unchanged public runtime and native adapter.

The separately linked [prepared matvec contribution](matvec-preparation.md)
adds an explicit reusable source artifact, complete dots, products and partial
reductions. Fixed-multiplier companions and rotation maps survive separate
requests and ordinary intermediate commands, without changing the runtime core.

`test/install-consumer/rotation_contribution.cpp` is compiled separately using
only installed public headers. It registers `example.rotation-pair`, a
multi-output recipe composing two `native.rotate` requests that share one
caller-supplied preparation. The installed consumer also passes an immediate
result into an explicit compiled region. No core opcode or matcher edit is
needed for the contribution.

`fhn-bench-runtime NATIVE_LIBRARY [REPS]` runs the existing `stats`,
`matvec-fused` and `matvec` corpus workloads in source order. The immediate and
prepared modes have compiler function pointers removed. The explicit region
mode uses the same imported mathematical operations and separately requests
compilation. All outputs and all logical slots are compared with the unchanged
independent corpus oracle on warmups and measured runs.

The harness drops intermediates after their last use. Its timing includes
dispatch, allocation, arithmetic and temporary release, while excluding input
import, cryptographic setup, preparation and decryption. Preparation costs are
reported separately and must be added for one-shot use. Measurement order
rotates across modes. This is a comparison between paths of this project, not
an OpenFHE/SEAL/HEaaN benchmark. The explicit region example chooses a 32 MiB
reservation; it is not an automatically determined optimum.

The [2026-10-31 validation report](lightweight-runtime-validation-2026-10-31.md)
records tests, measured costs and review corrections. Complete timing samples
are in [the result JSON](../benchmarks/results/2026-10-31-lightweight-runtime-release.json).

`stats` computes sums and scaled population variances of two 64-slot datasets;
`matvec-fused` computes an encrypted 8×8 matrix/vector example with eight
outputs; `matvec` is the existing cyclic rotation/scalar workload. Names do not
establish general application coverage.

Remaining scope: CGGI↔CKKS/BFV conversions, cross-device events/residency,
automatic cross-command reuse discovery, a stable C plugin ABI for arbitrary
recipes and application-level encrypted control flow are not implemented.
This runtime exposes public orchestration and reusable resources; it does not
remove the cost of hidden data-dependent branches.
