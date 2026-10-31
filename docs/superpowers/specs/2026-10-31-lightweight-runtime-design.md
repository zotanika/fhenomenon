# Lightweight execution runtime

User-approved direction, 2026-10-31: execute through a lightweight runtime;
compile only explicitly selected regions; preserve intermediate values and
prepared state across commands. This supersedes the mandatory semantic graph,
global offer search and whole-plan admission machinery in the October 28 design.

## Boundary

`Runtime::submit(request)` chooses an applicable registered recipe and executes
it synchronously. Requests are namespaced operations with immutable resource
handles and provider-owned attributes. They may denote one operation, several
outputs, preparation, or an application-level computation. No fixed arithmetic
taxonomy, source-language compiler, graph cover search, or inferred conversion
chain belongs to the core.

Recipes declare applicability using public inputs and configuration. Stable
priority/registration order chooses between applicable recipes; declines are
recorded. Once execution starts, failure is reported, never retried invisibly.
Contributors supply fallback recipes. Numerical correctness remains part of
their documented supported envelope and actual conformance tests, not a new
runtime proof system. Unknown operations fail immediately.

## Resources and reuse

An immutable handle carries identity, format, kind (value or artifact), tracked
reserved bytes, opaque typed payload and strong dependency handles. A resource
may outlive the command that produced it. Dependencies keep source values alive
until all consumers release the prepared state. Copying a handle does not copy
or recount the allocation. The runtime tracks live/peak retained bytes and
checks its budget before calling a resource factory. This budget excludes
provider-private temporaries, native context scratch, keys and allocator overhead;
it is not a process-memory bound.

Inputs owned by this API are immutable. New encryption/import creates a new
identity. Native prepared rotations require both the source handle and its
preparation handle; using preparation with a different source is rejected even
when shapes match. No ciphertext-dependent cache keyed only by a pointer or
shape is introduced. Input mutation through an escaped provider pointer is a
contract violation, not supported rebinding.

Outputs are returned only after successful recipe completion. No caller-owned
output is modified by the generic API. Providers must not mutate input payloads
or perform hidden semantic side effects. The first runtime is single-threaded,
has bounded trace retention, and performs no async scheduling or secret-driven
dispatch. In-process callbacks are trusted and cooperatively bounded.

## Native proof

Use the genuine native CPU CKKS backend. Its host client prepares keys separately;
compute recipes never receive a client/secret pointer.

- Direct recipes execute existing physical FHN kernels immediately.
- A preparation request retains source ciphertext data plus inverse-NTT source
  coefficients. Separate rotation commands share that preparation. Keep existing
  per-rotation basis conversion/key switching; do not assume full approximate
  basis conversion commutes with automorphism.
- Explicit region compilation wraps the existing `FhnCircuit` compiler. The
  caller requests it with a node/workspace bound, prepares keys on the trusted
  client, then invokes it as one recipe. Oversized regions decline before compile.
  Compilation failure does not affect prior runtime values or other requests.

The optional compiler remains useful within its existing supported semantics.
It does not run before ordinary submissions and does not compile a whole session
implicitly. Direct physical op semantics remain unchanged; especially raw
`MULT_CC` and `MULT_CS` are not silently replaced with circuit semantics.

## Required evidence

1. Core tests: immediate execution, independent named recipes, deterministic
   applicability/fallback, multi-output results, dependency lifetime, budget
   rejection before allocation, failures, cross-runtime handles, bounded traces.
2. Crypto differential tests: prepared rotations equal current rotations with
   identical ciphertexts/keys, including signed rotations and all supported
   levels; source snapshot, key expiry, invalid context and scratch failures.
3. Real encrypted integration: commands retain values and preparation; same-shape
   new ciphertext cannot use stale preparation; explicit compiled regions accept
   runtime results and return reusable outputs; a refused region leaves eager
   execution usable. Keys remain a separate host action.
4. Existing `stats` and `matvec-fused` graph/oracle unchanged. Compare immediate
   execution, optional compilation and prepared rotation fan-out where relevant.
   Report preparation, execution and tracked memory separately. No presumed
   performance improvement, no scheme-switching capability claim.

CGGI/CKKS/BFV conversions can later be explicit contributed recipes with real
algorithms and numerical/key contracts. No mock bridge counts as implementation.
