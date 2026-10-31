# Matvec reuse without changing the formula

User request: explain the negligible matvec-fused coarse-kernel gain and test
state reuse/data-movement reductions, with the runtime core fixed. Work in the
primary checkout, preserve existing changes, no Git writes. Runtime and
NativeRuntime headers/implementations plus corpus graph/oracle are hashed.

The app stays eight encrypted products sharing one vector, each followed by
rotate-add distances 4, 2, 1. Values, levels, scales, key-switch rounding and all
eight outputs must remain unchanged. The target is exact residue equivalence,
not merely agreement after loosening a numerical tolerance.

## Investigation and selected candidate

A phase profiler separates input/key validation, multiplication, three rotation
stages, output movement and publication. It checks its reproduced private
pipeline exactly against the existing public coarse kernel before timing.
Profiling/builds and final performance runs are serialized across agents.

The candidate is a caller-owned prepared source artifact containing an immutable
shared-vector snapshot, optional Shoup companion tables for that vector and the
immutable public evaluation keys, and optional rotation index maps. These are
fixed-multiplier/index precomputations; they do not change a mathematical
operation, modular result, rotation count, modulus drop or rounding decision.
Flags permit an unaccelerated prepared control, maps-only, keys-only,
vector-only and fully prepared comparisons. The keys flag also enables per-call
fixed-scalar Shoup hoisting; the reported ablation must disclose both effects. Retained bytes and setup costs are
reported for each. Pure output copy/allocation reductions are measured rather
than assumed to dominate.

Key objects are immutable; preparation validates them once. The artifact holds
weak key identities and each consumer acquires current matching strong leases
for the duration of that command. Expired/replaced keys reject. No client/secret
pointer is held. Source snapshot/companions cannot be rebound to a fresh
same-shaped Value; the contribution checks source identity.

Consumers include full packed dot, normalized product only, and reduction of an
existing matching product/intermediate. Thus separate row commands share the
prepared artifact, and a product may pass through ordinary rotate/add commands
before a prepared partial reduction completes it. Final values still publish
through unchanged NativeRuntime.importCopy; staging/copy costs are included.
No implicit compiler, graph search or new core operation is introduced.

## Safety and evidence

- Reserve exact slab/map bytes before preparation; check overflow and rollback.
- Preserve context/library ownership, weak key expiry, output transactionality,
  scratch bounds and no per-consumer heap allocation in native kernels.
- Exact differential tests across levels/digit configurations and optional
  preparation flags; stale source/context/key, aliases and failed budgets.
- Default genuine CKKS app oracle and exposed-intermediate interoperability.
- Keep unprepared algorithms available as controls; run component ablations,
  Debug/sanitizers, installed consumer, Release/test-off and isolated paired runs.
- Report measured cause, speedup or lack of it, extra retained memory, setup and
  break-even reuse count. Avoid generalizing one-host warm timings to service
  latency or different FHE parameters.
