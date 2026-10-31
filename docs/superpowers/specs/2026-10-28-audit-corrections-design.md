# Audit corrections and executable packed workloads

The user approved proceeding with the 2026-10-28 Astra audit recommendations. This change first restores reliable Session execution, buffer lifetime planning, and native/fallback semantics, then connects a portable packed workload verification path. It also closes reproducible CKKS validation/lifetime and packaging failures. It does not implement the remaining native CKKS evaluation/key layers or claim GPU results without a GPU run.

## Constraints

- Preserve C++17, the flat FHN IR, existing source compatibility where practical, and ABI version 1. Optional additive exports must not make existing backends unloadable.
- Add no third-party dependency. Reuse cached build dependencies for local verification.
- Preserve all user untracked files, especially examples/resnet_circuit.cpp, examples/README.md, examples/SPEED_ANALYSIS.md and .motif directories.
- Tests must exercise real behavior and the audit counterexamples. Observe failures before fixes; never substitute source-text checks for execution.
- Keep CKKS layer dependencies strictly downward. No hidden allocator, global parameter context, or new singleton.
- Distinguish internal Session temporaries from caller-observable values. Shared ownership alone does not imply that a value is internal.
- Reject known unsupported/unsafe requests before mutating the offending operation's caller buffers; execution is not transactional and earlier successful instructions may remain applied. Do not invent an unbudgeted opaque scratch allocation to make fallback aliasing appear supported.
- Real execution, planning-only results, unsupported cases and failures must be reported separately. Never claim GPU latency without completion synchronization.
- Product changes remain local and reviewable; no publish, push, merge or deployment is authorized by this task.

## Approach

Use narrow corrections to the existing architecture rather than replace the frontend or compiler. Give synthetic recording results explicit provenance, preserve profiles, and disable automatic diagnostic tree expansion. Bind movement plans to the exact program they describe. Keep byte-mode eviction a documented heuristic unless a stronger algorithm is independently proved. Make fallback primitive requirements and alias limitations explicit, with early rejection when the ABI cannot supply planned scratch.

For CKKS parameter magnitude checks, use exact integer arithmetic or an exact fallback; neither a tolerance nor wider floating point establishes P > digit. Validate layouts before narrow arithmetic. Protect Arena state across guard/move interactions without silently allocating.

The packed validation increment adds optional vector f64 host-side exports and backend configuration to the corpus consumer. Verify at least one shallow packed matvec through the public ABI with tolerances and native/fallback comparison. CPU ToyFHE validates plumbing; Cheddar support is implemented and reviewed from available source, with actual CUDA verification explicitly remaining hardware-dependent. Record executed/skipped/planned counts and make explicit required verification fail when nothing runs. Synchronize GPU measurement boundaries and make zero matching GPU tests fail.

## Alternatives considered

A wholesale frontend redesign and a general kernel-constraint schema would expand scope before an executable workload establishes requirements. Finishing native CKKS L2–L6 first would leave the current runtime defects and comparison baseline unresolved. The chosen approach fixes observable contracts and builds one measurable path, preserving the architecture's small boundary.

## Validation

Run the existing full CTest suite, focused regressions, ASan/UBSan, and the no-LTO build that exposed incomplete template definitions. Check test-off configuration, installation and a package consumer. Re-run the session expression and scaling/liveness cases. Exercise the public packed corpus against ToyFHE. Verify GPU-related behavior through source and CPU fixtures locally and clearly state that no GPU performance result is produced on this machine.
