# Coarse statistics implementation plan

Design: ../specs/2026-10-31-coarse-statistics-design.md.
User already authorized implementation; no additional approval gate.
Working tree: /Users/sunchuljung/repo/zotanika/fhenomenon. Preserve existing changes.

- [x] Native additive C API and kernels: energy, variance, dot, reduce; focused
  regression tests, scratch/transactional failure checks, native plugin retention.
- [x] Independent public-API contribution library: named recipes, owned services,
  tracked staging and canonical result import; integration/budget/lifetime tests.
- [x] Root build/install integration and corpus benchmark: original vs contributed
  vs split-boundary paths and explicit compiled reference; unchanged graphs/oracles.
- [x] Astra review, full relevant Debug and sanitizer checks, installed consumer,
  Release paired timing with all other workloads idle.
- [x] Persist measurements/limits and confirm frozen runtime hashes unchanged.
