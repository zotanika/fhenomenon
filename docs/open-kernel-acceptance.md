# Acceptance cases for an open kernel runtime

> Historical requirements for the superseded global-planning proposal below.
> Current acceptance and measured implementation live in
> [lightweight-runtime.md](lightweight-runtime.md). Workload semantics, all-slot
> correctness, source identity, failure isolation and accounting remain useful;
> mandatory graph matching/search and global numerical admission do not apply.

Status: requirements for the [proposed runtime](superpowers/specs/2026-10-28-open-kernel-runtime-design.md),
not new implementation or benchmark results. Date: 2026-10-28.

## Keep the current evidence, and give it the right scope

The existing corpus is an execution/compiler regression corpus. Its graphs must
not change to make a new provider look better. A separate application's
requirements need independent validation before these results can support an
adoption claim.

| Existing shape | Actual computation | Useful evidence |
| --- | --- | --- |
| `matvec` | Scalar-weighted cyclic rotations of one packed vector; a circulant-style transform. | Common-source rotation preparation and fan-out. |
| `matmul-tile` | Two such transforms in sequence. | Multi-stage execution, not a general matrix-multiplication implementation. |
| `matvec-fused` | Eight encrypted row dot products with a shared encrypted vector. | Real encrypted 8x8 matrix application, multiple outputs, shared input. |
| `stats` | For two datasets: `sum(x)` and `64*sum(x*x)-sum(x)^2`. | Escaping intermediate values and four outputs. The second result is `4096 * population_variance` for 64 elements. |
| `logreg` | Three dot products, each transformed by `3*d-d^3`, then summed. | Execution of that polynomial, not validation of a logistic model. |
| `psi-eq` | AND of 32 aligned-pair equality results. | Exact Boolean requirements, not a private set intersection protocol. |

These descriptions follow `benchmarks/corpus/corpus_shapes.cpp`. The current
`examples/resnet_circuit.cpp` computes plaintext vectors and ReLU. It is not an
encrypted integration test.

## First release: independent contribution on genuine CKKS

Freeze the following before timing or choosing a provider:

- Original graph, all input/output IDs, encryption status, logical slots and
  independently evaluated output meaning.
- Current native profile for like-for-like comparison: N8192, Q approximately
  50/40/40 bits, P approximately 60 bits, dnum3, initial scale 2^40. This profile's
  public-table screening is not independent security certification.
- Every required output slot must satisfy
  `abs(actual-expected) <= 1e-4 + 1e-6*abs(expected)`.
- This initial profile uses explicit `conformance_only` numerical assurance.
  Admission records a reviewed complete-plan recipe and actual execution
  evidence. A new numerical composition needs new evidence. Passing these cases
  is not an analytical all-input error or failure-probability bound.
- Baseline is the current direct native **circuit compiler**, not only the
  slower historical opcode interpreter.
- No plaintext fallback, fake scheme conversion, fabricated zero-cost boundary,
  dropped output or change to the error/security requirements.

Run these configurations on the same program and input contract:

1. Direct current compiler.
2. New runtime with only its baseline provider.
3. New runtime with independently loaded contributions.
4. Contribution removed, declined, or lacking a required capability.

Registration is not evidence of dispatch. Record the actual plan and execution
trace, including provider/version, source boundaries, produced resources and
artifact consumers. A provider-specific `if` in the planner, a corpus-shape-name
matcher, or adding a private compiler region to enable this contribution fails
the independence test. Provider matchers inspect semantics and declared public
conditions.

### Composition and escapes

Use the unchanged `stats` graph. An external reduction provider computes sums;
native residual regions complete the remaining operations. A larger provider
can propose both `(sum, dispersion)`. All four outputs remain required.

The complete plan must retain or explicitly recompute the sum that is both an
output and an operand of the dispersion expression. Recomputing is permitted
but charged. A multi-output proposal is executed and charged once. A smaller
candidate may coexist with a larger overlapping alternative when the resulting
resource DAG is valid. Adding a consumer of a previously internal value must
either preserve it through the new plan or explain why an offer was declined.

### Reusable non-value state

Use the unchanged `matvec` rotation fan-out. A real provider prepares an
input-derived artifact and several separate rotation tasks consume it. Verify
the cryptographic implementation first using identical ciphertexts and keys;
then check the unchanged application oracle.

Require evidence for one preparation, several consumers, charged bytes and
release after the last consumer. Rebinding or re-encrypting an input with the
same metadata invalidates the artifact. Removing or evicting it must select a
legal reconstruction or alternative plan. A reusable executable or an ordinary
shared ciphertext alone does not satisfy this case.

### Full encrypted matrix regression

Keep `matvec-fused` unchanged: all nine inputs encrypted, eight outputs, each
dot product repeated in all eight logical slots. Test mixed-provider execution
and all output ordering. Public weights or one checked slot are different
contracts and cannot substitute for this check.

### Failure and constraint cases

- Wrong input owner, key identity, logical slot mapping or format version.
- Missing evaluation capability and incompatible conversion direction.
- Authorized-but-not-yet-generated key capability: include preparation costs,
  provision only after selection, and handle provisioning failure explicitly.
- Malformed region boundaries, output IDs, cycles and duplicate candidate IDs.
- Stale ciphertext-dependent artifact after same-shaped input rebinding.
- Hard-budget failure including staged outputs and retained artifacts.
- Enumeration/search limit, with and without an independently validated baseline.
- Provider failure during preparation and after a task has started: no partial
  public outputs, no early release of live dependencies, no unreported retry.
- An application with one required unsupported operation: report the whole
  application unsupported, while preserving separate diagnostic coverage.
- An `analytical_bound` request with only conformance evidence: reject with
  `quality_unverified`; do not silently lower the assurance requirement.
- Hidden semantic side effects in a provider: inadmissible in the initial pure
  executor, regardless of whether its ciphertext outputs can be staged.

## Measurements

Report correctness, coverage, runtime overhead and contribution benefit
separately. A failed or unsupported application has no speedup entry.

Measure completed work and expose at least:

- discovery/analysis/planning time and the configured search limits;
- context, client and key provisioning costs;
- artifact preparation, conversion, transfer and evaluation time;
- first useful result under an explicitly named service boundary;
- repeated execution and amortized setup over a stated reuse count;
- reserved I/O, staging, intermediates, artifacts, scratch, keys and tables,
  without double-counting shared allocations;
- measured-versus-predicted cost and the provenance of estimates.

The current executable's slot-plus-scratch workspace is not total memory.
Unaccounted provider allocations preclude a hard total-memory claim. All phases
excluded from a latency figure remain visible in the report. A slower external
provider can establish composability, but not a speedup; it is not automatically
preferred on invented costs.

## Deferred mixed computation: Boolean/LUT and packed arithmetic

This is a concrete architecture stress case, not evidence of present scheme
switching support. A real service may refine the application requirements before
implementation; do not treat this invented fixture as adoption evidence.

Inputs are eight encrypted records `x_i`, each with eight features, and one
shared encrypted weight vector `w`. Thresholds `tau_i` are public configuration.

```text
s_i = sum_j x_ij * w_j
m_i = 1 if s_i >= tau_i else 0
count = sum_i m_i
selected_score = sum_i m_i * s_i
```

Inputs use nine ciphertexts, each holding eight logical values. Outputs are two
ciphertexts, each scalar replicated over eight logical slots, in the specified
packed arithmetic representation. Internal layouts are free to change, but
extraction, conversion, packing and replication are actual planned work. Masks
stay encrypted. Keep the original packed scores alive when their converted
representations are used for predicates, so they can feed the final aggregate.

Two **separate** numerical contracts are possible:

| Requirement | Exact quantized contract | Approximate-real contract |
| --- | --- | --- |
| Intended bridge | A genuinely supported BFV/CGGI pair. | A genuinely supported CKKS/CGGI pair. |
| Values | Signed 8-bit inputs/weights; signed 19-bit score and thresholds; signed 22-bit selected-score accumulator; count 0 through 8. | Explicit bounded input/score ranges and numerical error budget. |
| Predicate | Signed `>=`, including equality. No unintended modular wrap. | Correct outside a public margin `delta`; behavior inside the margin must be explicitly accepted by the application. |
| Extra requirement | A subtract-based comparator must accommodate signed 20-bit score-minus-threshold. Actual parameter selection must cover every algorithm intermediate. | The justified total score/switching/quantization error must be strictly below `delta`; no secret-dependent exclusion of boundary cases. |
| Final result | Exact under the admitted correctness/failure policy. | Count and aggregate consistent with permitted masks and the numeric error bound. |

For the exact fixture, a score can reach `8 * (-128) * (-128) = 131072`;
eight selected scores can reach `1048576`. The stated widths include these
positive endpoints. Bit widths alone do not select a valid batched plaintext
modulus or supply a switching protocol.

For the approximate fixture, exact threshold decisions on unrestricted real
values cannot be obtained merely by labeling a conversion exact. If that is the
application's demand, this approximate contract is inapplicable. A system may
not silently choose the quantized fixture as a substitute either.

Required scenarios once actual bridges exist:

1. Separated scores, with both forward conversion and returned packed masks.
2. Boundary cases at and around the declared margin, plus exact integer ties.
3. Positive/negative values and signed extrema, with no hidden wrap or unsigned
   comparison.
4. Two thresholds consuming the same converted scores, retaining packed scores
   for both masked aggregates; trace and lifetime evidence of reuse.
5. New ciphertext inputs with identical descriptors; reuse plan/key setup but
   recreate input-dependent conversion preparation.
6. Fused conversion/predicate/repacking versus separate proposals, including all
   setup, transfer and packing costs and identical final output layouts.

Missing any required key capability, conversion direction or numerical guarantee
makes the complete request unsupported. Abstract resource-graph tests are useful
before that point but are not FHE correctness or performance evidence.
