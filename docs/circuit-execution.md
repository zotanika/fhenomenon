# Native circuit execution

The optional circuit API describes application values and compiles them into a
reusable native CKKS execution plan. `MULTIPLY`, `SCALE`, and `ROTATE` are graph
semantics, not promises about kernel boundaries. The compiler can combine a
whole affine expression or a sum of products before choosing arithmetic and
storage. The existing `FhnProgram` opcode interface remains available with its
original physical semantics.

Measured application coverage and timings: [validation report](circuit-execution-validation-2026-10-28.md).

## Public API

Include `FHN/fhn_circuit_api.h`. The native plugin exports this complete optional
group, resolved separately from the ABI 1 kernel table:

1. `ckks_fhn_compile_circuit`: copy a graph and encrypted input signatures into
   an executable. Input values are not specialized into the plan.
2. `ckks_fhn_client_prepare_executable`: prepare the public evaluation keys
   required by that plan through the separate trusted client.
3. `ckks_fhn_run_executable`: bind input/output buffers and evaluate repeatedly.
4. `ckks_fhn_executable_stats`: report region counts and reserved slot/scratch
   bytes.
5. `ckks_fhn_executable_free`: release the plan before destroying its context.

Calls on a context/executable are serialized by the caller. Inputs can be
re-encrypted or rebound if their owner, packing, level and scale match the
compiled signatures. Execution performs no heap allocation. Public outputs are
published after evaluation succeeds; input/output aliasing is allowed. Different
output values need different destination buffers. The client owns secret and
public evaluation-key objects; the evaluator retains only weak public-key
references and cannot decrypt. Destroying the client expires those keys.

The circuit consists of explicit input IDs, topologically ordered value nodes,
and output IDs. Nodes support addition, subtraction, multiplication, scalar
scaling, scalar addition, negation, rotation and conjugation. The current native
lowering uses exact signed integer coefficients for affine regions; fractional
scaling uses a modulus-consuming path. Unsupported scale joins, insufficient
levels and out-of-range constants fail with diagnostics, without scale metadata
relabeling or a plaintext fallback. This first compiler bounds inputs plus nodes
and output ports to 65,536 each, affine forms to 256 terms, and recursive
materialization depth to 512; expressions exceeding these bounds are rejected
with diagnostics.

## Execution regions

Integer affine operations preserve level and scale. Compatible sums of products
can accumulate one three-component tensor and then perform one key switch and
rescale. Linear terms in such a region are lifted by the common scale using
modular arithmetic. Rotations can absorb compatible affine additions.

The native multiplication kernel also combines auxiliary-modulus removal and
rescaling into one exact centered division. Its residues are tested against the
separate tensor/relinearize/rescale path on identical inputs and keys. This
bit-exact statement applies to that kernel rewrite; graph-level scheduling can
change CKKS rounding/noise while retaining the application accuracy contract.

Storage slots are reused after their last consumers. A region destination never
overlaps live operands. Internal ciphertexts use their active Q basis and two
components; they need not reserve the public buffer's full three-component QP
capacity. `workspace_limit_bytes` bounds reserved internal slot and scratch
bytes. It excludes public I/O buffers, the backend's existing context scratch,
key/table storage, graph metadata, and allocator overhead; it is not a process
memory limit. Statistics report slot and scratch bytes separately.

## Corpus runner

`fhn-bench-circuit` imports the existing corpus's mathematical semantics into
circuits. This is an explicit application importer, not a generic rewrite of
arbitrary physical `FhnProgram` sequences. Explicit relinearize/rescale/level
operations and exact Boolean operations are rejected by that importer.

```sh
fhn-bench-circuit --backend /path/to/libckks_fhn.dylib --prefix ckks_ \
  --shape matvec-fused --warmup 2 --reps 7 --require-execution
```

Use the platform's shared-library extension. Omit `--shape` to check the whole
corpus. `--require-execution` makes any skipped shape an error; zero executed
shapes or incorrect results always fail. `UNSUPPORTED` includes a compile
rejection such as an insufficient workspace budget, with the concrete reason.
`--workspace-bytes`, `--config`, `--atol` and `--rtol` control those explicit
constraints. Defaults are the native profile and `atol=1e-4, rtol=1e-6`.

An optional `--baseline /path/to/preserved/plugin` compares with the legacy
interpreter using preallocated SSA buffers. Timing alternates baseline/current
order, verifies all output slots before and after repetitions, and times only
completed evaluation. Context/client creation, encryption, compilation,
evaluation-key preparation and validation decryption are reported separately.
`decrypt_verify_ms` is total decryption time across verification passes.
Baseline execution failures are reported explicitly; an app newly enabled by
compilation has no baseline speedup. The plaintext oracle evaluates the original
corpus graph, independently of the imported circuit and native execution.

The current user-owned `examples/resnet_circuit.cpp` runs plaintext vectors and
ReLU, so it is not included as encrypted application evidence. The ordinary
`Fhenon`/`Session` frontend still defaults to its existing backend; using this
circuit path is explicit. This implementation does not add bootstrapping, GPU
execution, serialization, or a security certification.
