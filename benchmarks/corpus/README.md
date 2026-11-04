# Portable packed verification

`fhn-corpus` plans 13 workloads and optionally executes them through the public
FHN C ABI. `matvec-fused` encrypts an 8-element vector and eight matrix rows,
then evaluates eight dot products using `HMULT` and `HROT_ADD`. Every output
slot is checked against the plaintext oracle; a unit test also checks an
independent row-by-column dot product.

## Genuine native CPU CKKS

The default build includes `ckks_fhn`; no external FHE library or GPU is needed.
On macOS (use `.so` on Linux):

```sh
build/bin/fhn-corpus --backend build/lib/libckks_fhn.dylib --prefix ckks_ \
  --shape matvec-fused --require-execution --compare-fallback \
  --atol 1e-4 --rtol 1e-6 --budget-bytes min
```

The loader creates an isolated trusted client, derives the program's required
evaluation keys, and encrypts fresh inputs for each path. Kernel execution has
no secret-key reference. See [native configuration and limits](../../docs/native-ckks.md).
`--max-depth` counts ciphertext multiplications only; native scalar multiplication
also consumes a level. Several older planning shapes contain raw tensor products
without CKKS relinearization/scale alignment, so they are not all executable CKKS
circuits. Select a validated shape instead of treating a successful plan as an
execution guarantee.

## Reusable application circuits

`fhn-bench-circuit` explicitly imports the corpus mathematical operations into
the new semantic circuit API. Under the unchanged native default profile it
executes matvec, matvec-fused, matmul-tile, reduce-tree, weighted-sum, diamond,
conv1d, iter-update, stats and wide-front. Horner15/logreg need more depth;
psi-eq needs exact Boolean operations.

```sh
build/bin/fhn-bench-circuit --backend build/lib/libckks_fhn.dylib --prefix ckks_ \
  --shape wide-front --warmup 2 --reps 7 --require-execution
```

The integer-scale rule and managed products belong to this explicit semantic
path; the old opcode meanings are unchanged. See [execution and measurement
contracts](../../docs/circuit-execution.md). Optional `--baseline LIB` alternates
preallocated legacy evaluation with compiled evaluation. Setup and trusted
crypto I/O are outside evaluation timing. There is no speedup claim for a shape
that the baseline cannot execute.

## CPU reference check

After building, run on macOS (use `.so` instead of `.dylib` on Linux):

```sh
build/bin/fhn-corpus --backend build/lib/libtoyfhe_fhn.dylib \
  --shape matvec-fused --require-execution --compare-fallback
build/bin/fhn-corpus --backend build/lib/libtoyfhe_fhn.dylib \
  --shape matvec --require-execution
```

The first command runs native fused kernels and their complete primitive
decompositions with fresh inputs, checks each against the oracle, and compares
their outputs. The second checks all 64 slots of the original matvec shape.

## Interpretation and limits

- The summary separates `planned`, successfully verified `executed`,
  `skipped`, and `failed` shapes. Planning counts overlap the other categories.
  Skips have explicit reasons. `--require-execution` fails for any skip or
  failure, or if nothing executes. Comparing fallback for a shape with no
  fused operations is a skip.
- Without a backend, results are planning estimates. Savings against LRU
  are neither measured device transfers nor elapsed-time speedups. Belady's
  varying-size byte-budget policy is a heuristic.
- `--budget-bytes N|min` uses the backend's fresh-level size model for both
  planning and execution. Statistics start after input residency normalization;
  initial evictions are separate plan metadata. Entry residency, scratch,
  reserved memory and transient three-component ciphertexts depend on the backend's declared model. Native CKKS reports its full fixed three-component slab at
  every level, so shrinking live payload never implies released capacity.
  This CPU backend has no physical eviction; key/table/scratch RAM is outside
  the movement model and the budget is not a process-memory limit.
  ToyFHE's abstract per-buffer cost does not account for packed vector storage.
- Optional `fhn_encrypt_vec_f64` / `fhn_decrypt_vec_f64` exports preserve ABI1
  and its vtable layout. The count is positive and exact on decryption, with
  no padding/truncation. Scalar-only backends remain loadable. `--config`
  passes file contents to `fhn_create`. The optional all-or-nothing
  `fhn_client_*` group keeps secret material in a separate trusted handle.
  Evaluation-only backends load but skip execution without a usable host data plane.
- Floating-point checks use

  $$
  \lvert \mathrm{actual}-\mathrm{expected}\rvert
  \le \mathrm{atol}+\mathrm{rtol}\,\lvert \mathrm{expected}\rvert.
  $$

  `--atol` and `--rtol` default to `1e-6`; nonfinite outputs fail. Integer
  scalar I/O retains exact comparisons.
- ToyFHE is an insecure CPU reference. Packed fixed-point values have a step
  of $\frac{1}{65536}$ and centered magnitude below roughly $32768$. Intermediate
  overflow is possible even in shallow graphs. The current packed `stats`
  and `logreg` exceed that range and correctly fail verification; tests pin
  this behavior. `--max-depth` is a filter, not a numerical safety proof.
  The broad smoke test uses depth zero; dedicated required checks cover
  `matvec-fused` and `wide-front`.

## Cheddar GPU check

On a CUDA machine with Cheddar and `cheddar_fhn` built, prepare rotation keys
for the 8-slot reduction and run:

```sh
python3 - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('refs/cheddar-fhe/parameters/testparam_light.json').read_text())
config['rotation_keys'] = [1, 2, 4]
Path('/tmp/fhenomenon-matvec.json').write_text(json.dumps(config))
PY
build/bin/fhn-corpus --backend build/lib/libcheddar_fhn.so --prefix '' \
  --config /tmp/fhenomenon-matvec.json --shape matvec-fused \
  --atol 1e-3 --rtol 1e-3 --require-execution --compare-fallback
```

Cheddar accepts power-of-two lengths up to half its polynomial degree.
`experiments/run.sh corpus` supplies the empty symbol prefix; pass the same
configuration and verification options. `experiments/run.sh gpu-test` fails
if no GPU tests match.

GPU smoke timings synchronize completion at both boundaries and use a
monotonic clock. They measure completed pipeline host wall time, not isolated
device kernels or statistically sampled performance. Old enqueue-only numbers
are not comparable. GPU compilation, correctness and timing remain unverified
on this CPU-only machine. Native CPU CKKS success does not establish GPU correctness or performance.
