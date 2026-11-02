# Encrypted personal-document search

This example encrypts both document embeddings and a query embedding, evaluates
cosine similarity with genuine CPU CKKS, and lets the owning client decrypt
scores and rank documents. The bundled vault contains **65 public project
documentation excerpts standing in for private documents**, with eight English
and Korean sample queries. Its vectors come from a real, pinned multilingual
sentence-transformer model. The executable needs no Python, model download,
network connection, or remote embedding service.

## Build and run

From the repository root:

```sh
cmake -S . -B build-search -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=OFF -DBUILD_DOCUMENTATION=OFF
cmake --build build-search --target fhn-private-search ckks_fhn
./build-search/bin/fhn-private-search \
  ./build-search/lib/libckks_fhn.dylib \
  examples/private_search/data/fixture.json
```

Use `libckks_fhn.so` on Linux. The default run uses the first sample query,
three repetitions, and the prepared tiled layout. It displays the query,
retrieved titles/text, and selected recipes. To compare all four execution
paths and verify all sample queries:

```sh
./build-search/bin/fhn-private-search \
  ./build-search/lib/libckks_fhn.dylib \
  examples/private_search/data/fixture.json \
  --compare --verify-all --reps 3 --json /tmp/private-search-results.json
```

The command interface is:

```text
fhn-private-search NATIVE_LIBRARY FIXTURE_JSON
  [--query ID] [--compare] [--reps N] [--verify-all]
  [--limit-documents N] [--json OUTPUT]
```

For example, `--query memory` selects the Korean memory-budget question.
`--limit-documents 9` exercises a full eight-document tile and a one-document
tail on a smaller prefix. `--verify-all` checks all eight bundled queries;
`--compare` evaluates `row-ordinary`, `row-prepared`, `tile-ordinary`, and
`tile-prepared` through the same high-level score request.

## Client and evaluator roles

One owner controls every document and the query, and owns the secret and public
evaluation keys. The client embeds text locally, encrypts both sets of vectors,
prepares evaluation keys, decrypts returned score buffers, and chooses the top
matches. It is already entitled to read every source document.

The evaluator receives ciphertext Values, public shape/layout metadata and
public evaluation-key access. Contributed recipes retain ciphertext/artifact
dependencies and compute services; they do not retain a client, plaintext
embedding, source text, or secret key. The client keeps display text separately.

These roles run **inside one process**. The example demonstrates object and
call-path separation. It does not supply operating-system isolation,
ciphertext serialization or network transport, encrypted top-k, private
retrieval/access patterns, or a production security claim. The evaluator learns
public corpus size, layout, operation structure and preparation availability.
Result decryption and ranking happen on the trusted client.

## Packing and selected recipes

Normalized embeddings have 384 components. Each is zero-padded to a 512-slot
block. A query is repeated across eight such blocks in a ciphertext with 4096
logical and physical slots, using the native default profile.

| Corpus layout | Document placement | Ciphertexts for 65 documents |
| --- | --- | --- |
| Row | One document repeated across all eight 512-slot blocks | 65 |
| Tile | Eight independent documents, one per 512-slot block | 9 |

The final tile contains one valid document and seven zero-filled blocks. The
padded dot product equals cosine similarity because the original vectors have
unit norm. For document `d`, read only result buffer `d / capacity` at slot
`(d % capacity) * 512`, where capacity is one for rows and eight for tiles.
Other lanes contain mixed partial sums and are not document scores. Zero-filled
tail blocks are excluded from ranking.

The separately linked `Fhenomenon::SearchKernels` contribution registers:

- `contrib.search.index`: retain validated ciphertext blocks and public
  `SearchShape{width, document_count, documents_per_ciphertext}` metadata.
- `contrib.search.prepare-query`: create explicit reusable preparation for the
  exact query Value and reduction width.
- `contrib.search.score`: compose collection scoring from block requests,
  sharing the same query preparation across every block.
- `contrib.search.score-block`: select a compatible row/tile and
  ordinary/prepared recipe using public metadata and explicit preparation.

The application submits the same program for either layout:

```cpp
registerSearchKernels(runtime, services, nativeLibraryPath);
auto index = runtime.submit({"contrib.search.index", encryptedBlocks,
                             SearchShape{512, documentCount, capacity}}).at(0);
auto prepared = runtime.submit({"contrib.search.prepare-query",
                                {encryptedQuery}, std::uint32_t{512}}).at(0);
auto scores = runtime.submit({"contrib.search.score",
                              {encryptedQuery, index, prepared}, {}});
```

Use capacity one or eight for the layouts above, or omit the preparation input
to select ordinary block recipes. The trusted client packs the matching layout;
the runtime selects implementations from the submitted public contract.

Every path performs an encrypted elementwise product and a width-512
rotate-add reduction. Prepared paths use the existing matvec preparation and
dot contribution. The one-document tail can select the row block recipe even
though its enclosing index uses tiled storage. Selection is deterministic
recipe dispatch; it is not encrypted-data-dependent selection or automatic
performance tuning. Query preparation is bound to its original Value identity;
a fresh encryption with the same dimensions is a different source.

Returned scores are ordinary native ciphertext Values and can enter existing
native commands. The index retains its ciphertext dependencies once rather
than copying plaintext or retaining the client. The runtime core, native
adapter, backend and cryptographic arithmetic require no search-specific edit.

## Validation and costs

The independent client oracle computes plaintext dot products from every
original vector. Numeric agreement with encrypted scores, top-k agreement,
near-tie handling, and the sample's hand-labeled relevance are distinct checks.
Every warm-up and timed evaluation is decrypted and checked outside the evaluation
timer. Fresh queries, or limited prefixes containing no labeled documents, show
relevance as unjudged; their encrypted scores are still verified.
A numerically correct computation does not establish search quality. The
bundled relevance labels are small, non-exhaustive judgments written before
model scoring: seven of eight queries have a labeled result in their plaintext
top three. The data README records the missed query and the model's truncation
limit.

Compare index setup, query setup/preparation, evaluation, and result decoding
separately. Include retained and peak tracked bytes as well as live ciphertext
bytes/counts. JSON separates native live payload from reserved ciphertext capacity;
preparation snapshots are counted in preparation bytes rather than again as
ciphertext Values. Peak tracked bytes are each mode runtime's lifetime peak across
queries, not a per-query peak. Runtime tracked slabs exclude keys, context tables/scratch and
allocator overhead; they are not process memory. Ciphertext payload size is
not measured network traffic. Reusable preparation has a setup and retained
memory cost that belongs in any one-shot comparison. Use the JSON output to preserve
evidence from the machine and configuration actually tested.
The four modes use independently randomized encryptions under the same key and
the same 4096-slot representation. All four indexes remain resident during a
comparison. This row layout is a readable baseline, not a claim of the best
possible row implementation. Query repetition, document placement and common-key
encryption are trusted-client contracts; public metadata cannot prove them.

## Recorded run

[Raw measurements and environment](../../benchmarks/results/2026-11-02-private-search-release.json)
record an Apple M4 Pro, macOS 26.6.2, Apple clang 21 Release build with LTO disabled.
There were no concurrent assistant builds/tests/inference during measurement.
The table pools 24 warm evaluation samples per mode: eight queries, three
repetitions each, after one warm-up per query/mode. All modes evaluate the same
65-document corpus and original dot-product formula.

| Runtime-selected path | Corpus ciphertexts | Evaluation median | Live corpus payload |
| --- | ---: | ---: | ---: |
| Row, ordinary | 65 | 1390.4 ms | 24.375 MiB |
| Row, prepared | 65 | 665.1 ms | 24.375 MiB |
| Tile, ordinary | 9 | 192.3 ms | 3.375 MiB |
| Tile, prepared | 9 | 92.0 ms | 3.375 MiB |

Packing eight documents per tile reduces the number of block evaluations from
65 to nine. Within the same tiled layout, shared query preparation improves the
median evaluation by **2.09×**. Tiled preparation itself takes a median
**13.8 ms** and retains **16.03 MiB**; it is reused across all nine blocks and
repetitions. Tiled corpus packing/encryption/import/index creation took about
20.5 ms, query packing/encryption/import about 2.3 ms, and final-result
decryption/extraction about 4.4 ms. Initial library/context/client/evaluation-key
setup took 418.0 ms. These are separate costs, not part of the evaluation table;
embedding inference is not run by the C++ executable.

Every warm-up and timed evaluation passed the numeric oracle: 8,320 document
scores across 128 evaluations. Maximum absolute error was **1.88e-7**, below
the 1e-5 tolerance. All 32 query/mode combinations had the exact same top-five
order as plaintext. Hand-labeled hit@3 remained 7/8, independent of execution
path. The tiled trace includes both tile and row recipes because the final
single-document tail shares the same query preparation.

Debug and ASAN/UBSAN suites passed 46/46 tests, including all queries on a
17-document prefix; final runner changes were rechecked in both builds.
The full-corpus Release run above supplies the 65-document evidence. The
installed public-interface consumer passed separately, and the installed
example ran offline from a `BUILD_TEST=OFF` installation. Runtime, NativeRuntime
and all native CKKS source files remained byte-identical to the base revision.

## Bring your own documents or query

The default executable reads the checked-in fixture. Optional Python tooling
can regenerate it or embed replacement text entirely on the client. See
[data/README.md](data/README.md) for the exact fixture schema, model revision,
source hashes, isolated dependency setup, offline model loading and fresh-query
commands. In particular, export with `--corpus`, optional repeated `--query`,
`--local-model-path`, `--local-files-only`, and `--output`, then pass the new JSON
file as `FIXTURE_JSON` above. A custom corpus or fresh query requires an
explicit `--output` destination. Keep personal corpus/export files outside the
repository. Unjudged fresh queries have empty relevance lists.

## Contribute another block implementation

An external C++ contribution can link the installed
`Fhenomenon::SearchKernels` target and include `FHN/contrib/SearchKernels.h`
with the existing public runtime headers. Register a recipe for
`contrib.search.score-block` using `Runtime::addRecipe`; give it priority greater
than 20 to take precedence over the built-in prepared recipes when applicable.
The collection recipe submits each block through the registry, so compatible
external kernels participate without modifying the collection or runtime core.

The request carries a query, a document block, optional typed query preparation,
and `SearchShape`. A recipe must validate its supported public context, basis,
scale, dimensions, valid-document count and preparation contract. It must return
one ordinary score ciphertext with valid scores at the documented block-start
lanes, preserve escaped inputs, obey resource accounting, and retain no
plaintext/client/secret state. An applicability decline allows another recipe;
an exception after execution begins is reported without retrying another
implementation. See [the runtime contract](../../docs/lightweight-runtime.md)
and [prepared matvec](../../docs/matvec-preparation.md) for shared artifacts and
intermediate-command composition.
