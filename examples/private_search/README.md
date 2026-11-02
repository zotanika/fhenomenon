# Encrypted personal-document search

Start with [main.cpp](main.cpp). It shows the owning client encrypting documents
and a query, an evaluator computing encrypted similarity scores, and the client
decrypting and ranking them. Both sets of embeddings are encrypted with native
CPU CKKS. The bundled sample contains 65 public documentation excerpts standing
in for a private vault, with eight English/Korean queries and real, pinned
multilingual sentence-transformer embeddings.

## Read the application, then follow the encrypted computation

| Reader's question | Entry point |
| --- | --- |
| How do I use encrypted search in an application? | [main.cpp](main.cpp): explicit client/evaluator steps, without timing or a plaintext oracle |
| Which values are encrypted, and who can decrypt? | [SearchWorkflow.h](SearchWorkflow.h): example-only ciphertext types and separate capabilities |
| What program does the evaluator submit? | [EncryptedSearch.cpp](EncryptedSearch.cpp): index, shared query preparation and score requests |
| Where does actual FHE computation happen? | [ScoreBlock.h](../../contributions/search/ScoreBlock.h): ordinary encrypted calculation and the prepared-kernel alternative |
| What must a contributed kernel preserve? | [Search contribution contract](../../contributions/search/README.md): score meaning, packing, keys, level/scale, reuse and replacement |
| How are keys prepared and ciphertexts created? | [TrustedClient.cpp](TrustedClient.cpp): client-only key setup, packing, encryption and decryption |
| Where are correctness checks and measurements? | [benchmark.cpp](benchmark.cpp): plaintext oracle, four paths, repeated evaluation and JSON diagnostics |

The important boundary in the application is:

```text
Client (plaintext + secret key)      Evaluator (ciphertext + public metadata)
  encryptDocuments -----------------> createIndex
  encryptQuery ---------------------> prepareQuery
                                      score
  decryptScores <-------------------- encrypted scores
  topK and display
```

`EncryptedDocuments`, `EncryptedQuery`, `EncryptedIndex`, `QueryPreparation`
and `EncryptedScores` make the values crossing that boundary explicit. They are
small example-level types over the existing runtime, not a new compiler or wire
protocol. Only the client methods encrypt or decrypt; ordinary `topK` runs on
already-decrypted scores. The evaluator's methods never take the client, source
text, plaintext embeddings or a secret key.

`score` composes block requests whose implementations are selected at runtime.
There is no single recorded circuit in main. The ordinary block implementation
in `ScoreBlock.h` computes a ciphertext product and a rotate/add reduction; the
prepared implementation delegates the same score contract to a larger matvec
kernel. That sequence is a readable implementation, not an instruction set
that every contribution must use. The same query preparation survives across
block requests, including the single-document tail.

Index creation retains ciphertext references and public metadata. Query
preparation constructs reusable native state without decrypting or producing
scores. Actual encrypted score arithmetic starts with the block implementations.

## Build and run the application

From the repository root:

```sh
cmake -S . -B build-search -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=OFF -DBUILD_DOCUMENTATION=OFF
cmake --build build-search --target fhn-private-search fhn-bench-private-search ckks_fhn
./build-search/bin/fhn-private-search \
  ./build-search/lib/libckks_fhn.dylib \
  examples/private_search/data/fixture.json --query memory
```

Use `libckks_fhn.so` on Linux. Both executables build with tests disabled and are
installed along with the fixture. Running them needs no Python, model download,
network connection or remote embedding service.

```text
fhn-private-search NATIVE_LIBRARY FIXTURE_JSON [--query ID]
```

The default is the first query with prepared eight-document tiles. The demo
prints the client's top-five matches and the selected search recipe names.
Use the benchmark executable for layout comparisons and numerical verification:

```sh
./build-search/bin/fhn-bench-private-search \
  ./build-search/lib/libckks_fhn.dylib \
  examples/private_search/data/fixture.json \
  --compare --verify-all --reps 3 --json /tmp/private-search-results.json
```

```text
fhn-bench-private-search NATIVE_LIBRARY FIXTURE_JSON
  [--query ID] [--compare] [--reps N] [--verify-all]
  [--limit-documents N] [--json OUTPUT]
```

The previous diagnostic flags moved to `fhn-bench-private-search`. The demo
rejects them with a pointer to that executable rather than silently ignoring
them. `--limit-documents 9` on the benchmark exercises a full tile and a
one-document tail. All four paths submit the same high-level score request:
row-ordinary, row-prepared, tile-ordinary and tile-prepared.

## Roles, keys and execution scope

One owner controls every document and query, and keeps the secret and public
evaluation keys alive through evaluation. The client embeds text locally before
running the demo; its fixture already contains embeddings. It is entitled to
read every document and owns the display text separately from the evaluator.

The two roles explicitly share one `Runtime` and native context inside one
process. This is necessary because Values carry runtime-specific accounting and
identity. Sharing the runtime preserves preparation identity without hidden
imports or caches. Compute recipes receive only compute services and ciphertext
resources. `EncryptedSearch` is a view of this shared recipe registry, not a
separate provider identity or namespace. This does not establish operating-system isolation or transport;
the example has no ciphertext serialization, encrypted top-k or private
retrieval/access-pattern protocol, and makes no production security claim.

The client gets default kernel requirements from
`nativeSearchKeyRequirements(width)`. Its internal `FhnProgram` named
`evaluation_key_request` adapts that list to the native key-preparation API;
**it is not the search circuit**. The helper covers the built-in native kernels.
A new kernel with different key requirements must arrange those capabilities
before evaluation; runtime dispatch does not automatically discover or create
keys. See the contribution contract for details.

## Packing and validation

The 384-dimensional normalized vectors are padded to width 512. The query repeats
across eight blocks in 4096 slots. Row layout repeats one document in a whole
ciphertext (65 ciphertexts); tile layout places eight independent documents in
each ciphertext (nine ciphertexts, including one partial tile).

Only lane `j * 512` contains document `j`'s score in a result block. Other lanes
contain mixed sliding sums. The trusted client extracts only valid score lanes
before ranking, and excludes the zero-filled tail. Common-key encryption,
query repetition and document placement are client contracts that public
metadata cannot prove. This single-owner example is not a scores-only
information-disclosure mechanism for another document owner.

The separate benchmark compares every warm-up and timed result against an
independent plaintext dot-product oracle outside the evaluation timer. It
reports exact/tolerance-aware top-five agreement separately from hand-labeled
relevance. Unjudged queries have null relevance metrics. Setup, query
preparation, warm evaluation, decrypt/extract and ranking costs are separate.
Native live ciphertext payload and reserved capacity are distinct; preparation
snapshots are counted in preparation bytes. The runtime peak is a mode's lifetime
peak, not a per-query measurement or process RSS. All four indexes coexist, and
the row representation is a readable baseline, not an optimal-row claim.

## Recorded run before this readability refactor

These measurements were captured by the original combined runner, committed as
`fcd296a`. Its measurement code now lives in `benchmark.cpp` and runs as
`fhn-bench-private-search`. The JSON source hashes identify that historical code;
the figures below are not a new timing claim for the refactored files.

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

## Contribute a kernel

Follow [the search contribution contract](../../contributions/search/README.md)
from the encrypted block implementation to public recipe registration. A
compatible block replacement or a larger collection implementation can join
through public interfaces, preserve shared artifacts and return ordinary native
ciphertext Values. No runtime-core change is needed.
