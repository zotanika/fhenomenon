# Search computation and contribution contract

Start with [ScoreBlock.h](ScoreBlock.h) to read the encrypted computation.
[SearchKernels.cpp](SearchKernels.cpp) checks applicability, retains artifacts and
composes collection requests from those block implementations. The application
entry point is [the private-search example](../../examples/private_search/main.cpp);
its evaluator records the scoped program in
[ScopedSearch.cpp](../../examples/private_search/ScopedSearch.cpp). The immediate
alternative remains in [EncryptedSearch.cpp](../../examples/private_search/EncryptedSearch.cpp).

The ordinary recipes execute immediately; the optional native Session scope
records their requests before bounded span selection. Neither path needs a
recorded circuit or compiler. A compatible larger contributed kernel can implement the same
request directly; the multiply/rotate/add sequence is one implementation of the
contract, not a required decomposition for every contribution.

## What is encrypted here?

The owning client encrypts normalized document and query embeddings under one
key. The evaluator receives ciphertext Values, shape metadata and access to
public evaluation keys. It produces encrypted scores. It never embeds text,
decrypts a value, compares encrypted scores, or selects the final top-k. Those
steps belong to the client; see the short application workflow linked above.

For padded width $W$, the block request promises:

$$
\mathrm{score}_j = \sum_{t=0}^{W-1} \mathrm{query}_t\,\mathrm{document}_{j,t}.
$$

This equals cosine similarity for $\ell_2$-normalized vectors, up to CKKS error.
The default example has 384 meaningful components followed by 128 zeros, so
$W = 512$. It repeats the query eight times in 4096 slots. A document block
contains either one repeated document or up to eight distinct documents.

## Ciphertext and layout contract

`contrib.search.score-block` takes `{query, document_block[, preparation]}` and
`SearchShape{width, valid_document_count, documents_per_ciphertext}`. It returns
one native ciphertext Value.

| Property | Requirement or result |
| --- | --- |
| Width | Positive power of two, fitting the logical slot count |
| Capacity | Positive; $\mathtt{width}\cdot\mathtt{documents\_per\_ciphertext} \le \mathtt{logical\_slots}$ |
| Valid documents | Between one and capacity for a block request |
| Query packing | The same width-slot query repeats across logical slots |
| Document packing | Document $j$ occupies width-slot block $j$; unused tail blocks are zero |
| Input context and key | Same native context and owning encryption key |
| Input metadata | Two components, equal logical slot counts and levels $L \ge 1$; scales match within relative $10^{-12}$ |
| Output metadata | Two components at level $L-1$; the native multiplication/rescale scale |
| Valid output lanes | Only slot $j\cdot\mathtt{width}$, for $j < \mathtt{valid\_document\_count}$ |

If both input scales are $s$, the product is rescaled by the dropped modulus
prime, producing scale $\frac{s^2}{q_L}$. The default profile starts at level two; this
search consumes one level. Rotation and addition preserve that product's level
and scale. No implicit scale/level alignment, bootstrapping or scheme switching
occurs here.

The ordinary block implementation multiplies the encrypted vectors, then adds
positive rotations at distances $W/2, W/4, \ldots, 1$. Native positive rotation
moves slot $i+\mathrm{distance}$ to slot $i$. At a block-start lane this sums exactly the
document's $W$ products. Other lanes contain sliding sums that may cross document
boundaries; **they are not document scores and must not enter ranking**.

Collection scoring uses $\left\lceil\frac{\mathtt{document\_count}}{\mathtt{capacity}}\right\rceil$ ciphertext blocks.
Document $d$ is read from result buffer $\left\lfloor\frac{d}{\mathtt{capacity}}\right\rfloor$, lane $(d \bmod \mathtt{capacity})W$.
A final one-document tile selects the row recipe while retaining the tile's
original capacity and result mapping.

Public metadata checks cannot prove query repetition, document placement, zero
padding or common-key encryption. These are trusted-client packing obligations.
The example's owner is entitled to read all documents, including non-score lanes
after decryption; this contract is not a scores-only disclosure mechanism for a
different document owner.

## Evaluation keys, not a circuit

The public helper in [SearchKernels.h](../../include/FHN/contrib/SearchKernels.h)
describes the built-in native kernels' requirements:

```cpp
const auto keys = nativeSearchKeyRequirements(512);
// keys.multiplication == true
// keys.rotation_steps == {1, 2, 4, 8, 16, 32, 64, 128, 256}
```

Width one needs multiplication keys and no rotation keys. Invalid widths are
rejected. This helper does not choose CKKS parameters or establish that a width
fits a particular context. The native backend turns the multiplication request
into the required relinearization capability. The client adapts this list to
the backend's `FhnProgram` key-preparation API; that object is an evaluation-key
request, not the executed search circuit.

Both built-in ordinary and prepared paths need these capabilities. The client
owns the evaluation keys and must keep them alive through scoring; compute
contexts only borrow their leases. No recipe retains a client or a secret key.

This helper is **not automatic key discovery for arbitrary registered recipes**.
A replacement kernel that needs different rotations or capabilities must expose
its requirements to the caller and arrange their preparation before execution.
Changing the selected implementation cannot silently create missing keys.

## Prepared state and mixed granularity

`contrib.search.prepare-query` wraps a `contrib.matvec.prepare` artifact. The
wrapper binds the exact encrypted query identity, reduction width and provider.
A fresh encryption of the same plaintext is a different query; passing its
predecessor's preparation is rejected. Raw matvec artifacts or copied wrapper
payloads are not accepted as search preparations.

The same preparation is shared by the collection's row/tile requests, including
the final partial tile, and can be reused on subsequent score submissions.
[The prepared block implementation](ScoreBlock.h) delegates to the contributed
matvec dot request. Its native realization is in
[PreparedMatvec.cpp](../../src/CKKS/backend/PreparedMatvec.cpp); its state and
reuse contract are described in [prepared matvec](../../docs/matvec-preparation.md).
It preserves the ordinary path's score, level, scale and rounding behavior.

Index dependencies retain the encrypted document Values; preparation dependencies
retain their exact source and native preparation. Copies share charges rather
than duplicating buffers. Returned score Values remain ordinary native values
that other registered commands can consume. No intermediate needs to be
decrypted merely to cross a kernel boundary.

## Native Session scopes

[ScopedSearch](../../examples/private_search/ScopedSearch.h) records a separate
`contrib.search.prepare-query` command followed by one
`contrib.search.score-block` command per ciphertext. It accepts only ciphertexts
and public layout metadata. The capture callback runs once; no registered scope
kernel receives the client or plaintext.

The same sealed provider registers two scope contributions:

| Kernel | Priority | Selected span |
| --- | --- | --- |
| `contrib.search.scope.collection` | 20 | Two through eight consecutive prepared full blocks |
| `contrib.search.scope.block` | 10 | One prepared full or tail block |

Collection commands must share the same symbolic query and preparation inputs,
width and capacity. Their native metadata must be compatible, and their concrete
preparation must bind the exact query, width and provider. Each command must
define exactly one score output. The collection stops before a tail, a different
layout or an incompatible command. Unprepared commands use ordinary dispatch;
`ScopeOptions::use_kernels = false` also preserves that path.

Execution composes the existing block requests through `Runtime::submit`, so
higher-priority ordinary block recipes still take effect inside the span. Each
block is computed, and only outputs requested by the selected span's liveout
contract are returned. Exported intermediate scores and downstream consumers
retain the same ciphertexts. This contribution orchestrates multiple recorded
commands; it introduces no cryptographic fusion or speedup claim.

The example can explicitly export both preparation and scores:

```cpp
auto session = Session::create(runtime);
auto values = session->run([&](Scope &scope) {
  ScopedSearch search(scope);
  auto preparation = search.prepareQuery(query);
  auto scores = search.score(query, documents, preparation);
  return search.exportResults(preparation, scores);
});
auto result = ScopedSearch::results(std::move(values), documents.shape);
```

Reuse requires importing `result.preparation.artifact` into the next scope with
`scope.input(...)`, then passing that symbol to `score`. This shares the exact
concrete artifact identity; no identity copy, hidden cache or automatic import
occurs. Symbols themselves remain local to their original recording.

A third-party `ScopeKernel` with priority above 20 can replace these search
spans without a runtime-core edit. It must return exactly the concrete values
listed by `view.liveOuts(view.size())`, including exported intermediates. A
selected executor failure is final; it cannot fall back to another kernel.

## Where a new contribution fits

Link `Fhenomenon::SearchKernels` and register a recipe for
`contrib.search.score-block`. A priority above 20 lets an applicable replacement
precede the built-in prepared recipes. Collection scoring submits each block
through the registry, so a replacement needs no runtime-core edit. A contribution
can instead target collection-level `contrib.search.score` when it has a useful
larger scope and its own supported index/preparation contract.

Validate the public inputs and supported preparation before execution. Preserve
escaped inputs, retain/account for resources, and return a compatible native
score ciphertext with the lane semantics above. An applicability decline allows
another recipe to be considered. A failure after execution begins is reported;
the runtime does not retry another implementation with hidden extra work.

[SearchContributionTest.cpp](../../test/SearchContributionTest.cpp) exercises
third-party block replacement, ordinary-command interoperability, row/tile and
tail agreement, key requirements, source/provider/width binding and failure
cleanup. The general callback and ownership rules are in
[the runtime contract](../../docs/lightweight-runtime.md).
[SearchScopeTest.cpp](../../test/SearchScopeTest.cpp) checks real encrypted
Session execution with collection, tail and ordinary native commands, exact
preparation reuse, bounded spans, liveouts, disabled/unprepared paths and
external scope/block replacements.
