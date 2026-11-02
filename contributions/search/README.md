# Search computation and contribution contract

Start with [ScoreBlock.h](ScoreBlock.h) to read the encrypted computation.
[SearchKernels.cpp](SearchKernels.cpp) checks applicability, retains artifacts and
composes collection requests from those block implementations. The application
entry point is [the private-search example](../../examples/private_search/main.cpp);
its evaluator submits the requests in
[EncryptedSearch.cpp](../../examples/private_search/EncryptedSearch.cpp).

These are immediate runtime programs. No recorded circuit or compiler is needed
to execute them. A compatible larger contributed kernel can implement the same
request directly; the multiply/rotate/add sequence is one implementation of the
contract, not a required decomposition for every contribution.

## What is encrypted here?

The owning client encrypts normalized document and query embeddings under one
key. The evaluator receives ciphertext Values, shape metadata and access to
public evaluation keys. It produces encrypted scores. It never embeds text,
decrypts a value, compares encrypted scores, or selects the final top-k. Those
steps belong to the client; see the short application workflow linked above.

For padded width `W`, the block request promises:

```text
score[j] = sum(query[t] * document[j][t], t = 0 .. W-1)
```

This equals cosine similarity for L2-normalized vectors, up to CKKS error.
The default example has 384 meaningful components followed by 128 zeros, so
`W = 512`. It repeats the query eight times in 4096 slots. A document block
contains either one repeated document or up to eight distinct documents.

## Ciphertext and layout contract

`contrib.search.score-block` takes `{query, document_block[, preparation]}` and
`SearchShape{width, valid_document_count, documents_per_ciphertext}`. It returns
one native ciphertext Value.

| Property | Requirement or result |
| --- | --- |
| Width | Positive power of two, fitting the logical slot count |
| Capacity | Positive; `width * documents_per_ciphertext <= logical_slots` |
| Valid documents | Between one and capacity for a block request |
| Query packing | The same width-slot query repeats across logical slots |
| Document packing | Document `j` occupies width-slot block `j`; unused tail blocks are zero |
| Input context and key | Same native context and owning encryption key |
| Input metadata | Two components, equal logical slot counts and levels `L >= 1`; scales match within relative `1e-12` |
| Output metadata | Two components at level `L-1`; the native multiplication/rescale scale |
| Valid output lanes | Only slot `j * width`, for `j < valid_document_count` |

If both input scales are `s`, the product is rescaled by the dropped modulus
prime, producing scale `s*s/q_L`. The default profile starts at level two; this
search consumes one level. Rotation and addition preserve that product's level
and scale. No implicit scale/level alignment, bootstrapping or scheme switching
occurs here.

The ordinary block implementation multiplies the encrypted vectors, then adds
positive rotations at distances `W/2, W/4, ..., 1`. Native positive rotation
moves slot `i+distance` to slot `i`. At a block-start lane this sums exactly the
document's `W` products. Other lanes contain sliding sums that may cross document
boundaries; **they are not document scores and must not enter ranking**.

Collection scoring uses `ceil(document_count / capacity)` ciphertext blocks.
Document `d` is read from result buffer `d / capacity`, lane `(d % capacity)*W`.
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
