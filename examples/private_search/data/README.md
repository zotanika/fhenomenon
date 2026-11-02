# Public documentation embedding fixture

These are public Fhenomenon project documentation excerpts **standing in for a
single owner's private document vault**. They are not private user notes. The
sample encrypted search executable consumes the checked-in `fixture.json`
without Python, model weights, a network connection, or access to source files.
Do not use this small sample as a search-quality benchmark.

## Contents and provenance

- `corpus.json`: 65 verbatim documentation paragraphs and eight queries without
  vectors. Five queries are English and three are Korean. The paragraph selection
  includes 16 chunks from `docs/lightweight-runtime.md`, 10 from
  `docs/matvec-preparation.md`, 15 from `docs/native-ckks.md`, 11 from
  `docs/circuit-execution.md`, and 13 from `docs/coarse-statistics-kernels.md`.
- `fixture.json`: schema version 1; `model` has `id`, `revision`, and `dimensions`.
  `documents` contain string `id`, `title`, `text`, `source`, and `embedding`.
  `queries` contain `id`, `text`, `relevant_document_ids`, and `embedding`.
  Every vector contains 384 finite real numbers and has unit L2 norm.
- `source_provenance.json`: exact repository paths, inclusive 1-based line ranges,
  SHA-256 of complete source files, and SHA-256 of each excerpt's UTF-8 text.
  The fixture's `source` string is `path#Lstart-Lend`. Text retains source line
  breaks but omits a trailing newline. Titles record the nearest section heading.
- `model_manifest.json`: pinned model identity, file hashes, model-card license
  (`apache-2.0`), 128-token limit, and attention-mask mean-pooling configuration.
- `provenance.json`: fixture/input/exporter/manifest hashes, Python and package
  versions, CPU encoding parameters, token lengths and truncated excerpt IDs.
  Neither weights nor an environment is included in this repository.

The model is
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42),
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. The exporter checks all
inference-file hashes before loading it. It embeds **text only** with the real
transformer, attention-mask mean pooling, then L2 normalization. It does not
embed titles or add query instructions. The pinned model truncates inputs at
128 tokens; the provenance reports affected excerpts explicitly. Longer source
paragraphs remain intact for display. These are actual model outputs, not
random features, TF-IDF, or synthesized similarity scores.

Relevance labels were hand-authored by reading the excerpts before model
scoring. They are illustrative and non-exhaustive. They were not selected from
nearest-neighbor results. With the checked-in vectors, seven of eight queries
have a labeled document in their top three. The Korean `key-lifetime` query
misses its initial labels in the top three; its first result also discusses key
expiry, illustrating the limits of these sparse judgments. Numeric agreement
between plaintext and encrypted scores is a separate claim from retrieval
relevance. Fresh queries have an empty relevance list because they have not
been adjudicated.

## Reproduce locally

From the repository root, create an isolated environment outside the checkout
(Python 3.12 was used for the checked-in fixture):

```sh
python3.12 -m venv /tmp/fhenomenon-search-python
/tmp/fhenomenon-search-python/bin/python -m pip install \
  -r examples/private_search/requirements.txt
```

The requirements pin the complete environment used for generation. This example
is CPU-only. Platform or numerical-library changes can produce tiny floating
point differences even with identical weights. Model file identity and semantic
validation are portable; byte identity is specific to the recorded environment.

For a previously downloaded complete model snapshot, run fully offline:

```sh
HF_HOME=/tmp/fhenomenon-search-hf TOKENIZERS_PARALLELISM=false \
/tmp/fhenomenon-search-python/bin/python \
  examples/private_search/prepare_embeddings.py \
  --local-model-path /path/to/the/pinned/model/snapshot \
  --local-files-only \
  --provenance examples/private_search/data/provenance.json
```

Alternatively omit `--local-model-path` to load the pinned revision from the
Hugging Face cache; also omit `--local-files-only` if downloading public model
files is needed. Only model files are downloaded. Corpus and query text are
processed locally and never sent to an embedding service. `--revision` accepts
only the recorded revision; changing models requires deliberately updating the
manifest and exporter together.

Validation and exact documentation-source verification use the Python standard
library and do not import inference packages:

```sh
python3 examples/private_search/prepare_embeddings.py \
  --validate-only examples/private_search/data/fixture.json \
  --verify-source-root .
```

## Use your own corpus or fresh queries

Supply `--corpus /path/to/corpus.json` in the same schema as `corpus.json`:
nonempty `documents` and `queries`, unique string IDs, nonempty text, and document
`title`/`source` strings. Each query's `relevant_document_ids` is an array of
known document IDs, or empty when unjudged. `--query` replaces the corpus's
sample queries with newly embedded text and may be repeated. Replacement corpora
and fresh queries require an explicit `--output` destination. Always write
personal exports to a location outside the repository:

```sh
HF_HOME=/tmp/fhenomenon-search-hf TOKENIZERS_PARALLELISM=false \
/tmp/fhenomenon-search-python/bin/python \
  examples/private_search/prepare_embeddings.py \
  --corpus /path/to/my-corpus.json \
  --query '내 문서에서 평가 키의 수명에 대한 설명을 찾아줘' \
  --local-model-path /path/to/the/pinned/model/snapshot \
  --local-files-only --output /tmp/my-search-fixture.json
```

The exporter writes a companion `my-search-fixture.provenance.json`, validates
IDs, relevance references, dimensions, finite numbers, and unit norms, and
prints plaintext top-three scores for inspection. Source verification applies
only to the bundled public corpus. Replacement-corpus provenance belongs to its
owner; the exporter records the input corpus hash but does not certify arbitrary
`source` strings.
