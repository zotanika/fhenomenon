#!/usr/bin/env python3
"""Export local sentence embeddings; inference never calls a remote service."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

DATA = Path(__file__).resolve().parent / "data"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DIMENSIONS = 384


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def validate(data, embeddings=True):
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if embeddings:
        model = data.get("model", {})
        if model != {"id": MODEL_ID, "revision": MODEL_REVISION, "dimensions": DIMENSIONS}:
            raise ValueError("unexpected model identity, pinned revision, or dimensions")
    ids = set()
    document_ids = set()
    for group in ("documents", "queries"):
        entries = data.get(group)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{group} must be a nonempty array")
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError(f"{group} entries must be objects")
            nonempty(item.get("id"), f"{group}.id")
            if item["id"] in ids:
                raise ValueError(f"duplicate ID: {item['id']}")
            ids.add(item["id"])
            nonempty(item.get("text"), f"{item['id']}.text")
            if group == "documents":
                document_ids.add(item["id"])
                nonempty(item.get("title"), "title")
                nonempty(item.get("source"), "source")
            else:
                relevant = item.get("relevant_document_ids")
                if not isinstance(relevant, list) or any(not isinstance(value, str) for value in relevant):
                    raise ValueError("relevant_document_ids must be a string array")
                if len(set(relevant)) != len(relevant) or not set(relevant) <= document_ids:
                    raise ValueError(f"unknown or duplicate relevance ID: {item['id']}")
            if embeddings:
                vector = item.get("embedding")
                if not isinstance(vector, list) or len(vector) != DIMENSIONS:
                    raise ValueError(f"{item['id']}: expected {DIMENSIONS} dimensions")
                if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
                    raise ValueError(f"{item['id']}: nonfinite or nonnumeric component")
                if abs(math.sqrt(sum(x*x for x in vector)) - 1.0) > 1e-5:
                    raise ValueError(f"{item['id']}: embedding must have unit norm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DATA / "corpus.json", help="Schema-1 documents and queries without embeddings")
    parser.add_argument("--output", type=Path, help="Export destination; required for a replacement corpus or fresh query")
    parser.add_argument("--provenance", type=Path, help="Export metadata path; defaults to OUTPUT stem + .provenance.json")
    parser.add_argument("--revision", default=MODEL_REVISION, choices=[MODEL_REVISION], help="Pinned model revision")
    parser.add_argument("--local-model-path", type=Path, help="Existing complete snapshot, verified against bundled model hashes")
    parser.add_argument("--local-files-only", action="store_true", help="Disallow model download; use cached files only")
    parser.add_argument("--query", action="append", help="Replace sample queries with fresh text; repeat for multiple queries")
    parser.add_argument("--verify-source-root", type=Path, help="Verify bundled corpus source paths, lines and SHA-256 against a repository checkout")
    parser.add_argument("--validate-only", type=Path, metavar="FIXTURE", help="Validate an existing export without loading model packages")
    args = parser.parse_args()
    if args.output is None:
        custom_corpus = args.corpus.resolve() != (DATA / "corpus.json").resolve()
        if not args.validate_only and (custom_corpus or args.query):
            parser.error("--corpus overrides and --query require an explicit --output destination")
        args.output = DATA / "fixture.json"
    if args.verify_source_root:
        provenance = read_json(DATA / "source_provenance.json")
        bundled = read_json(DATA / "corpus.json")
        by_id = {entry["id"]: entry for entry in bundled["documents"]}
        for name, expected in provenance["source_files_sha256"].items():
            if digest(args.verify_source_root / name) != expected:
                raise ValueError(f"source file hash mismatch: {name}")
        for chunk in provenance["chunks"]:
            lines = (args.verify_source_root / chunk["path"]).read_text(encoding="utf-8").splitlines()
            text = "\n".join(lines[chunk["line_start"]-1:chunk["line_end"]])
            source = f"{chunk['path']}#L{chunk['line_start']}-L{chunk['line_end']}"
            if text != by_id[chunk["id"]]["text"] or source != by_id[chunk["id"]]["source"] or hashlib.sha256(text.encode()).hexdigest() != chunk["text_sha256"]:
                raise ValueError(f"source chunk mismatch: {chunk['id']}")
        print("Verified 65 bundled source chunks and exact source file hashes")
    if args.validate_only:
        validate(read_json(args.validate_only))
        print(f"Validated {args.validate_only}")
        return
    corpus = read_json(args.corpus)
    if args.query:
        corpus["queries"] = [{"id": f"fresh-query-{i+1}", "text": text, "relevant_document_ids": []} for i, text in enumerate(args.query)]
    validate(corpus, embeddings=False)

    # Import model packages only for export. Merely validating JSON needs stdlib.
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    manifest = read_json(DATA / "model_manifest.json")
    model_path = args.local_model_path
    if model_path is None:
        model_path = Path(snapshot_download(repo_id=MODEL_ID, revision=args.revision,
                         local_files_only=args.local_files_only,
                         allow_patterns=list(manifest["files_sha256"])))
    model_path = model_path.resolve()
    for name, expected in manifest["files_sha256"].items():
        if digest(model_path / name) != expected:
            raise ValueError(f"model snapshot hash mismatch: {name}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    if model.get_sentence_embedding_dimension() != DIMENSIONS or model.max_seq_length != 128:
        raise ValueError("unexpected model dimension or token limit")
    records = corpus["documents"] + corpus["queries"]
    texts = [record["text"] for record in records]
    vectors = model.encode(texts, batch_size=8, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
    data = {"schema_version": 1, "model": {"id": MODEL_ID, "revision": args.revision, "dimensions": DIMENSIONS}}
    offset = 0
    for group in ("documents", "queries"):
        data[group] = []
        for entry in corpus[group]:
            item = {key: value for key, value in entry.items() if key != "embedding"}
            item["embedding"] = [float(x) for x in vectors[offset]]
            data[group].append(item)
            offset += 1
    validate(data)
    if args.output.resolve() == args.corpus.resolve():
        raise ValueError("output must differ from input corpus")
    provenance_path = args.provenance or args.output.with_suffix(".provenance.json")
    if provenance_path.resolve() in (args.output.resolve(), args.corpus.resolve()):
        raise ValueError("provenance must differ from corpus and output")
    write_json(args.output, data)
    lengths = [len(tokens) for tokens in model.tokenizer(texts, truncation=False, verbose=False)["input_ids"]]
    provenance = {
        "fixture_sha256": digest(args.output), "corpus_sha256": digest(args.corpus),
        "exporter_sha256": digest(Path(__file__)), "model_manifest_sha256": digest(DATA / "model_manifest.json"),
        "model": data["model"], "python": platform.python_version(), "platform": sys.platform + "-" + platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in ("sentence-transformers", "torch", "transformers", "numpy", "tokenizers", "huggingface-hub")},
        "encoding": {"input": "text only; no title or prompt prefix", "device": "cpu", "threads": 1, "batch_size": 8, "max_sequence_length": 128, "pooling": "attention-mask mean", "normalization": "L2", "dtype": "float32"},
        "token_lengths_before_truncation": {entry["id"]: length for entry, length in zip(records, lengths)},
        "truncated_ids": [entry["id"] for entry, length in zip(records, lengths) if length > 128],
        "fresh_query_override": args.query or [],
        "reproducibility": "Pinned model file hashes and CPU encoding. Package/hardware changes may cause small floating-point differences; semantic and norm validation is portable, byte identity is environment-specific."
    }
    write_json(provenance_path, provenance)
    print(f"Wrote {len(data['documents'])} documents and {len(data['queries'])} queries to {args.output}")
    for query in data["queries"]:
        ranked = sorted(((sum(a*b for a,b in zip(query["embedding"], doc["embedding"])), doc["id"]) for doc in data["documents"]), reverse=True)
        print(query["id"] + " top-3: " + ", ".join(f"{id}={score:.6f}" for score,id in ranked[:3]))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
