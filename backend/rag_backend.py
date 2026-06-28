"""
RAG retrieval backend for the demo — M1 pipeline (BM25 + rerank).
Mirrors M1_dtrain_dtest.py's retrieve() and build_context_from_hits() exactly.
Call init(d_train_path) once at startup, then retrieve(prefix) per request.
"""

import json, torch
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BM25_TOP_K   = 10
FINAL_TOP_K  = 3
LEFT_CTX_LINES = 30
RERANK_MODEL = "flax-sentence-embeddings/st-codesearch-distilroberta-base"

_bm25     = None
_reranker = None
_ready    = False


def get_dep_apis(item):
    deps = item.get("deprecated api", [])
    if isinstance(deps, list):
        return deps
    if isinstance(deps, dict):
        out, i = [], 0
        while str(i) in deps:
            v = deps[str(i)]
            if isinstance(v, str) and "." in v:
                out.append(v)
            i += 1
        return out
    return []


def build_context_from_hits(hits, k=FINAL_TOP_K):
    seen, parts = set(), []
    for h in hits[:k]:
        rep  = h.node.metadata.get("replacement", "").strip()
        deps = h.node.metadata.get("deprecated_apis", [])
        ref  = h.node.metadata.get("reference", "").strip()
        key  = (rep, tuple(sorted(deps)))
        if key in seen or not rep:
            continue
        seen.add(key)
        text = f"Replacement API: {rep}\nDeprecated: {deps}"
        if ref:
            text += f"\nExample: {ref}"
        parts.append(text)
    return "\n\n---\n\n".join(parts)


def init(d_train_path: str):
    global _bm25, _reranker, _ready
    print(f"[rag_backend] Loading D_train from {d_train_path} ...")
    d_train = json.load(open(d_train_path, encoding="utf-8"))

    kb_docs = []
    for item in d_train:
        rep  = item.get("replacement api", "").strip()
        deps = get_dep_apis(item)
        fn   = item.get("function", "").strip()
        ref  = item.get("reference", "").strip()
        if not rep or not deps or not fn:
            continue
        kb_docs.append(Document(
            text=(f"Replacement API: {rep}\nDeprecated: {deps}\n{fn}"),
            metadata={"deprecated_apis": deps, "replacement": rep,
                      "reference": ref[:300]}
        ))

    print(f"[rag_backend] Building BM25 index from {len(kb_docs)} docs ...")
    nodes = SentenceSplitter(chunk_size=2048).get_nodes_from_documents(kb_docs)
    _bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=BM25_TOP_K)

    print(f"[rag_backend] Loading reranker {RERANK_MODEL} ...")
    _reranker = SentenceTransformer(RERANK_MODEL, device=DEVICE)

    _ready = True
    print("[rag_backend] RAG pipeline ready.")


def is_ready() -> bool:
    return _ready


def retrieve(prefix: str) -> str:
    """Returns the EVOLUTION INFO string to splice into build_prompt()."""
    if not _ready:
        return ""
    lines = prefix.strip().splitlines()
    query = "\n".join(lines[-LEFT_CTX_LINES:])
    hits  = _bm25.retrieve(query)
    if not hits:
        return ""
    docs   = [h.node.get_content() for h in hits]
    q_emb  = _reranker.encode(query, normalize_embeddings=True)
    d_embs = _reranker.encode(docs,  normalize_embeddings=True)
    scores = d_embs @ q_emb
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    top    = [h for _, h in ranked[:FINAL_TOP_K]]
    return build_context_from_hits(top)