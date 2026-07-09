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
"""
rag_backend_m2_additions.py
------------------------------
M2 (intent-extended RRF-fusion retrieval) additions for rag_backend.py.

APPEND THIS TO THE END OF YOUR EXISTING rag_backend.py — it adds a
separate, parallel set of functions (init_m2/retrieve_m2) alongside your
existing M1 functions (init/retrieve), without modifying them. M0/M1
continue to work exactly as before.

Adapted from your M3_eval.py (which implements what your thesis calls M2
— the script's internal naming says M3, this file uses M2 to match your
UI/examples_data.py convention). Preserves the doc_id alignment fixes:
  - node_id -> doc_id via ref_doc_id (not fragile index arithmetic)
  - meta stores 'intention'/'function_snippet' per doc_id directly
  - intent_idx_to_doc_id reverse mapping for correct dense-search alignment

Requires: numpy, sentence_transformers (SentenceTransformer + CrossEncoder
— CrossEncoder is new here, M1 only used the bi-encoder SentenceTransformer
class, so no new pip package, just a new import from the same library).
"""

import numpy as np
from sentence_transformers import CrossEncoder

M2_RERANK_MODEL = "BAAI/bge-reranker-base"
M2_INTENT_MODEL = "BAAI/bge-small-en-v1.5"
M2_BM25_TOP_K   = 30
M2_RRF_K        = 60

_m2_code_docs            = None
_m2_bm25_code            = None
_m2_node_id_to_doc_id    = None
_m2_doc_id_to_meta       = None
_m2_intent_encoder       = None
_m2_reranker             = None
_m2_intent_embeddings    = None
_m2_intent_idx_to_doc_id = None
_m2_ready = False


def is_m2_ready() -> bool:
    return _m2_ready


def init_m2(d_train_path: str = None):
    """
    Builds the M2 index. If d_train_path is omitted, reuses the same
    D_train data already loaded by init() for M1 (as long as that data's
    items include an 'intention' field — if your M1 KB source doesn't
    have intent fields, pass the path to a version that does, e.g.
    D_train_with_intent.json).
    """
    global _m2_code_docs, _m2_bm25_code, _m2_node_id_to_doc_id
    global _m2_doc_id_to_meta, _m2_intent_encoder, _m2_reranker
    global _m2_intent_embeddings, _m2_intent_idx_to_doc_id, _m2_ready

    import json
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.retrievers.bm25 import BM25Retriever

    path = d_train_path
    if path is None:
        raise ValueError(
            "init_m2() requires d_train_path on first call — pass the "
            "path to your intent-annotated D_train (e.g. "
            "D_train_with_intent.json, or plain D_train.json if it "
            "already includes 'intention' fields per item)."
        )

    print(f"[rag_backend] [M2] Loading KB from {path} ...")
    kb_data = json.load(open(path, encoding="utf-8"))
    print(f"[rag_backend] [M2] KB: {len(kb_data)} items")

    code_docs, intent_texts = [], []
    doc_id_to_meta = {}
    intent_idx_to_doc_id = {}

    for kb_idx, item in enumerate(kb_data):
        rep       = item.get("replacement api", "").strip()
        deps      = item.get("deprecated api", [])
        if isinstance(deps, dict):
            deps = [v for v in deps.values() if isinstance(v, str) and "." in v]
        fn        = item.get("function", "").strip()
        intention = item.get("intention", "").strip()
        ref       = item.get("reference", "").strip()

        if not rep or not deps or not fn:
            continue

        doc_id = f"m2doc_{kb_idx}"
        meta = {
            "doc_id": doc_id,
            "deprecated_apis": deps,
            "replacement": rep,
            "reference": ref[:300],
            "intention": intention[:300],
            "function_snippet": fn[:300],
        }

        code_docs.append(Document(
            text=f"Replacement API: {rep}\nDeprecated: {deps}\n{fn}",
            metadata=meta,
            id_=doc_id,
        ))

        intent_pos = len(intent_texts)
        intent_text = intention if intention else "\n".join(fn.splitlines()[:2])
        intent_idx_to_doc_id[intent_pos] = doc_id
        intent_texts.append(intent_text)
        doc_id_to_meta[doc_id] = meta

    print(f"[rag_backend] [M2] {len(code_docs)} KB docs")

    code_nodes = SentenceSplitter(chunk_size=4096).get_nodes_from_documents(code_docs)
    bm25_code = BM25Retriever.from_defaults(nodes=code_nodes, similarity_top_k=M2_BM25_TOP_K)

    node_id_to_doc_id = {}
    for node in code_nodes:
        parent_doc_id = node.ref_doc_id
        if parent_doc_id:
            node_id_to_doc_id[node.node_id] = parent_doc_id

    print("[rag_backend] [M2] Loading intent encoder + cross-encoder reranker ...")
    intent_encoder = SentenceTransformer(M2_INTENT_MODEL, device=DEVICE)
    reranker = CrossEncoder(M2_RERANK_MODEL, device=DEVICE)

    print("[rag_backend] [M2] Encoding KB intentions (dense) ...")
    intent_embeddings = intent_encoder.encode(
        intent_texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False,
    )
    intent_embeddings = np.array(intent_embeddings)
    print(f"[rag_backend] [M2] Dense intent index: {intent_embeddings.shape[0]} vectors")

    _m2_code_docs = code_docs
    _m2_bm25_code = bm25_code
    _m2_node_id_to_doc_id = node_id_to_doc_id
    _m2_doc_id_to_meta = doc_id_to_meta
    _m2_intent_encoder = intent_encoder
    _m2_reranker = reranker
    _m2_intent_embeddings = intent_embeddings
    _m2_intent_idx_to_doc_id = intent_idx_to_doc_id
    _m2_ready = True
    print("[rag_backend] [M2] Ready.")


def retrieve_m2(prefix: str, intention: str = "") -> str:
    """
    Returns the EVOLUTION INFO string for M2 (intent-extended RRF fusion),
    matching build_context_from_hits' format so it's a drop-in replacement
    for M1's retrieve() in build_prompt().

    `intention` should be:
      - the KNOWN intent from a D_test case, if available (preferred —
        matches exactly what your thesis evaluation used), OR
      - "" to fall back to code-only retrieval (dense intent track skipped,
        pure BM25 code-track ranking used) — the CALLER is responsible for
        invoking intent_backend.generate_intent() first if a live-generated
        intent is wanted; this function does not generate intents itself,
        keeping GPU/CPU concerns separate from retrieval logic.
    """
    if not _m2_ready:
        return ""

    left_ctx_lines = 20
    lines = prefix.strip().splitlines()
    code_query = "\n".join(lines[-left_ctx_lines:])

    code_hits = _m2_bm25_code.retrieve(code_query)
    code_rank = {}
    for i, h in enumerate(code_hits):
        doc_id = _m2_node_id_to_doc_id.get(h.node.node_id)
        if doc_id:
            code_rank[doc_id] = i + 1

    intent_rank = {}
    if intention.strip():
        q_emb = _m2_intent_encoder.encode(intention, normalize_embeddings=True)
        sims = _m2_intent_embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:M2_BM25_TOP_K]
        for rank, idx in enumerate(top_idx):
            doc_id = _m2_intent_idx_to_doc_id.get(int(idx))
            if doc_id:
                intent_rank[doc_id] = rank + 1

    all_ids = set(code_rank) | set(intent_rank)
    rrf_scores = {
        nid: (1.0 / (M2_RRF_K + code_rank.get(nid, 1000)) +
              1.0 / (M2_RRF_K + intent_rank.get(nid, 1000)))
        for nid in all_ids
    }
    top_doc_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:M2_BM25_TOP_K]

    code_hit_map = {
        _m2_node_id_to_doc_id.get(h.node.node_id): h
        for h in code_hits
        if _m2_node_id_to_doc_id.get(h.node.node_id)
    }

    fused = []  # list of (doc_id, content_text)
    for doc_id in top_doc_ids:
        if doc_id in code_hit_map:
            fused.append((doc_id, code_hit_map[doc_id].node.get_content()))
        elif doc_id in _m2_doc_id_to_meta:
            m = _m2_doc_id_to_meta[doc_id]
            text = (
                f"Intent: {m.get('intention', '')}\n"
                f"Replacement API: {m['replacement']}\n"
                f"Deprecated: {m['deprecated_apis']}\n"
                f"{m.get('function_snippet', '')}"
            )
            fused.append((doc_id, text))

    if not fused:
        return ""

    pairs = [[code_query, content] for _, content in fused]
    scores = _m2_reranker.predict(pairs)
    ranked = sorted(zip(scores, fused), key=lambda x: x[0], reverse=True)

    top_k = 3
    seen, parts = set(), []
    for _, (doc_id, _content) in ranked[:top_k]:
        m = _m2_doc_id_to_meta.get(doc_id, {})
        rep = m.get("replacement", "").strip()
        deps = m.get("deprecated_apis", [])
        ref = m.get("reference", "").strip()
        key = (rep, tuple(sorted(deps)))
        if key in seen or not rep:
            continue
        seen.add(key)
        text = f"Replacement API: {rep}\nDeprecated: {deps}"
        if ref:
            text += f"\nExample: {ref}"
        parts.append(text)

    return "\n\n---\n\n".join(parts)