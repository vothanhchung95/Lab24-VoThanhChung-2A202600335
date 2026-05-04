"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words (space-separated) for BM25."""
    if not (text or "").strip():
        return ""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return text.strip()


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        if not self.documents:
            self.corpus_tokens = []
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self.corpus_tokens = []
        for chunk in self.documents:
            seg = segment_vietnamese(chunk.get("text", "") or "")
            self.corpus_tokens.append(seg.split() if seg else [])
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        q_tokens = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(q_tokens)
        n = len(scores)
        top_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=dict(self.documents[i].get("metadata") or {}),
                method="bm25",
            )
            for i in top_indices
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return
        texts = [c.get("text", "") or "" for c in chunks]
        model = self._get_encoder()
        vectors = model.encode(
            texts,
            show_progress_bar=len(texts) > 8,
            normalize_embeddings=True,
        )
        points = []
        for i, c in enumerate(chunks):
            row = vectors[i]
            vec = row.tolist() if hasattr(row, "tolist") else list(row)
            payload = {**(c.get("metadata") or {}), "text": c.get("text", "") or ""}
            points.append(PointStruct(id=i, vector=vec, payload=payload))
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        raw = self._get_encoder().encode(query, normalize_embeddings=True)
        if getattr(raw, "ndim", 0) > 1:
            query_vector = raw[0].tolist()
        else:
            query_vector = raw.tolist()
        res = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        out: list[SearchResult] = []
        for hit in res.points or []:
            pl = dict(hit.payload or {})
            text = pl.pop("text", "")
            out.append(
                SearchResult(
                    text=text,
                    score=float(hit.score),
                    metadata=pl,
                    method="dense",
                )
            )
        return out


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, float] = {}
    meta_by_text: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in meta_by_text:
                meta_by_text[key] = dict(result.metadata or {})
    ordered = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)[:top_k]
    return [
        SearchResult(
            text=t,
            score=rrf_scores[t],
            metadata=dict(meta_by_text.get(t, {})),
            method="hybrid",
        )
        for t in ordered
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
