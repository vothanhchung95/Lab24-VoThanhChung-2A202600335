"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import RERANK_TOP_K
except ImportError:
    RERANK_TOP_K = 3  # Fallback nếu không tìm thấy file config


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    # Thay đổi model mặc định sang bản MiniLM để pass tiêu chí Latency < 5s trên CPU
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Option B: Sử dụng sentence_transformers
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError("Vui lòng cài đặt: pip install sentence-transformers")
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        # 1. model = self._load_model()
        model = self._load_model()

        # 2. pairs = [(query, doc["text"]) for doc in documents]
        pairs = [(query, doc["text"]) for doc in documents]

        # 3. scores = model.predict(pairs)      # CrossEncoder
        scores = model.predict(pairs)

        # 4. Combine: [(score, doc) for score, doc in zip(scores, documents)]
        combined = [(float(score), doc) for score, doc in zip(scores, documents)]

        # 5. Sort by score descending
        combined.sort(key=lambda x: x[0], reverse=True)

        # 6. Return top_k RerankResult
        results = []
        for i, (score, doc) in enumerate(combined[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
                rank=i + 1
            ))
            
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
            
        # TODO (optional): from flashrank import Ranker, RerankRequest
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError:
            raise ImportError("Vui lòng cài đặt: pip install flashrank")

        # Khởi tạo model nếu chưa có
        if self._model is None:
            self._model = Ranker() 
            
        # Truyền thêm id vào passages để map lại metadata và original_score
        passages = [
            {"id": i, "text": d["text"], "meta": d.get("metadata", {})} 
            for i, d in enumerate(documents)
        ]
        
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        reranked_data = self._model.rerank(RerankRequest(query=query, passages=passages))
        
        results = []
        for i, res in enumerate(reranked_data[:top_k]):
            original_doc = documents[res["id"]]
            results.append(RerankResult(
                text=res["text"],
                original_score=original_doc.get("score", 0.0),
                rerank_score=float(res.get("score", 0.0)),
                metadata=res.get("meta", {}),
                rank=i + 1
            ))
            
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    # Warm-up run (không tính vào thời gian đo)
    _ = reranker.rerank(query, documents)

    # 1. times = []
    times = []
    
    # 2. for _ in range(n_runs):
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # ms
        
    # 3. return {"avg_ms": mean(times), "min_ms": min(times), "max_ms": max(times)}
    if not times:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        
    return {
        "avg_ms": sum(times) / len(times), 
        "min_ms": min(times), 
        "max_ms": max(times)
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    
    print("--- Test CrossEncoder ---")
    reranker_ce = CrossEncoderReranker()
    for r in reranker_ce.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
        
    print("\n--- Test Benchmark (CrossEncoder) ---")
    stats = benchmark_reranker(reranker_ce, query, docs, n_runs=3)
    print(f"Avg Latency: {stats['avg_ms']:.2f} ms")