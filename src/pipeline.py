"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4+M5."""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY


_OPENAI_CLIENT = None

# Aggregated stage timings — populated by build_pipeline + run_query, dumped by evaluate_pipeline.
LATENCY_STATS: dict = {
    "setup": {},          # one-time costs (chunking, enrichment, indexing, reranker_load)
    "per_query_ms": {     # collected per query (search, rerank, llm, total)
        "search": [], "rerank": [], "llm": [], "total": [],
    },
}


def _get_llm():
    """Cached OpenAI client for answer generation."""
    global _OPENAI_CLIENT
    if not OPENAI_API_KEY:
        return None
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _OPENAI_CLIENT


def _generate_answer(query: str, contexts: list[str]) -> str:
    """Use LLM with strict context-only prompt → maximize Faithfulness."""
    client = _get_llm()
    if client is None or not contexts:
        return contexts[0] if contexts else "Không tìm thấy thông tin trong tài liệu."
    context_str = "\n\n".join(contexts)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Bạn là trợ lý chỉ trả lời dựa CHÍNH XÁC trên context được cung cấp. "
                        "Tuyệt đối không suy đoán, không dùng kiến thức ngoài context. "
                        "Nếu context không có thông tin → trả lời: 'Không tìm thấy thông tin trong tài liệu.' "
                        "Trả lời ngắn gọn, đúng trọng tâm câu hỏi, bằng tiếng Việt."},
            {"role": "user",
             "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}\n\nTrả lời:"},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def build_pipeline():
    """Build production RAG pipeline. Records setup latencies into LATENCY_STATS['setup']."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/4] Chunking documents...")
    t0 = time.perf_counter()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    LATENCY_STATS["setup"]["chunking_s"] = round(time.perf_counter() - t0, 3)
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents ({LATENCY_STATS['setup']['chunking_s']}s)")

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    t0 = time.perf_counter()
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    LATENCY_STATS["setup"]["enrichment_s"] = round(time.perf_counter() - t0, 3)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks ({LATENCY_STATS['setup']['enrichment_s']}s)")
    else:
        print("  ⚠️  M5 not implemented — using raw chunks (fallback)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    t0 = time.perf_counter()
    search = HybridSearch()
    search.index(all_chunks)
    LATENCY_STATS["setup"]["indexing_s"] = round(time.perf_counter() - t0, 3)
    print(f"  Indexed in {LATENCY_STATS['setup']['indexing_s']}s")

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker()
    LATENCY_STATS["setup"]["reranker_load_s"] = round(time.perf_counter() - t0, 3)
    print(f"  Reranker loaded in {LATENCY_STATS['setup']['reranker_load_s']}s")

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline. Records per-query latencies into LATENCY_STATS."""
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    results = search.search(query)
    LATENCY_STATS["per_query_ms"]["search"].append((time.perf_counter() - t0) * 1000)

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    LATENCY_STATS["per_query_ms"]["rerank"].append((time.perf_counter() - t0) * 1000)

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    t0 = time.perf_counter()
    answer = _generate_answer(query, contexts)
    LATENCY_STATS["per_query_ms"]["llm"].append((time.perf_counter() - t0) * 1000)

    LATENCY_STATS["per_query_ms"]["total"].append((time.perf_counter() - t_total) * 1000)
    return answer, contexts


def _save_latency_report(path: str = "reports/latency_breakdown.json") -> None:
    """Aggregate per-query timings + write JSON + print table. Bonus +2đ."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def agg(samples: list[float]) -> dict:
        if not samples:
            return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "max_ms": 0, "n": 0}
        s = sorted(samples)
        return {
            "avg_ms": round(statistics.mean(s), 1),
            "p50_ms": round(s[len(s) // 2], 1),
            "p95_ms": round(s[max(0, int(len(s) * 0.95) - 1)], 1),
            "max_ms": round(s[-1], 1),
            "n": len(s),
        }

    report = {
        "setup": LATENCY_STATS["setup"],
        "per_query": {stage: agg(samples) for stage, samples in LATENCY_STATS["per_query_ms"].items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print("Setup (one-time):")
    for k, v in report["setup"].items():
        print(f"  {k:<20} {v}s")
    print(f"\nPer-query ({report['per_query']['total']['n']} queries):")
    print(f"  {'stage':<10} | {'avg':>7} | {'p50':>7} | {'p95':>7} | {'max':>7}")
    print("  " + "-" * 50)
    for stage in ["search", "rerank", "llm", "total"]:
        s = report["per_query"][stage]
        print(f"  {stage:<10} | {s['avg_ms']:>5.1f}ms | {s['p50_ms']:>5.1f}ms | {s['p95_ms']:>5.1f}ms | {s['max_ms']:>5.1f}ms")
    print(f"\nReport saved to {path}")


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    _save_latency_report()
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
