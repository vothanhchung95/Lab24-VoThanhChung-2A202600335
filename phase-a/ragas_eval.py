"""Phase A: Run RAGAS evaluation on 50-question synthetic test set."""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import build_pipeline, run_query
from src.m4_eval import evaluate_ragas

SYNTHETIC_CSV = os.path.join(os.path.dirname(__file__), "synthetic_test_set.csv")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "ragas_results.csv")

THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.65,
    "context_precision": 0.75,
    "context_recall": 0.75,
}


def load_synthetic_set():
    with open(SYNTHETIC_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_eval():
    rows = load_synthetic_set()
    search, reranker = build_pipeline()

    questions, answers, all_contexts, ground_truths, ids, types, domains = [], [], [], [], [], [], []
    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] {row['question'][:60]}...")
        answer, contexts = run_query(row["question"], search, reranker)
        questions.append(row["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(row["ground_truth"])
        ids.append(row["id"])
        types.append(row["type"])
        domains.append(row["domain"])

    print("\nRunning RAGAS evaluation...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    per_q = results.get("per_question", [])
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "question", "faithfulness", "answer_relevancy",
            "context_precision", "context_recall", "avg_score", "type", "domain",
        ])
        writer.writeheader()
        for idx, r in enumerate(per_q):
            avg = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4
            writer.writerow({
                "id": ids[idx],
                "question": r.question,
                "faithfulness": round(r.faithfulness, 4),
                "answer_relevancy": round(r.answer_relevancy, 4),
                "context_precision": round(r.context_precision, 4),
                "context_recall": round(r.context_recall, 4),
                "avg_score": round(avg, 4),
                "type": types[idx],
                "domain": domains[idx],
            })

    agg = {k: round(v, 4) for k, v in results.items() if k != "per_question"}
    n = len(per_q)
    if n:
        agg = {
            "faithfulness": round(sum(r.faithfulness for r in per_q) / n, 4),
            "answer_relevancy": round(sum(r.answer_relevancy for r in per_q) / n, 4),
            "context_precision": round(sum(r.context_precision for r in per_q) / n, 4),
            "context_recall": round(sum(r.context_recall for r in per_q) / n, 4),
        }

    print("\n=== RAGAS Aggregate (50 questions) ===")
    all_pass = True
    for m, threshold in THRESHOLDS.items():
        v = agg.get(m, 0)
        passed = v >= threshold
        if not passed:
            all_pass = False
        print(f"  {'✓' if passed else '✗'} {m}: {v} (min {threshold})")

    print(f"\nResults saved to {RESULTS_CSV}")

    bottom10 = sorted(per_q, key=lambda r: (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4)[:10]
    print("\n=== Bottom-10 Failures ===")
    for r in bottom10:
        avg = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4
        worst = min({"faithfulness": r.faithfulness, "answer_relevancy": r.answer_relevancy,
                     "context_precision": r.context_precision, "context_recall": r.context_recall},
                    key=lambda k: {"faithfulness": r.faithfulness, "answer_relevancy": r.answer_relevancy,
                                   "context_precision": r.context_precision, "context_recall": r.context_recall}[k])
        print(f"  [{avg:.3f}] worst={worst} | {r.question[:60]}")

    return agg, all_pass


if __name__ == "__main__":
    agg, passed = run_eval()
    if not passed:
        print("\n⚠ Some metrics below threshold — see failure_analysis.md")
