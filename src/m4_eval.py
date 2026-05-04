"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _heuristic_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Word-overlap heuristics — fallback khi RAGAS/OpenAI không khả dụng."""
    per_question: list[EvalResult] = []

    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        ctx_text = " ".join(ctx).lower()
        ctx_words = set(ctx_text.split())
        gt_words = set(gt.lower().split())
        a_words = set(a.lower().split()) if a else set()
        q_words = set(q.lower().split())

        # Context recall: ground_truth keywords covered by contexts
        context_recall = len(gt_words & ctx_words) / max(len(gt_words), 1)
        # Context precision: query keywords matched in contexts
        context_precision = len(q_words & ctx_words) / max(len(q_words), 1)
        # Faithfulness: answer words grounded in contexts
        faithfulness = len(a_words & ctx_words) / max(len(a_words), 1) if a_words else 0.0
        # Answer relevancy: query keywords present in answer
        answer_relevancy = len(q_words & a_words) / max(len(q_words), 1)

        per_question.append(EvalResult(
            question=q,
            answer=a,
            contexts=ctx,
            ground_truth=gt,
            faithfulness=min(faithfulness, 1.0),
            answer_relevancy=min(answer_relevancy, 1.0),
            context_precision=min(context_precision, 1.0),
            context_recall=min(context_recall, 1.0),
        ))

    if not per_question:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    n = len(per_question)
    return {
        "faithfulness": sum(r.faithfulness for r in per_question) / n,
        "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
        "context_precision": sum(r.context_precision for r in per_question) / n,
        "context_recall": sum(r.context_recall for r in per_question) / n,
        "per_question": per_question,
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation với 4 metrics. Fallback sang heuristic nếu RAGAS/LLM không khả dụng.

    Returns:
        dict với keys: faithfulness, answer_relevancy, context_precision,
                       context_recall (float), per_question (list[EvalResult])
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        per_question: list[EvalResult] = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=float(row.get("faithfulness") or 0.0),
                answer_relevancy=float(row.get("answer_relevancy") or 0.0),
                context_precision=float(row.get("context_precision") or 0.0),
                context_recall=float(row.get("context_recall") or 0.0),
            )
            for _, row in df.iterrows()
        ]

        return {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "context_recall": float(df["context_recall"].mean()),
            "per_question": per_question,
        }

    except Exception as e:
        print(f"[M4] RAGAS unavailable ({type(e).__name__}: {e}), dùng heuristic fallback")
        return _heuristic_evaluate(questions, answers, contexts, ground_truths)


# Diagnostic Tree: ngưỡng, chẩn đoán, và đề xuất fix cho mỗi metric
_DIAGNOSTIC_TREE: dict[str, tuple[float, str, str]] = {
    "faithfulness": (
        0.85,
        "LLM hallucinating",
        "Tighten prompt, lower temperature",
    ),
    "context_recall": (
        0.75,
        "Missing relevant chunks",
        "Improve chunking or add BM25",
    ),
    "context_precision": (
        0.75,
        "Too many irrelevant chunks",
        "Add reranking or metadata filter",
    ),
    "answer_relevancy": (
        0.80,
        "Answer doesn't match question",
        "Improve prompt template",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Phân tích bottom-N câu hỏi kém nhất theo Diagnostic Tree.

    Args:
        eval_results: danh sách EvalResult từ evaluate_ragas()
        bottom_n: số lượng failures cần phân tích

    Returns:
        list[dict] với keys: question, worst_metric, score, diagnosis, suggested_fix
    """
    if not eval_results:
        return []

    def _avg_score(r: EvalResult) -> float:
        return (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4

    # Sort ascending — worst questions đầu tiên
    sorted_results = sorted(eval_results, key=_avg_score)
    bottom = sorted_results[:bottom_n]

    output: list[dict] = []
    for r in bottom:
        metric_scores = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy,
        }
        worst_metric = min(metric_scores, key=metric_scores.get)  # type: ignore[arg-type]
        worst_score = metric_scores[worst_metric]

        threshold, diagnosis, suggested_fix = _DIAGNOSTIC_TREE[worst_metric]
        # Nếu worst metric vẫn trên ngưỡng → overall low quality
        if worst_score >= threshold:
            diagnosis = "Low overall quality"
            suggested_fix = "Review retrieval corpus and prompt configuration"

        output.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return output


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
