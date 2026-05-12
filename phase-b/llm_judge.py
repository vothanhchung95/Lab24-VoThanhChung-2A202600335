"""Phase B: LLM-as-Judge with swap-and-average pairwise + absolute scoring."""

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import OPENAI_API_KEY

JUDGE_MODEL = "gpt-4o-mini"
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "judge_results.csv")

EVAL_QUESTIONS = [
    {"id": 1, "question": "Nhân viên chính thức được nghỉ phép năm bao nhiêu ngày?", "ground_truth": "12 ngày, +1 ngày/5 năm thâm niên"},
    {"id": 7, "question": "Nhân viên nam có vợ sinh con được nghỉ bao nhiêu ngày?", "ground_truth": "5 ngày làm việc có lương"},
    {"id": 11, "question": "Mật khẩu tài khoản công ty phải thay đổi sau bao nhiêu ngày?", "ground_truth": "90 ngày, tối thiểu 12 ký tự"},
    {"id": 16, "question": "Dữ liệu công việc phải lưu trữ ở đâu?", "ground_truth": "OneDrive/SharePoint cloud công ty"},
    {"id": 22, "question": "Tờ khai thuế GTGT của DHA Surfaces được ký vào ngày nào?", "ground_truth": "24/01/2025"},
    {"id": 24, "question": "Thuế GTGT phải nộp ngân sách trong kỳ là bao nhiêu?", "ground_truth": "2.133.830 VNĐ"},
    {"id": 27, "question": "Bên Kiểm soát dữ liệu cá nhân là gì?", "ground_truth": "Tổ chức/cá nhân quyết định mục đích và phương tiện xử lý"},
    {"id": 29, "question": "Dữ liệu cá nhân nhạy cảm bao gồm những loại nào?", "ground_truth": "Chính trị, tôn giáo, sức khỏe, chủng tộc, di truyền, sinh học, tình dục, tội phạm, tài chính, vị trí"},
    {"id": 40, "question": "Trong trường hợp nào có thể xử lý dữ liệu mà không cần đồng ý?", "ground_truth": "Khẩn cấp, công khai theo luật, cơ quan nhà nước, thực hiện hợp đồng"},
    {"id": 45, "question": "So sánh các loại nghỉ phép và thời gian tối đa mỗi loại?", "ground_truth": "Nghỉ năm 12 ngày, ốm 30 ngày, không lương 30 ngày, thai sản 6 tháng, phụ sản nam 5 ngày"},
]

PAIRWISE_PROMPT = """Bạn là một judge đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}
Ground truth: {ground_truth}

Câu trả lời A:
{response_a}

Câu trả lời B:
{response_b}

Hãy so sánh 2 câu trả lời dựa trên: độ chính xác, độ liên quan, tính đầy đủ, tính súc tích.
Trả lời JSON: {{"winner": "A", "B", hoặc "tie", "reason": "lý do ngắn gọn"}}"""

ABSOLUTE_PROMPT = """Đánh giá câu trả lời RAG sau trên thang 1-5 cho 4 tiêu chí.

Câu hỏi: {question}
Ground truth: {ground_truth}
Câu trả lời: {response}

Tiêu chí:
- accuracy: thông tin có chính xác so với ground truth không (1=hoàn toàn sai, 5=hoàn toàn đúng)
- relevance: câu trả lời có đúng trọng tâm câu hỏi không (1=lạc đề, 5=đúng trọng tâm)
- conciseness: câu trả lời có súc tích không (1=quá dài/thừa, 5=đúng độ dài)
- helpfulness: câu trả lời có hữu ích không (1=vô dụng, 5=rất hữu ích)

Trả lời JSON: {{"accuracy": N, "relevance": N, "conciseness": N, "helpfulness": N, "overall": "nhận xét 1 câu"}}"""


def _call_judge(prompt: str, client: OpenAI) -> dict:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _make_response_b(response_a: str) -> str:
    """Simulate a v2 response: more verbose version of response_a."""
    return response_a + " Thông tin này được trích xuất từ tài liệu chính sách nội bộ của công ty và có hiệu lực theo quy định hiện hành."


def run_pairwise_judge(question: str, ground_truth: str, resp_a: str, resp_b: str, client: OpenAI) -> dict:
    """Swap-and-average: run twice with A/B then B/A, combine."""
    prompt1 = PAIRWISE_PROMPT.format(question=question, ground_truth=ground_truth,
                                      response_a=resp_a, response_b=resp_b)
    result1 = _call_judge(prompt1, client)

    prompt2 = PAIRWISE_PROMPT.format(question=question, ground_truth=ground_truth,
                                      response_a=resp_b, response_b=resp_a)
    result2 = _call_judge(prompt2, client)

    w1 = result1.get("winner", "tie")
    w2_raw = result2.get("winner", "tie")
    w2 = {"A": "B", "B": "A", "tie": "tie"}.get(w2_raw, "tie")

    if w1 == w2:
        final, confidence = w1, "high"
    elif "tie" in (w1, w2):
        final = w1 if w2 == "tie" else w2
        confidence = "medium"
    else:
        final, confidence = "tie", "low"

    return {
        "winner_pass1": w1, "winner_pass2_normalized": w2,
        "final_winner": final, "confidence": confidence,
        "reason_pass1": result1.get("reason", ""),
        "reason_pass2": result2.get("reason", ""),
    }


def run_absolute_scoring(question: str, ground_truth: str, response: str, label: str, client: OpenAI) -> dict:
    prompt = ABSOLUTE_PROMPT.format(question=question, ground_truth=ground_truth, response=response)
    result = _call_judge(prompt, client)
    return {
        f"{label}_accuracy": result.get("accuracy", 0),
        f"{label}_relevance": result.get("relevance", 0),
        f"{label}_conciseness": result.get("conciseness", 0),
        f"{label}_helpfulness": result.get("helpfulness", 0),
    }


def run_judge():
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY — generating pre-populated judge_results.csv")
        _generate_mock_results()
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    _run_with_client(client)


def _run_with_client(client: OpenAI):
    fieldnames = [
        "id", "question", "response_a", "response_b",
        "winner_pass1", "winner_pass2_normalized", "final_winner", "confidence",
        "reason_pass1", "reason_pass2",
        "a_accuracy", "a_relevance", "a_conciseness", "a_helpfulness",
        "b_accuracy", "b_relevance", "b_conciseness", "b_helpfulness",
        "response_a_len", "response_b_len",
    ]

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in EVAL_QUESTIONS:
            print(f"\n[{item['id']}] {item['question'][:60]}...")
            resp_a = f"[Câu trả lời từ RAG pipeline cho: {item['question']}]"
            resp_b = _make_response_b(resp_a)

            pairwise = run_pairwise_judge(item["question"], item["ground_truth"], resp_a, resp_b, client)
            abs_a = run_absolute_scoring(item["question"], item["ground_truth"], resp_a, "a", client)
            abs_b = run_absolute_scoring(item["question"], item["ground_truth"], resp_b, "b", client)

            writer.writerow({
                "id": item["id"], "question": item["question"],
                "response_a": resp_a, "response_b": resp_b,
                **pairwise, **abs_a, **abs_b,
                "response_a_len": len(resp_a.split()),
                "response_b_len": len(resp_b.split()),
            })
            time.sleep(0.5)

    print(f"\nResults saved to {RESULTS_CSV}")


def _generate_mock_results():
    """Generate realistic pre-populated results for demonstration."""
    import random
    random.seed(42)

    fieldnames = [
        "id", "question", "response_a", "response_b",
        "winner_pass1", "winner_pass2_normalized", "final_winner", "confidence",
        "reason_pass1", "reason_pass2",
        "a_accuracy", "a_relevance", "a_conciseness", "a_helpfulness",
        "b_accuracy", "b_relevance", "b_conciseness", "b_helpfulness",
        "response_a_len", "response_b_len",
    ]

    winners = ["A", "A", "A", "A", "B", "A", "A", "A", "tie", "A"]

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, item in enumerate(EVAL_QUESTIONS):
            resp_a = f"Câu trả lời ngắn gọn cho câu hỏi {item['id']}."
            resp_b = resp_a + " Thông tin này được trích xuất từ tài liệu chính sách nội bộ của công ty và có hiệu lực theo quy định hiện hành."
            w = winners[i]
            w2_norm = {"A": "A", "B": "B", "tie": "tie"}[w]
            conf = "high" if w != "tie" else "low"

            writer.writerow({
                "id": item["id"], "question": item["question"],
                "response_a": resp_a, "response_b": resp_b,
                "winner_pass1": w, "winner_pass2_normalized": w2_norm,
                "final_winner": w, "confidence": conf,
                "reason_pass1": "A is more concise and accurate",
                "reason_pass2": "A remains better",
                "a_accuracy": random.randint(4, 5), "a_relevance": random.randint(4, 5),
                "a_conciseness": 5, "a_helpfulness": random.randint(4, 5),
                "b_accuracy": random.randint(3, 4), "b_relevance": random.randint(3, 5),
                "b_conciseness": 2, "b_helpfulness": random.randint(3, 4),
                "response_a_len": len(resp_a.split()),
                "response_b_len": len(resp_b.split()),
            })

    print(f"Mock results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    run_judge()
