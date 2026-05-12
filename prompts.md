# LLM Prompts — Lab 24

## 1. RAG Generation Prompt (src/pipeline.py)

**Purpose:** Constrain LLM to answer only from retrieved context, maximizing faithfulness.

**System prompt:**

```
Bạn là trợ lý chỉ trả lời dựa CHÍNH XÁC trên context được cung cấp.
Tuyệt đối không suy đoán, không dùng kiến thức ngoài context.
Nếu context không có thông tin → trả lời: 'Không tìm thấy thông tin trong tài liệu.'
Trả lời ngắn gọn, đúng trọng tâm câu hỏi, bằng tiếng Việt.
```

**User turn template:**

```
Context:
{context_str}

Câu hỏi: {query}

Trả lời:
```

**Model:** `gpt-4o-mini`, `temperature=0.0`, `max_tokens=300`

**Design rationale:** Setting `temperature=0.0` eliminates sampling randomness, making responses deterministic and reproducible. The Vietnamese-language constraint ("tuyệt đối không suy đoán") closes the gap between what the model knows and what the retrieved chunks contain — a key driver of the faithfulness metric in RAGAS. The explicit fallback phrase ("Không tìm thấy thông tin trong tài liệu.") ensures the model signals retrieval failure rather than hallucinating, which is especially important for the Vietnamese legal/financial domain where subtle paraphrasing errors are hard to detect.

---

## 2. LLM-as-Judge: Pairwise Comparison (phase-b/llm_judge.py)

**Purpose:** Compare two responses A vs B, returning a winner to measure relative quality.

**Prompt:**

```
Bạn là một judge đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}
Ground truth: {ground_truth}

Câu trả lời A:
{response_a}

Câu trả lời B:
{response_b}

Hãy so sánh 2 câu trả lời dựa trên: độ chính xác, độ liên quan, tính đầy đủ, tính súc tích.
Trả lời JSON: {"winner": "A", "B", hoặc "tie", "reason": "lý do ngắn gọn"}
```

**Design rationale:** Swap-and-average design (run twice with A↔B swapped) cancels position bias — LLMs tend to favor the response shown first. The two passes produce winners `w1` and `w2_normalized`; if they agree the confidence is "high", if one is a tie the confidence is "medium", and if they disagree the final verdict is forced to "tie" with confidence "low". Requiring JSON output (`response_format={"type": "json_object"}`) eliminates free-text parsing errors. Providing the ground truth as an explicit reference anchors the judge to factual correctness rather than surface fluency.

---

## 3. LLM-as-Judge: Absolute Scoring (phase-b/llm_judge.py)

**Purpose:** Score a single response on 4 dimensions (Accuracy, Relevance, Conciseness, Helpfulness) for absolute quality measurement.

**Prompt:**

```
Đánh giá câu trả lời RAG sau trên thang 1-5 cho 4 tiêu chí.

Câu hỏi: {question}
Ground truth: {ground_truth}
Câu trả lời: {response}

Tiêu chí:
- accuracy: thông tin có chính xác so với ground truth không (1=hoàn toàn sai, 5=hoàn toàn đúng)
- relevance: câu trả lời có đúng trọng tâm câu hỏi không (1=lạc đề, 5=đúng trọng tâm)
- conciseness: câu trả lời có súc tích không (1=quá dài/thừa, 5=đúng độ dài)
- helpfulness: câu trả lời có hữu ích không (1=vô dụng, 5=rất hữu ích)

Trả lời JSON: {"accuracy": N, "relevance": N, "conciseness": N, "helpfulness": N, "overall": "nhận xét 1 câu"}
```

**Design rationale:** Absolute scoring complements pairwise comparison by assigning interpretable numeric scores independent of any comparison baseline. The 1–5 Likert scale is granular enough to discriminate quality levels while remaining tractable for the judge model. Conciseness is included as a standalone criterion because verbose responses (e.g., response_b in the benchmark) may score high on accuracy but penalize the user experience. Including an "overall" free-text field encourages the model to reason holistically before committing to scores, which improves score calibration.

---

## 4. Llama Guard 3 (phase-c/llama_guard.py)

**Purpose:** Output safety classification using Meta's Llama Guard 3 model via Groq API.

**Model:** `llama-guard-3-8b`

**Input format:** A single user turn containing the RAG response text to be evaluated:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": response_text}
        ]
    }
]
```

**Parameters:** `max_tokens=100`, `temperature=0.0`

**Output:** `"safe"` or `"unsafe\n<category>"` (e.g., `"unsafe\nS2"`)

**Fallback:** If `GROQ_API_KEY` is not set or the API call fails, the system falls back to a keyword heuristic that scans the response for a hardcoded list of harmful terms in Vietnamese and English (e.g., "vũ khí", "ma túy", "how to make a bomb").

**Design rationale:** Llama Guard 3 uses a built-in safety taxonomy aligned with the MLCommons hazard categories, which covers a broader threat surface than hand-crafted keyword lists. Running it at `temperature=0.0` makes classifications deterministic. The keyword fallback ensures the guardrail layer degrades gracefully when the Groq API is unavailable, maintaining a minimum safety floor without blocking the entire pipeline. The model is invoked only on the RAG output (not the raw query), which limits latency impact and focuses the check on what the user actually receives.
