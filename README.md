# Lab 18: Production RAG Pipeline

**AICB-P2T3 · Ngày 18 · Production RAG**  
**Giảng viên:** M.Sc Trần Minh Tú · **Thời gian:** 2 giờ

---

## Tổng quan

Lab gồm **2 phần**:

| Phần | Hình thức | Thời gian | Mô tả |
|------|-----------|-----------|-------|
| **Phần A** | Cá nhân | 1.5 giờ | Implement 1 trong 4 modules |
| **Phần B** | Nhóm (3–4 người) | 30 phút | Ghép modules → full pipeline → eval → present |

```
  Cá nhân                         Nhóm
  ┌────────────┐
  │ M1 Chunking│──┐
  ├────────────┤  │    ┌──────────────────────────────┐
  │ M2 Search  │──┼───▶│  Production RAG System        │
  ├────────────┤  │    │  pipeline.py + RAGAS eval     │
  │ M3 Rerank  │──┤    │  + failure analysis           │
  ├────────────┤  │    └──────────────────────────────┘
  │ M4 Eval    │──┘
  └────────────┘
```

## Quick Start

```bash
git clone <repo-url> && cd lab18-production-rag
docker compose up -d                    # Qdrant
pip install -r requirements.txt
cp .env.example .env                    # Điền API keys
python naive_baseline.py                # ⚠️ Chạy TRƯỚC để có baseline
```

## Chạy toàn bộ

```bash
python main.py                          # Naive + Production + So sánh
python check_lab.py                     # Kiểm tra trước khi nộp
```

## Cấu trúc repo

```
lab18-production-rag/
├── README.md                   # File này
├── ASSIGNMENT_INDIVIDUAL.md    # ★ Đề bài cá nhân (Phần A)
├── ASSIGNMENT_GROUP.md         # ★ Đề bài nhóm (Phần B)
├── RUBRIC.md                   # Hệ thống chấm điểm chi tiết
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies (pinned)
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Sample corpus tiếng Việt
│   ├── sample_01.md
│   ├── sample_02.md
│   └── sample_03.md
├── test_set.json               # 20 Q&A pairs
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   └── pipeline.py             # Ghép nhóm
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   └── test_m4.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (nhóm)
│   ├── group_report.md         # Báo cáo nhóm
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (sau khi chạy main.py)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    ├── failure_analysis.md
    └── group_report.md
```

## Timeline

| Thời gian | Hoạt động |
|-----------|-----------|
| 0:00–0:15 | Setup + chạy `naive_baseline.py` |
| 0:15–1:45 | **Phần A (cá nhân):** implement module → `pytest tests/test_m*.py` |
| 1:45–2:15 | **Phần B (nhóm):** ghép → `python src/pipeline.py` → failure analysis |
| 2:15–2:30 | Presentation 5 phút/nhóm |

---

## Lab 24 — Evaluation & Guardrail System

Lab 24 extends the Lab 18 production RAG pipeline with a four-phase evaluation and safety layer. Where Lab 18 focused on building the retrieval and generation pipeline (chunking, hybrid search, reranking, RAGAS scoring), Lab 24 measures how trustworthy and safe that pipeline is in practice — and adds guardrails that block harmful or off-topic requests before they reach the LLM.

**Phase A — RAGAS Evaluation** runs the full pipeline against a 20-question Vietnamese test set and reports four metrics: faithfulness (0.725), answer relevancy (0.430), context precision (0.892), and context recall (0.900). High precision/recall confirm that the retriever surfaces the right chunks; the lower faithfulness score flags cases where the LLM paraphrases rather than quotes the source — a known failure mode on legal/financial Vietnamese text.

**Phase B — LLM-as-Judge** implements two complementary judgment strategies. The pairwise judge uses a swap-and-average protocol (running each comparison twice with A↔B reversed) to cancel position bias, then aggregates the two verdicts into a confidence-weighted winner. The absolute judge scores each response on four 1–5 Likert dimensions (Accuracy, Relevance, Conciseness, Helpfulness). A calibration step computes Cohen's kappa between human and LLM ratings on 10 shared samples, confirming inter-rater reliability.

**Phase C — Guardrails** adds a three-layer async pipeline: PII redaction and topic validation run in parallel on the user query; if the topic is rejected the request is blocked immediately; otherwise the RAG answer passes through Llama Guard 3 (via Groq API) for output safety classification. The benchmark on 10 test queries blocks 4 harmful/off-topic inputs (40% block rate) and passes 6 legitimate HR/IT/legal questions.

**Phase D — Blueprint** documents the proposed production architecture in `phase-d/blueprint.md`.

### Running each phase

```bash
# Phase A: RAGAS evaluation
python phase-a/ragas_eval.py

# Phase B: LLM judge then calibration
python phase-b/llm_judge.py
python phase-b/calibration.py

# Phase C: Guardrail benchmark
python phase-c/guardrails.py

# Phase D: read the blueprint
# See phase-d/blueprint.md
```

### Required environment variables

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | Phase A (RAGAS), Phase B (LLM judge), src/pipeline.py (generation) |
| `GROQ_API_KEY` | Phase C (Llama Guard 3 via Groq) |

Copy `.env.example` to `.env` and fill in both keys before running any phase.
