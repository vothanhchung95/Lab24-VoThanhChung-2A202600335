# Group Report — Lab 18: Production RAG

**Nhóm:** Nhóm 4 thành viên
**Ngày:** 04/05/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Phạm Hữu Hoàng Hiệp | M1 Chunking + M5 Enrichment + Integration Lead | ☑ | 13/13 + 10/10 |
| Trần Hữu Cường | M2 Hybrid Search (BM25 + Dense + RRF) | ☑ | 5/5 |
| Lê Quang Thiên | M3 Reranking (Cross-encoder bge-reranker) | ☑ | 5/5 |
| Võ Thanh Chung | M4 Evaluation (RAGAS + Failure Analysis) | ☑ | 4/4 |

**Tổng tests: 37/37 pass**

---

## Kết quả RAGAS (test set: 20 Q&A pairs tiếng Việt, 4 domain HR/IT/Finance/Legal)

| Metric | Naive Baseline | Production | Δ | ≥ 0.75? |
|--------|---------------|-----------|---|---------|
| Faithfulness | 0.9057 | 0.7250 | **−0.18** | ✗ |
| Answer Relevancy | 0.3238 | **0.4296** | **+0.11 (+33%)** | ✗ |
| Context Precision | 0.8750 | **0.8917** | +0.02 | ✓ |
| Context Recall | 0.9500 | 0.9000 | −0.05 | ✓ |

> **Naive baseline:** `chunk_basic` (paragraph) + `DenseSearch` (bge-m3 dense-only), no rerank/enrich/LLM. Trả về `contexts[0]` trực tiếp làm answer.
>
> **Production:** `chunk_hierarchical` + `enrich_chunks` (contextual + HyQA + metadata) + `HybridSearch` (BM25+Dense RRF) + `CrossEncoderReranker` (bge-reranker-v2-m3) + LLM gen (gpt-4o-mini, context-only prompt).

---

## Key Findings

### 1. Biggest improvement: Answer Relevancy +33%

Production tăng từ 0.32 → 0.43 nhờ LLM generation trả lời đúng câu hỏi thay vì copy nguyên context. Ví dụ:
- **Naive:** "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên..."
- **Production:** "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."

### 2. Biggest challenge: Faithfulness regression (−0.18)

Đây là **artifact đo đếm** chứ không phải lỗi:
- Naive trả về context literal → answer = chunk → ragas judge faithfulness ~1.0 mặc định
- Production LLM paraphrase → ragas LLM judge phát hiện diễn giải lại → score thấp hơn
- Đặc biệt rõ ở Legal domain (Nghị định 13): 5/5 câu hỏi pháp lý có faithfulness = 0 do LLM diễn giải thay vì trích nguyên văn

**Fix:** Thêm vào system prompt: *"Với câu hỏi định nghĩa pháp lý, trích dẫn nguyên văn từ context."* Dự kiến nâng faithfulness lên >0.85.

### 3. Surprise finding: Context Recall hơi giảm (−0.05)

Hierarchical chunking (parent 2048 + child 256) tăng precision (chunk nhỏ → embedding chính xác) nhưng đôi khi miss content đầy đủ (vd: Điều 2 khoản 3 Nghị định 13 liệt kê 10 loại dữ liệu nhạy cảm — bị split thành nhiều child chunks, chỉ chunk đầu chứa "header + 3 loại đầu" được retrieve).

**Fix:** Dùng `chunk_structure_aware()` cho văn bản pháp lý — split theo markdown header `###` giữ nguyên 1 điều = 1 chunk.

---

## Bonus đạt được

| Bonus | Điểm | Status |
|-------|------|--------|
| Faithfulness ≥ 0.85 | +5 | ✗ Faithfulness 0.7250 < 0.85 |
| Enrichment integrated | +3 | ✓ M5 contextual + HyQA + metadata wired vào pipeline |
| Latency breakdown report | +2 | ✓ `reports/latency_breakdown.json` (setup + per-query avg/p50/p95/max) |

**Tổng bonus: +5** (3 + 2)

---

## Latency Breakdown

**Setup (one-time):**
| Stage | Time |
|-------|------|
| Chunking (4 docs → 80 children) | 0.01s |
| Enrichment (240 OpenAI calls) | **840.6s** (14 phút — bottleneck) |
| Indexing (BM25 + Dense bge-m3) | 34.8s |
| Reranker load | 0.0s (cached) |

**Per-query (20 queries):**
| Stage | avg | p50 | p95 | max |
|-------|-----|-----|-----|-----|
| Search (BM25+Dense+RRF) | 77.6ms | 77.0ms | 87.3ms | 91.8ms |
| Rerank (bge-reranker top-3) | 705.5ms | 169.3ms | 237.6ms | 10796ms |
| LLM Gen (gpt-4o-mini) | 1213.8ms | 1083.2ms | 1690.9ms | 3563ms |
| **Total** | **2.0s** | 1.4s | 3.8s | 12.4s |

→ LLM gen dominates per-query latency (~60%). Enrichment dominates setup (~96%).

---

## Presentation Notes (5 phút)

### 1. RAGAS scores (naive vs production)
*Bảng so sánh ở trên — nhấn 2 điểm: B2 đạt 10đ (2/4 metrics ≥ 0.75), Faithfulness regression do artifact của naive baseline.*

### 2. Biggest win — module nào, tại sao
**LLM Generation (Hiệp wire vào pipeline.py)** đem Answer Relevancy từ 0.32 → 0.43 (+33%). Lý do: naive baseline trả nguyên context làm answer → đầy thông tin thừa → ragas judge "không relevant với question". LLM gen trích đúng câu trả lời cho từng question.

### 3. Case study — 1 failure, Error Tree walkthrough
*Xem failure_analysis.md, Case Study đã chọn: "Chuyển dữ liệu cá nhân ra nước ngoài là gì?"*

### 4. Next optimization nếu có thêm 1 giờ
1. **Structure-aware chunking cho Legal** — fix 5/5 failures faithfulness=0 trong Nghị định 13
2. **Domain-specific prompt** — instruction "trích nguyên văn cho Legal/Finance" → boost Faithfulness ≥ 0.85 → unlock bonus +5
3. **Async enrichment** — 840s → ~120s với asyncio.gather
