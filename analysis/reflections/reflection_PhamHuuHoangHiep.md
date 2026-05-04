# Individual Reflection — Lab 18

**Tên:** Phạm Hữu Hoàng Hiệp - 2A202600415
**Module phụ trách:** M1 (Chunking) + M5 (Enrichment) + Integration Lead

---

## 1. Đóng góp kỹ thuật

**Module 1 — Advanced Chunking** (`src/m1_chunking.py`):
- `chunk_hierarchical()` — Parent (~2048 chars) + Child sliding window (256 chars), `parent_id` link giữa parent.metadata và child.parent_id để pipeline retrieve child → return parent context.
- `chunk_semantic()` — Split sentences theo regex `(?<=[.!?])\s+|\n\n`, encode bằng `all-MiniLM-L6-v2`, gom các câu liên tiếp khi cosine similarity ≥ threshold.
- `chunk_structure_aware()` — Regex split markdown headers `(^#{1,3}\s+.+$)`, pair header + content thành chunk có metadata `section`.
- `compare_strategies()` — Aggregate stats (num_chunks, avg/min/max length) + in bảng so sánh 4 strategies.
- Cache encoder ở module level (`_SEMANTIC_ENCODER`) để tránh load lại ~2s mỗi lần gọi.

**Module 5 — Enrichment Pipeline** (`src/m5_enrichment.py`):
- `contextual_prepend()` — LLM viết 1 câu context (prompt VI), prepend với `\n\n` để đảm bảo text gốc giữ nguyên (substring constraint).
- `summarize_chunk()` — `gpt-4o-mini` tóm tắt 2-3 câu VI; có extractive fallback (2 câu đầu) khi thiếu API key.
- `generate_hypothesis_questions()` — Generate N câu hỏi VI để bridge vocabulary gap, parse + strip enumeration prefix.
- `extract_metadata()` — Dùng `response_format=json_object`, parse `{topic, entities, category, language}`, defensive try/except cho malformed JSON.
- `enrich_chunks()` — Orchestrate 4 techniques, support `methods=["contextual", "hyqa", "metadata"]`, giữ nguyên `original_text` cho mỗi `EnrichedChunk`.
- Cache OpenAI client ở module level.

**Integration Lead — `src/pipeline.py`**:
- Wire LLM generation thật (`gpt-4o-mini`, temperature=0.0, prompt ràng buộc context-only) thay placeholder `contexts[0]` → critical cho Faithfulness ≥ 0.85 (+5 bonus).
- Add `LATENCY_STATS` dict + `_save_latency_report()` → bảng setup-time + per-query (avg/p50/p95/max) cho search/rerank/llm/total → save `reports/latency_breakdown.json` (+2 bonus).

**Infrastructure & Data Engineering** (ngoài scope cá nhân, cần thiết để toàn nhóm chạy được):
- Tạo `requirements.txt` (12 deps) + `docker-compose.yml` (Qdrant) — file gốc thiếu khỏi repo.
- OCR 2 PDF (BCTC + Nghị định 13/2023) bằng OpenAI Vision (gpt-4o-mini) — script `ocr_pdfs.py` có cache per-page + retry với backoff.
- Tạo 2 sample MD (HR + IT policy) phù hợp test cases trong `test_m1.py` & `test_m5.py`.
- Tạo `test_set.json` 20 Q&A pairs trải đều 4 domain (HR/IT/Finance/Legal).

**Kết quả tests:**
- M1: **13/13 tests pass** (`pytest tests/test_m1.py`)
- M5: **10/10 tests pass** (`pytest tests/test_m5.py`)
- ruff: clean trên `m1_chunking.py`, `m5_enrichment.py`, `pipeline.py`
- TODO markers: 0/0 còn sót trong code mình owner

---

## 2. Kiến thức học được

**Khái niệm mới nhất:**
- **Hierarchical chunking pattern**: index child (precision của embedding cao do chunk nhỏ) nhưng return parent (đủ context cho LLM). Đây là default recommendation cho production RAG, khác với "fixed-size chunk" naive.
- **Contextual Retrieval (Anthropic)**: thêm 1 câu context vào mỗi chunk trước khi embed → giảm 49% retrieval failure. Cost = one-time enrichment.
- **HyQA (Hypothesis Question-Answer)**: index câu hỏi mà chunk có thể trả lời → bridge vocabulary gap khi user query khác từ vựng với doc.
- **RAGAS Diagnostic Tree**: mỗi metric thấp ánh xạ một root cause cụ thể (faithfulness ↓ → hallucination, context_recall ↓ → missing chunks, etc.) — biến failure analysis từ guess work thành quy trình.

**Điều bất ngờ nhất:**
- Latency của LLM generation (`gpt-4o-mini` ~500-800ms) chiếm phần lớn thời gian per-query, lớn hơn nhiều so với search (BM25+Dense thường <50ms) và rerank (<200ms với bge-reranker). → Optimize đúng chỗ: LLM batch / streaming / nhỏ hơn.
- 2 PDF lab cung cấp là **scanned** (không có text layer), pdfplumber/pypdfium2 đều extract 0 chars. Thực tế production RAG luôn phải có pipeline OCR fallback.

**Kết nối với bài giảng:**
- Slide "Indexing strategies" — hierarchical chunking match pattern parent/child từ slide.
- Slide "Enrichment trước embedding" — implement đúng 4 techniques: summary, HyQA, contextual prepend, auto metadata.
- Slide "Error Tree analysis" — diagnostic mapping trong M4 (do Chung làm) là cốt lõi của failure analysis.

---

## 3. Khó khăn & Cách giải quyết

**Khó khăn lớn nhất:**
- Repo gốc **thiếu hạ tầng**: không có `requirements.txt`, không có `docker-compose.yml`, `data/` chứa PDF scanned thay vì markdown, `test_set.json` chỉ có 1 entry joke + JSON syntax invalid → Phần A bình thường 1.5 giờ phải dành ~30 phút cho infra trước khi code được.

**Cách giải quyết:**
- Suy ra `requirements.txt` từ phân tích imports trong toàn bộ `src/`, `tests/`, `naive_baseline.py`.
- Viết `docker-compose.yml` standard cho Qdrant theo `config.py`.
- OCR 2 PDF bằng OpenAI Vision API — script có cache per-page (resume khi rate limit) + retry exponential backoff (gặp 429 lần đầu khi parallel 5 workers, hạ xuống 2 workers + caching giải quyết).
- Tạo thêm 2 sample MD phù hợp test cases (HR + IT policy) → corpus diverse hơn cho failure analysis.
- Tạo 20 Q&A test set mapping rõ tới content trong corpus để RAGAS evaluate được.

**Khó khăn kỹ thuật:**
- Test M1 `test_hierarchical_valid_parent_ids` yêu cầu `c.parent_id ∈ {p.metadata["parent_id"] for p in parents}` → ban đầu mình chỉ set `parent_id` trên Chunk dataclass, quên set `metadata["parent_id"]` cho parent. Fix bằng helper `_flush_parent` set cả 2.
- Test M5 `test_contextual_contains_original` yêu cầu `SAMPLE in result` → đảm bảo prompt chỉ generate context, dùng `f"{context}\n\n{text}"` để text gốc literal nằm trong output.

**Thời gian debug:** ~10 phút cho 2 issue trên (đọc test kỹ → spec rõ → fix nhanh).

---

## 4. Nếu làm lại

- **Async OpenAI calls** trong `enrich_chunks()` — hiện tại sequential, nếu corpus lớn (>100 chunks) sẽ rất chậm. Dùng `asyncio.gather` + `AsyncOpenAI` sẽ rút thời gian enrichment 5-10x.
- **Adaptive chunk_size cho semantic** — hiện dùng threshold đơn lẻ; có thể dùng adaptive threshold theo độ phong phú vocabulary của doc.
- **Hybrid scoring weight tunable** trong RRF — hiện `k=60` fixed, có thể grid search trên test set.
- **Tracking cost per query** trong latency_breakdown.json — không chỉ ms mà còn $$ (prompt tokens, completion tokens).
- **OCR pipeline tích hợp vào load_documents** — hiện OCR là one-off script, nên integrate `pdf → md` vào load_documents để repo deploy được mà không cần manual step.

**Module muốn thử tiếp:** M3 Reranking — muốn so sánh `bge-reranker-v2-m3` với Cohere Rerank API về quality/latency tradeoff trên tiếng Việt.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 5 |
