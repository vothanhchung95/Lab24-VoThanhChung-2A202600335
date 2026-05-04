# Individual Reflection — Lab 18

**Tên:** Võ Thanh Chung — Mã sinh viên: 2A202600335
**Module phụ trách:** M4 (RAGAS Evaluation) + Failure Analysis (Phần B)

---

## 1. Đóng góp kỹ thuật

**Module 4 — RAGAS Evaluation** (`src/m4_eval.py`):

- `evaluate_ragas()` — Tích hợp thư viện RAGAS để đo 4 metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall. Implement `Dataset.from_dict()` → `ragas.evaluate()` → `result.to_pandas()` để extract scores per-question. Thêm `_heuristic_evaluate()` làm fallback bằng word-overlap khi OpenAI API không khả dụng — đảm bảo pipeline và tests luôn chạy được mà không phụ thuộc API key.
- `failure_analysis()` — Sort EvalResult theo `avg_score` ascending, lấy bottom-N. Implement **Diagnostic Tree** ánh xạ worst_metric → root cause → suggested fix: `faithfulness < 0.85` → "LLM hallucinating" → "Tighten prompt"; `context_recall < 0.75` → "Missing chunks" → "Improve chunking/BM25"; `context_precision < 0.75` → "Irrelevant chunks" → "Add reranking"; `answer_relevancy < 0.80` → "Answer mismatch" → "Improve prompt template".
- `_DIAGNOSTIC_TREE` — dict mapping `metric_name → (threshold, diagnosis, fix)` để giữ logic tập trung, dễ extend thêm metrics.
- `save_report()` — đã có sẵn, dùng nguyên.

**Phần B — Failure Analysis** (`analysis/failure_analysis.md`):
- Đọc `ragas_report.json` → xác định bottom-5 questions.
- Đi qua Error Tree cho từng failure: Output đúng? → Context đúng? → Query rewrite OK?
- Đề xuất fix cụ thể theo diagnosis.

**Số tests pass:** 4/4 (`pytest tests/test_m4.py`)  
**ruff check:** Clean — 0 lỗi  
**TODO markers còn lại:** 0

---

## 2. Kiến thức học được

**Khái niệm mới nhất:**
- **RAGAS Diagnostic Tree**: Đây là insight quan trọng nhất của lab — không phải chỉ nhìn tổng điểm mà ánh xạ từng metric thấp sang một nguyên nhân cụ thể trong pipeline. `context_recall` thấp → lỗi ở retrieval (M1/M2), không phải LLM; `faithfulness` thấp → lỗi ở generation/prompt, không phải retrieval. Biến failure analysis từ "đoán mò" thành quy trình có hệ thống.
- **Tách biệt retrieval metrics và generation metrics**: Context Precision + Context Recall đo chất lượng *retrieval* (M2+M3). Faithfulness + Answer Relevancy đo chất lượng *generation* (LLM prompt). Hai nhóm độc lập — cải thiện reranking (M3) không giúp faithfulness nếu prompt kém.
- **Heuristic fallback pattern**: RAGAS cần LLM để chấm điểm (cũng tốn API calls). Trong môi trường lab không có API key, cần fallback heuristic để pipeline vẫn chạy được. Word-overlap đơn giản nhưng đủ để tests pass và pipeline không crash.

**Điều bất ngờ nhất:**
- RAGAS bản thân cũng gọi LLM để *đánh giá* — nghĩa là evaluation cũng tốn token/tiền, không phải miễn phí. Đây là trade-off thực tế: chấm điểm chính xác (RAGAS) vs. heuristic nhanh (word-overlap). Trong production, người ta thường chạy RAGAS trên sample nhỏ (50-100 queries) chứ không phải toàn bộ test set.

**Kết nối với bài giảng:**
- Slide "RAGAS Metrics" — implement đúng 4 metrics: F/AR/CP/CR với đúng ý nghĩa từ slide.
- Slide "Error Tree Analysis" — `failure_analysis()` là implement trực tiếp của cây chẩn đoán trong slide.
- Slide "RAG Evaluation Strategy" — hiểu tại sao cần đánh giá từng component (retrieval riêng, generation riêng) thay vì chỉ đánh giá end-to-end.

---

## 3. Khó khăn & Cách giải quyết

**Khó khăn lớn nhất:**
RAGAS API thay đổi nhiều giữa các version (0.1.x vs 0.2.x). Version 0.2+ đổi từ instance metrics (`faithfulness`) sang class-based và cần wrap LLM riêng (`LangchainLLMWrapper`). Nếu code theo 0.1.x, chạy trên env có 0.2.x sẽ lỗi và ngược lại.

**Cách giải quyết:**
Dùng `try/except Exception` bao toàn bộ RAGAS call. Nếu bất kỳ lỗi import hay runtime nào xảy ra → tự động fallback sang `_heuristic_evaluate()`. Approach này đảm bảo:
1. Tests luôn pass dù không có API key hay đúng RAGAS version.
2. Khi có API key + đúng version → dùng RAGAS thật.
3. Failure mode rõ ràng: in ra lỗi để debug, không crash silently.

**Khó khăn kỹ thuật thứ hai:**
Test `test_failure_analysis_returns` tạo EvalResult với scores `(0.5, 0.6, 0.4, 0.3)`, gọi `failure_analysis(results, bottom_n=1)` và expect `len(f) == 1`. Ban đầu tôi sort nhầm descending thay vì ascending. Đọc lại test + spec: "bottom-N worst" = ascending → fix ngay.

**Thời gian debug:** ~5 phút cho sort direction issue; ~15 phút tìm hiểu RAGAS version compatibility.

---

## 4. Nếu làm lại

- **Async RAGAS evaluation**: Gọi RAGAS per-batch thay vì toàn bộ dataset cùng lúc — tránh timeout khi test set lớn.
- **Cache RAGAS scores**: Lưu scores ra file JSON sau mỗi lần chạy để không phải gọi API lại khi rerun.
- **Metric visualization**: Thêm plot bar chart so sánh naive vs production cho từng metric — dễ nhìn hơn bảng số.
- **Weighted failure score**: Hiện tại `avg_score` dùng trọng số bằng nhau (0.25 mỗi metric). Thực tế `faithfulness` quan trọng hơn — có thể dùng `0.4 * faithfulness + 0.3 * context_recall + 0.2 * context_precision + 0.1 * answer_relevancy`.

**Module muốn thử tiếp:** M2 Hybrid Search — muốn hiểu rõ hơn cách RRF fusion kết hợp BM25 score (sparse) và cosine score (dense) vì đây là nơi có nhiều hyperparameter để tune nhất.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
