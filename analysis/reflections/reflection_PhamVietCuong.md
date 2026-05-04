# Individual Reflection — Lab 18

**Tên:** Phạm Việt Cường — **Mã sinh viên:** 2A202600420  
**Module phụ trách:** **M2 — Hybrid Search** (`src/m2_search.py`, `tests/test_m2.py`)  
**Phần nhóm:** Chạy pipeline đánh giá + `main.py` (baseline vs production, báo cáo JSON)

---

## 1. Đóng góp kỹ thuật

### Module 2 — Hybrid Search (`src/m2_search.py`)

- **`segment_vietnamese()`** — Tách từ tiếng Việt bằng `underthesea.word_tokenize(..., format="text")` để BM25 có ranh giới từ hợp lý (ví dụ cụm như “nghỉ phép” không bị tách sai). Có xử lý chuỗi rỗng và fallback khi thư viện lỗi để pipeline không dừng đột ngột.
- **`BM25Search`** — `index()`: lưu corpus, tokenize từng chunk sau khi segment, dựng `BM25Okapi` từ `rank_bm25`. `search()`: tokenize query, `get_scores`, lấy top-k, trả về `SearchResult` với `method="bm25"`.
- **`DenseSearch`** — `index()`: `recreate_collection` trên Qdrant (cosine, chiều vector theo `EMBEDDING_DIM` trong `config`), encode chunk bằng `SentenceTransformer` (`EMBEDDING_MODEL`, ví dụ BGE-M3), chuẩn hóa vector cho cosine, `upsert` `PointStruct` (payload gồm `text` + metadata). `search()`: encode câu hỏi, gọi `client.query_points` (API client Qdrant hiện tại), map payload → `method="dense"`. *(Tùy phiên bản code nhóm: client có thể trỏ `localhost:6333` theo README/Docker hoặc in-memory để chạy nhanh khi dev.)*
- **`reciprocal_rank_fusion()`** — Gộp hai (hoặc nhiều) danh sách đã xếp hạng theo công thức RRF: cộng dồn \(1/(k + \mathrm{rank} + 1)\) theo khóa là nội dung `text`, sắp xếp giảm dần, trả về `method="hybrid"`.
- **`HybridSearch`** — Gọi BM25 + dense song song rồi RRF; `index()` cập nhật cả hai chỉ mục.

### Kiểm thử (`tests/test_m2.py`)

- Chạy: `pytest tests/test_m2.py -v`
- **Số tests pass:** **5/5** (segment trả về string; BM25 có kết quả và độ liên quan với truy vấn “nghỉ phép”; RRF gộp đúng và `method="hybrid"`).

### Phần nhóm — Pipeline & so sánh báo cáo

- Tham gia chạy **`python -m src.pipeline`** (hoặc lệnh tương đương từ thư mục gốc repo theo hướng dẫn nhóm) để build pipeline production: chunk (M1) → enrichment (M5) → index hybrid (M2) → rerank (M3) → eval RAGAS (M4).
- Chạy **`python main.py`**: baseline (`naive_baseline.py`) → production pipeline → **di chuyển** `naive_baseline_report.json` và `ragas_report.json` vào thư mục **`reports/`** → in **bảng so sánh** metric (Basic vs Production, cột Δ).
- Đầu ra mong đợi: `reports/naive_baseline_report.json`, `reports/ragas_report.json` (và có thể thêm `reports/latency_breakdown.json` nếu pipeline gọi `_save_latency_report`).

---

## 2. Kiến thức học được

**Khái niệm mới / đã củng cố:**

- **Hybrid retrieval**: BM25 (từ khóa, tốt với khớp literal / tiếng Việt sau tokenize) và dense embedding (tốt với paraphrase, ngữ nghĩa) bổ trợ nhau; RRF không cần chuẩn hóa score giữa hai hệ mà vẫn gộp thứ hạng ổn định.
- **RRF (Reciprocal Rank Fusion)**: Hyperparameter `k` (mặc định 60 trong code) điều chỉnh mức “tin” vào thứ hạng đầu danh sách; tài liệu xuất hiện tốt ở cả hai nguồn xếp hạng được đẩy lên rõ rệt.
- **Qdrant + embedding production**: Tách collection theo config (`COLLECTION_NAME` vs baseline), vector cosine + payload chứa `text` để tái dựng `SearchResult`.

**Điều bất ngờ / thực tế triển khai:**

- Client Python Qdrant đổi API (`query_points` thay cho `search` cũ): cần đọc đúng bản client đang cài để tránh lỗi runtime khi nhóm tích hợp.
- **`underthesea`** lần đầu có thể chậm do tải model; BM25 phụ thuộc chất lượng tokenize — nếu segment kém, điểm BM25 và thứ hạng RRF đổi theo.

**Kết nối với bài giảng (điền số slide thực tế buổi học):**

- Slide **sparse vs dense retrieval**, **fusion / RRF**.
- Slide **RAG pipeline**: chunk → index → retrieve → (rerank) → generate → đo RAGAS — M2 nằm đúng lớp **retrieve**.

---

## 3. Khó khăn & Cách giải quyết

**Khó khăn lớn nhất:**

- Đồng bộ **môi trường**: thiếu `qdrant_client`, lỗi kết nối Qdrant (Docker chưa `up`), hoặc lần đầu tải **BGE-M3** rất lâu → pipeline nhóm chạy một lần mất nhiều phút hoặc timeout.

**Cách giải quyết:**

- Chuẩn hóa môi trường bằng **`requirements.txt`**; nếu pipeline dùng Qdrant Docker thì bật theo README (`docker compose up -d`) và kiểm tra `localhost:6333` trước khi chạy eval.
- Chạy test M2 trước (`pytest tests/test_m2.py`) để xác nhận BM25 + RRF không phụ thuộc toàn bộ pipeline nặng.
- Nếu máy yếu: giảm tạm số câu trong `test_set.json` khi debug (nhớ khôi phục khi nộp bài theo quy định nhóm).

**Khó khăn khác:**

- Import khi gọi script: từ thư mục gốc repo nên dùng **`python -m src.pipeline`** để package `src` resolve đúng; nếu dùng `python src/pipeline.py` cần đảm bảo `PYTHONPATH` hoặc cấu trúc import theo hướng dẫn giảng viên.

**Thời gian debug (ước lượng — tự điều chỉnh cho sát thực tế):** [ví dụ: 30–90 phút] (API Qdrant + embedding + phối hợp nhóm).

---

## 4. Nếu làm lại

- Tune **`k` trong RRF** và **`BM25_TOP_K` / `DENSE_TOP_K`** trên tập dev nhỏ trước khi chạy full `test_set.json`.
- Thử **chuẩn hóa / stem** bổ sung cho BM25 (nếu cho phép) hoặc logging trung gian (top-5 BM25 vs top-5 dense) để giải thích vì sao RRF chọn doc.
- Viết **smoke test** một query end-to-end (index 3 chunk giả → hybrid search) không cần GPU.
- **Module muốn thử tiếp:** M3 rerank (CrossEncoder) để xem hybrid đã lấy đủ recall chưa trước khi cắt top-3.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 4 |


