# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm 4  
**Thành viên:** Hiệp → M1+M5 · Cường → M2 · Thiên → M3 · Chung → M4

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.52 | 0.84 | **+0.32** |
| Answer Relevancy | 0.58 | 0.79 | **+0.21** |
| Context Precision | 0.61 | 0.78 | **+0.17** |
| Context Recall | 0.67 | 0.77 | **+0.10** |

> *Naive baseline: paragraph chunking + dense-only search, không LLM generation (trả về context[0] trực tiếp)*  
> *Production: hierarchical chunking + hybrid BM25+dense + reranking + GPT-4o-mini generation*

---

## Bottom-5 Failures

### #1
- **Question:** Tổng doanh thu và thuế GTGT hàng hóa, dịch vụ bán ra của DHA Surfaces là bao nhiêu?
- **Expected:** 3.703.685.610 đồng Việt Nam.
- **Got:** DHA Surfaces có doanh thu từ hoạt động kinh doanh trong kỳ báo cáo. Để biết số liệu cụ thể, vui lòng xem bảng cân đối kế toán.
- **Worst metric:** context_recall = 0.41
- **Error Tree:**
  - Output đúng? → **Không** (thiếu con số cụ thể)
  - Context đúng? → **Không** (context không chứa bảng số liệu tài chính, chỉ lấy được phần mô tả chung của BCTC)
  - Query rewrite OK? → **Có** (câu hỏi rõ ràng, không cần rewrite)
  - → **Fix ở bước Retrieval (R):** chunking BCTC bị cắt sai chỗ khiến bảng số liệu nằm ở parent chunk nhưng không được retrieve
- **Root cause:** Bảng số liệu tài chính (dạng tabular data) trong BCTC.md bị hierarchical chunker tách ra khỏi header bảng. Child chunk chứa số "3.703.685.610" bị embed riêng không có context "doanh thu" → cosine similarity thấp với query.
- **Suggested fix:** Dùng `chunk_structure_aware()` cho BCTC thay vì hierarchical — giữ nguyên header + row của bảng cùng một chunk. Hoặc thêm metadata `table_row: true` để không split giữa các dòng.

---

### #2
- **Question:** Chuyển dữ liệu cá nhân ra nước ngoài là gì?
- **Expected:** Là hoạt động sử dụng không gian mạng, thiết bị, phương tiện điện tử chuyển dữ liệu cá nhân của công dân Việt Nam ra ngoài lãnh thổ Việt Nam, hoặc dùng địa điểm ngoài Việt Nam để xử lý dữ liệu cá nhân của công dân Việt Nam.
- **Got:** Chuyển dữ liệu cá nhân ra nước ngoài là việc di chuyển thông tin cá nhân vượt ra ngoài biên giới quốc gia.
- **Worst metric:** faithfulness = 0.48
- **Error Tree:**
  - Output đúng? → **Không** (thiếu chi tiết pháp lý: "không gian mạng", "công dân Việt Nam")
  - Context đúng? → **Có** (context chứa đúng Điều 26 Nghị định 13/2023)
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Generation (G):** LLM paraphrase mất đi ngôn ngữ pháp lý chính xác
- **Root cause:** LLM tóm tắt quá mức (over-summarize) định nghĩa pháp lý. Prompt không yêu cầu giữ nguyên thuật ngữ chuyên ngành — LLM đơn giản hóa "không gian mạng, thiết bị điện tử" thành "biên giới quốc gia" (sai nghĩa pháp lý).
- **Suggested fix:** Thêm vào system prompt: "Với câu hỏi định nghĩa pháp lý, trích dẫn nguyên văn thay vì paraphrase." Hoặc dùng extraction prompt thay vì summarization prompt cho domain Legal.

---

### #3
- **Question:** Xử lý dữ liệu cá nhân bao gồm những hoạt động nào?
- **Expected:** Một hoặc nhiều hoạt động tác động tới dữ liệu cá nhân như: thu thập, ghi, phân tích, xác nhận, lưu trữ, chỉnh sửa, công khai, kết hợp, truy cập, mã hóa, sao chép, chia sẻ, truyền đưa, chuyển giao, xóa, hủy.
- **Got:** Xử lý dữ liệu cá nhân bao gồm thu thập, lưu trữ, sử dụng và xóa dữ liệu.
- **Worst metric:** context_precision = 0.52
- **Error Tree:**
  - Output đúng? → **Không** (thiếu 12/16 hoạt động trong danh sách)
  - Context đúng? → **Một phần** (context có chứa phần đầu của danh sách, nhưng reranker trả về 3 chunks trong đó 2 chunks không liên quan đến định nghĩa xử lý)
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Reranking (R):** reranker đưa 2 chunks về chủ đề "quyền của chủ thể dữ liệu" lên top thay vì chunk chứa danh sách đầy đủ các hoạt động
- **Root cause:** Cross-encoder reranker bị nhầm lẫn giữa "xử lý dữ liệu" (điều 2 Nghị định 13) và "quyền về dữ liệu" (điều 9) vì cùng domain. Context precision thấp → LLM chỉ generate từ context không đầy đủ.
- **Suggested fix:** Tăng `RERANK_TOP_K` từ 3 lên 5 để giữ nhiều context hơn. Thêm metadata filter: nếu query chứa "bao gồm những" → ưu tiên chunks chứa list/enumeration.

---

### #4
- **Question:** Dữ liệu cá nhân nhạy cảm bao gồm những loại nào?
- **Expected:** Quan điểm chính trị, tôn giáo, tình trạng sức khỏe, nguồn gốc chủng tộc, di truyền, sinh học, đời sống tình dục, dữ liệu tội phạm, thông tin tài chính khách hàng, vị trí cá nhân.
- **Got:** Dữ liệu cá nhân nhạy cảm bao gồm thông tin về sức khỏe, tôn giáo và các thông tin có thể gây hại cho cá nhân nếu bị lộ.
- **Worst metric:** context_recall = 0.55
- **Error Tree:**
  - Output đúng? → **Không** (chỉ liệt kê 3/10 loại dữ liệu nhạy cảm)
  - Context đúng? → **Không** (hybrid search trả về chunk nói về "dữ liệu cơ bản" thay vì chunk về "dữ liệu nhạy cảm" — 2 section liền kề trong Nghị định 13)
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Retrieval (R):** BM25 tokenize "nhạy cảm" tốt nhưng dense search không phân biệt được "nhạy cảm" vs "cơ bản" vì 2 chunks ngữ nghĩa gần nhau
- **Root cause:** Hierarchical chunking tách Điều 2 khoản 3 (dữ liệu nhạy cảm) thành nhiều child chunk. Child chunk đầu tiên chứa header + 3 loại đầu → được retrieve. Các loại còn lại (khoản 3d-j) nằm ở child chunk sau → bị miss.
- **Suggested fix:** Với văn bản pháp lý có khoản/điểm, dùng `chunk_structure_aware()` split theo markdown header (`##`, `###`) thay vì hierarchical split theo độ dài. Đảm bảo mỗi "điều" là một chunk hoàn chỉnh.

---

### #5
- **Question:** Tờ khai thuế GTGT của DHA Surfaces áp dụng cho kỳ tính thuế nào?
- **Expected:** Quý 4 năm 2024.
- **Got:** Tờ khai thuế GTGT của DHA Surfaces được áp dụng cho các kỳ tính thuế theo quy định của pháp luật Việt Nam.
- **Worst metric:** answer_relevancy = 0.38
- **Error Tree:**
  - Output đúng? → **Không** (câu trả lời quá chung chung, không có thông tin cụ thể)
  - Context đúng? → **Không** (search không tìm được đúng chunk chứa "Quý 4 năm 2024" trong BCTC)
  - Query rewrite OK? → **Không** — "Tờ khai thuế GTGT" → BM25 tokenize tốt nhưng dense embedding nhầm sang chunks về quy trình nộp thuế trong IT policy
  - → **Fix ở bước Pre-RAG (P):** query disambiguation — "DHA Surfaces" là proper noun cần metadata filter
- **Root cause:** Dense search embed "tờ khai thuế GTGT" gần với context "IT policy" vì cả 2 đều có từ "kê khai", "hệ thống". Thiếu entity-level filter: DHA Surfaces + thuế GTGT → chỉ tìm trong BCTC.md.
- **Suggested fix:** Thêm metadata filter theo `source_document` trong search query. Dùng NER để detect "DHA Surfaces" → restrict search về BCTC corpus. Enrichment (M5) với `extract_metadata()` đã có `entities` field — cần dùng field này trong hybrid search filter.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Chuyển dữ liệu cá nhân ra nước ngoài là gì?"* (Failure #2)

**Error Tree walkthrough:**
1. **Output đúng?** → **Không** — LLM trả lời thiếu chi tiết pháp lý ("không gian mạng", "công dân Việt Nam"), dùng ngôn ngữ đời thường thay vì ngôn ngữ pháp lý
2. **Context đúng?** → **Có** — hybrid search + reranker trả về đúng Điều 26 Nghị định 13/2023, context chứa đầy đủ định nghĩa
3. **Query rewrite OK?** → **Có** — câu hỏi rõ ràng, không cần rewrite
4. **Fix ở bước:** **Generation (G)** — lỗi nằm ở LLM prompt, không phải retrieval

**Root cause:** Faithfulness thấp (0.48) vì LLM over-summarize định nghĩa pháp lý. Đây là điển hình của *hallucination by simplification* — LLM không bịa ra thông tin mới nhưng lại mất thông tin quan trọng trong quá trình paraphrase.

**Fix đề xuất:** Thêm instruction vào system prompt: *"Với câu hỏi về định nghĩa pháp lý hoặc quy định, hãy trích dẫn nguyên văn từ context thay vì paraphrase."* Điều này tăng Faithfulness mà không ảnh hưởng Answer Relevancy.

---

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Structure-aware chunking cho văn bản pháp lý** — Nghị định 13 và BCTC đều có cấu trúc rõ (điều/khoản/điểm, bảng số liệu). Dùng `chunk_structure_aware()` thay vì hierarchical sẽ giải quyết failures #1, #3, #4.
2. **Metadata-based filtering** — Dùng `entities` từ M5 enrichment để restrict search domain (BCTC vs HR Policy vs IT Policy vs Nghị định), giải quyết failure #5.
3. **Legal-domain prompt** — Thêm domain-specific instruction cho câu hỏi pháp lý/tài chính: ưu tiên trích dẫn nguyên văn, giữ số liệu chính xác.
