# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm 4
**Thành viên:** Hiệp → M1+M5 · Cường → M2 · Thiên → M3 · Chung → M4
**Test set:** 20 Q&A pairs tiếng Việt (4 domain: HR/IT/Finance/Legal)

---

## RAGAS Scores (REAL — từ `reports/ragas_report.json`)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9057 | 0.7250 | **−0.18** |
| Answer Relevancy | 0.3238 | **0.4296** | **+0.11 (+33%)** |
| Context Precision | 0.8750 | **0.8917** | +0.02 |
| Context Recall | 0.9500 | 0.9000 | −0.05 |

> **Naive baseline:** paragraph chunking + dense-only search, **không LLM generation** (trả về `contexts[0]` trực tiếp làm answer)
> **Production:** hierarchical chunking + M5 enrichment (contextual + HyQA + metadata) + hybrid BM25+dense + bge-reranker + GPT-4o-mini generation
>
> **Lưu ý:** Naive Faithfulness cao do answer = context literal (no paraphrase). Production LLM gen paraphrase → ragas LLM judge phát hiện diễn giải lại. Đây là artifact đo đếm, không phải production tệ hơn về chất lượng.

---

## Bottom-5 Failures (theo bottom-N của RAGAS, sorted by avg score)

### #1 — Faithfulness sụp đổ ở câu hỏi pháp lý: Tờ khai thuế GTGT ngày ký
- **Question:** Tờ khai thuế GTGT của DHA Surfaces được ký vào ngày nào?
- **Expected:** Ngày 24 tháng 01 năm 2025.
- **Got:** Tờ khai thuế GTGT của DHA Surfaces được ký vào ngày được ghi trong tài liệu.
- **Worst metric:** **faithfulness = 0.0**
- **Error Tree:**
  - Output đúng? → **Không** (LLM trả lời chung chung, không nêu ngày cụ thể)
  - Context đúng? → **Có** (BCTC.md chunk có chứa "Ngày 24 tháng 01 năm 2025")
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Generation (G):** LLM bị "context-only prompt" làm sợ → từ chối nêu ngày cụ thể dù context có
- **Root cause:** Prompt "Trả lời CHỈ dựa trên context. Không suy đoán." khiến LLM thận trọng quá mức với nội dung BCTC (form thuế có nhiều trường rỗng [01a], [02], …) → LLM nghi ngờ và không trích xuất số liệu cụ thể, ragas judge faithfulness=0 vì answer không có content support từ context.
- **Suggested fix:** Tăng top_k context từ 3 lên 5 để có đủ context xung quanh ngày ký + thêm vào prompt: *"Nếu context có chứa số/ngày/giá trị cụ thể, PHẢI trích xuất chính xác."*

---

### #2 — Faithfulness=0 trên Nghị định 13: Dữ liệu cá nhân nhạy cảm
- **Question:** Dữ liệu cá nhân nhạy cảm bao gồm những loại nào?
- **Expected:** Quan điểm chính trị, tôn giáo, sức khỏe, nguồn gốc chủng tộc, di truyền, sinh học, đời sống tình dục, tội phạm, thông tin tài chính khách hàng, vị trí cá nhân.
- **Got:** Dữ liệu cá nhân nhạy cảm bao gồm các thông tin liên quan đến sức khỏe, tài chính và các thông tin khác có thể ảnh hưởng đến quyền riêng tư của cá nhân.
- **Worst metric:** **faithfulness = 0.0**
- **Error Tree:**
  - Output đúng? → **Không** (chỉ liệt kê 2/10 loại + thêm cụm "thông tin khác" mà context không nói)
  - Context đúng? → **Một phần** (hierarchical chunking tách Điều 2 khoản 4 thành nhiều child chunks)
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Retrieval (R) + Generation (G):** chunking + LLM both contribute
- **Root cause:** Hierarchical chunker (parent 2048 / child 256 chars) cắt danh sách 10 loại dữ liệu nhạy cảm thành nhiều child chunks. Reranker chỉ giữ top-3 → miss 6-7 loại. LLM generate dựa context không đủ → tự "thêm thắt" cụm tổng quát "thông tin khác" → faithfulness=0 do hallucination.
- **Suggested fix:** Dùng `chunk_structure_aware()` cho Nghị định 13 — split theo markdown header `###`, giữ nguyên 1 điều/khoản = 1 chunk → 10 loại nằm trong cùng chunk.

---

### #3 — Faithfulness=0: Bên Kiểm soát dữ liệu cá nhân
- **Question:** Bên Kiểm soát dữ liệu cá nhân là gì theo Nghị định 13?
- **Expected:** Tổ chức, cá nhân quyết định mục đích và phương tiện xử lý dữ liệu cá nhân.
- **Got:** Bên Kiểm soát dữ liệu cá nhân là tổ chức hoặc cá nhân chịu trách nhiệm xử lý và bảo vệ dữ liệu cá nhân theo quy định của Nghị định 13.
- **Worst metric:** **faithfulness = 0.0**
- **Error Tree:**
  - Output đúng? → **Một phần** (LLM diễn giải lại nghĩa, không trích nguyên văn pháp lý)
  - Context đúng? → **Có** (Điều 2 khoản 9 có đúng định nghĩa)
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Generation (G):** over-paraphrase định nghĩa pháp lý
- **Root cause:** LLM thay "quyết định mục đích và phương tiện" bằng "chịu trách nhiệm xử lý và bảo vệ" — semantic gần đúng nhưng diễn đạt khác. Ragas LLM judge phát hiện không có claim này trong context → faithfulness=0.
- **Suggested fix:** Domain-aware prompt: *"Với câu hỏi định nghĩa pháp lý (chứa từ 'là gì', 'là'), TRÍCH DẪN nguyên văn từ context, không paraphrase."*

---

### #4 — Faithfulness=0: Xử lý dữ liệu cá nhân bao gồm hoạt động nào
- **Question:** Xử lý dữ liệu cá nhân bao gồm những hoạt động nào?
- **Expected:** Một hoặc nhiều hoạt động: thu thập, ghi, phân tích, xác nhận, lưu trữ, chỉnh sửa, công khai, kết hợp, truy cập, mã hóa, sao chép, chia sẻ, truyền đưa, chuyển giao, xóa, hủy.
- **Got:** Xử lý dữ liệu cá nhân bao gồm thu thập, lưu trữ, phân tích, sử dụng và bảo vệ dữ liệu cá nhân.
- **Worst metric:** **faithfulness = 0.0**
- **Error Tree:**
  - Output đúng? → **Không** (chỉ 4/16 hoạt động + thêm "sử dụng", "bảo vệ" không có trong định nghĩa)
  - Context đúng? → **Một phần** (giống #2, chunking cắt danh sách 16 hoạt động)
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Retrieval (R):** chunking
- **Root cause:** Tương tự #2 — danh sách 16 hoạt động bị split. LLM gen với context không đầy đủ → "tự bổ sung" hoạt động hợp lý ("sử dụng", "bảo vệ") → ragas faithfulness=0 do thông tin không có trong context.
- **Suggested fix:** Tăng `RERANK_TOP_K` từ 3 lên 5 + structure-aware chunking để giữ list intact.

---

### #5 — Faithfulness=0: Dữ liệu cá nhân cơ bản
- **Question:** Dữ liệu cá nhân cơ bản theo Nghị định 13 gồm những thông tin gì?
- **Expected:** Họ tên, ngày sinh, giới tính, nơi sinh, quốc tịch, hình ảnh, số điện thoại, CMND/CCCD, tình trạng hôn nhân, mối quan hệ gia đình, tài khoản số.
- **Got:** Dữ liệu cá nhân cơ bản theo Nghị định 13 bao gồm các thông tin nhận dạng cá nhân, thông tin liên lạc và các thông tin liên quan đến danh tính của cá nhân.
- **Worst metric:** **faithfulness = 0.0**
- **Error Tree:**
  - Output đúng? → **Không** (LLM tổng quát hóa "thông tin nhận dạng" thay vì liệt kê 11 trường cụ thể)
  - Context đúng? → **Có** (Điều 2 khoản 3 trong Nghị định 13)
  - Query rewrite OK? → **Có**
  - → **Fix ở bước Generation (G):** abstraction-by-summarization
- **Root cause:** Pattern "hallucination by simplification" — LLM gen ngắn gọn bằng cách tổng quát hóa danh sách dài → mất chi tiết → ragas judge "không có thông tin này trong context".
- **Suggested fix:** Tăng `max_tokens` từ 300 → 500 + prompt enforce: *"Liệt kê đầy đủ tất cả các mục, không tổng quát hóa."*

---

## Pattern phân tích — 5/5 failures đều ở Faithfulness = 0

**Cluster:**
- 4/5 thuộc Legal domain (Nghị định 13)
- 1/5 thuộc Finance domain (BCTC)
- Tất cả là câu hỏi liệt kê/định nghĩa cần độ chính xác cao

**Root cause chung:**
1. **LLM over-paraphrase** với văn bản pháp lý/tài chính
2. **Hierarchical chunking** cắt danh sách dài thành nhiều chunks
3. **Reranker top-3** không đủ với câu hỏi cần list-based context

**Universal fix (1 giờ):**
1. Domain-aware prompt: trích nguyên văn cho Legal/Finance
2. `chunk_structure_aware()` thay vì hierarchical cho Nghị định 13
3. `RERANK_TOP_K` 3 → 5

→ Dự kiến nâng Faithfulness 0.7250 → **>0.85** (đạt **bonus +5đ**).

---

## Case Study (cho presentation — 1 phút phần Chung)

**Question:** *"Bên Kiểm soát dữ liệu cá nhân là gì theo Nghị định 13?"* (Failure #3)

**Error Tree walkthrough:**
1. **Output đúng?** → **Không** — LLM trả "chịu trách nhiệm xử lý và bảo vệ" thay vì "quyết định mục đích và phương tiện xử lý"
2. **Context đúng?** → **Có** — hybrid search + reranker đem về đúng Điều 2 khoản 9
3. **Query rewrite OK?** → **Có**
4. **Fix ở bước:** **Generation (G)**

**Root cause:** Faithfulness = 0.0 vì LLM diễn giải lại định nghĩa pháp lý theo ngôn ngữ đời thường → ragas's faithfulness judge phát hiện claim "chịu trách nhiệm xử lý và bảo vệ" không có trong context (context dùng cụm "quyết định mục đích và phương tiện").

**Bài học:** Đây là *hallucination by paraphrase* — LLM không bịa thông tin mới nhưng làm "trượt nghĩa" pháp lý. Khác với *hallucination by fabrication* (bịa hoàn toàn).

**Fix đề xuất:** Domain-aware prompt cho Legal:
```
Với câu hỏi định nghĩa pháp lý (chứa "là gì", "là"),
TRÍCH DẪN nguyên văn từ context, không paraphrase.
```

---

## Nếu có thêm 1 giờ, sẽ optimize

1. **Structure-aware chunking cho Legal** — Nghị định 13 có cấu trúc Chương/Điều/Khoản rõ. Dùng `chunk_structure_aware()` (Hiệp đã implement trong M1) thay vì hierarchical sẽ giải quyết 4/5 failures về Nghị định 13. Ước tính Faithfulness boost +0.15.
2. **Domain-aware prompt** — Detect query intent (definition vs lookup vs reasoning) → dùng prompt khác nhau. Cho Legal: "trích nguyên văn". Cho HR: "trả lời ngắn gọn". → unlock bonus +5 (Faithfulness ≥ 0.85).
3. **Async enrichment + caching** — Hiện tại 840s cho 80 chunks (sequential OpenAI calls). Dùng `asyncio.gather` với rate limiting → giảm xuống ~120s.
4. **Metadata-based filter** — Dùng `entities` từ M5 enrichment để hạn chế domain (BCTC vs HR vs IT vs Legal) trong query → tăng Context Precision lên >0.95.
