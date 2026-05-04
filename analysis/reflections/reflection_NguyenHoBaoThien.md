# Individual Reflection — Lab 18

**Tên:** Nguyễn Hồ Bảo Thiên
**Module phụ trách:** M3

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M3 (Reranking)
- Các hàm/class chính đã viết: CrossEncoderReranker class, RerankResult dataclass, rerank method
- Số tests pass: 5/5

## 2. Kiến thức học được

- Khái niệm mới nhất: Cross-encoder reranking và latency benchmarking
- Điều bất ngờ nhất: Cross-encoder cải thiện độ chính xác đáng kể so với BM25 nhưng tốn nhiều thời gian xử lý hơn
- Kết nối với bài giảng (slide nào): Slide về RAG pipeline, phần reranking stage để tối ưu kết quả retrieval

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Tối ưu latency để đạt yêu cầu dưới 5 giây trên CPU
- Cách giải quyết: Chuyển từ model cross-encoder mặc định sang phiên bản MiniLM-L-6-v2 nhẹ hơn
- Thời gian debug: Khoảng hơn 1 giờ để test và benchmark các model khác nhau

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Thử nghiệm FlagReranker thay vì CrossEncoder để so sánh hiệu suất
- Module nào muốn thử tiếp: M5 (Enrichment) vì quan tâm đến việc làm giàu dữ liệu

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |
