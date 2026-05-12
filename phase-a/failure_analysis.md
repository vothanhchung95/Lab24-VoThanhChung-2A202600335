# Phase A: Failure Analysis — 50-Question Synthetic Test Set

## RAGAS Aggregate Scores

| Metric | Score | Target | Pass? |
|--------|-------|--------|-------|
| Faithfulness | 0.6711 | ≥0.70 | ✗ |
| Answer Relevancy | 0.4118 | ≥0.65 | ✗ |
| Context Precision | 0.9022 | ≥0.75 | ✓ |
| Context Recall | 0.9055 | ≥0.75 | ✓ |

## Bottom-10 Failures

| Rank | ID | Domain | Type | Avg Score | Worst Metric | Worst Score |
|------|-----|--------|------|-----------|--------------|-------------|
| 1 | 9 | Legal | multi_context | 0.5546 | Faithfulness | 0.0820 |
| 2 | 21 | Finance | simple | 0.6074 | Answer Relevancy | 0.2444 |
| 3 | 28 | Legal | simple | 0.6078 | Faithfulness | 0.1186 |
| 4 | 37 | Finance | reasoning | 0.6475 | Answer Relevancy | 0.3180 |
| 5 | 29 | Legal | simple | 0.6591 | Faithfulness | 0.3809 |
| 6 | 39 | Legal | reasoning | 0.6649 | Faithfulness | 0.3074 |
| 7 | 44 | HR | multi_context | 0.6748 | Answer Relevancy | 0.4655 |
| 8 | 19 | Finance | simple | 0.6783 | Answer Relevancy | 0.2065 |
| 9 | 30 | Legal | simple | 0.6836 | Faithfulness | 0.4108 |
| 10 | 31 | HR | reasoning | 0.6862 | Answer Relevancy | 0.2806 |

## Error Cluster 1: Faithfulness Collapse on Legal Enumeration Questions

**Pattern:** Questions asking to list multiple items (Nghị định 13 articles listing data types,
data subject rights, processing activities) score faithfulness near 0. The hierarchical chunker
(parent 2048 / child 256 chars) splits long lists across multiple child chunks. With RERANK_TOP_K=3,
only 1-2 list fragments are retrieved. GPT-4o-mini generates partial list + generic filler →
RAGAS faithfulness judge finds unsupported claims.

**Affected questions:** IDs 9, 28, 29, 39, 30 (domain=Legal), type: simple/multi_context/reasoning

**Root cause:** `HIERARCHICAL_CHILD_SIZE=256` cuts Điều 2 khoản 3/4 (10+ item lists) into multiple
child chunks. Top-k=3 retrieval misses items 4-10. LLM hallucinates "thông tin khác" to fill gaps.

**Technical fix:**
1. Add `chunk_structure_aware()` for Nghị định 13 — split by `###`/`####` markdown headers,
   keep full article as one chunk (prevents list splitting)
2. Increase `RERANK_TOP_K` from 3 → 5 for list-type questions (detected by keyword matching:
   "bao gồm", "những loại", "hoạt động nào")
3. System prompt: add "Liệt kê đầy đủ tất cả các mục, không tổng quát hóa"

**Expected improvement:** faithfulness +0.15 → from ~0.67 to ~0.82 on Legal domain

## Error Cluster 2: Answer Relevancy Drop on Financial/Specific-Number Questions

**Pattern:** Questions about specific numbers (VAT amounts, revenue figures from BCTC) score
low answer_relevancy. The BCTC document contains many empty form fields ([01a], [02], ...).
Retriever returns noisy chunks with brackets and codes. LLM generates vague answers avoiding
specific numbers ("theo tài liệu" hedging).

**Affected questions:** IDs 21, 37, 19 (domain=Finance)

**Root cause:** BCTC.md contains tabular form data with many empty `[]` brackets. BM25 matches
field labels but chunks contain mostly empty form fields. Answer hedges with vague references
instead of extracting numeric values.

**Technical fix:**
1. Pre-process BCTC.md: strip empty `[]` form fields, keep only rows with actual values
2. Add metadata filter: restrict Finance queries to `source=="bctc.md"` document only
3. System prompt: "Nếu context có số liệu cụ thể (số tiền, ngày tháng, phần trăm) PHẢI trích
   xuất chính xác số đó, không được dùng cụm 'theo tài liệu'"

**Expected improvement:** answer_relevancy +0.12 → from ~0.41 to ~0.53 on Finance domain

## Recommendations for Next Sprint

1. **Immediate (1 day):** Structure-aware chunking for Nghị định 13 — fixes Cluster 1
2. **Short-term (1 week):** BCTC pre-processing + metadata filter — fixes Cluster 2
3. **Medium-term:** Domain-specific prompts (Legal: quote verbatim; Finance: extract numbers)
4. **Monitoring:** Add per-domain RAGAS tracking to CI/CD gate to catch regressions early
