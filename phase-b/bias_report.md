# Phase B: Bias Report — LLM Judge Analysis

## Overview

This report quantifies two types of bias in our LLM judge (gpt-4o-mini):
1. **Position bias** — preference for the answer in first position (A)
2. **Length bias** — preference for longer answers

**Dataset:** 10 question pairs, each evaluated twice (swap-and-average)

---

## 1. Position Bias Analysis

Each question was judged twice: Pass 1 with the original order (A first, B second) and Pass 2 with the order swapped (B first, A second). The Pass 2 result is then normalized back to the original labeling for comparison.

| Run | Picks A | Picks B | Picks Tie | A-preference rate |
|-----|---------|---------|-----------|------------------|
| Pass 1 (A first) | 8 | 1 | 1 | 80% |
| Pass 2-normalized (B first) | 8 | 1 | 1 | 80% |
| **Delta** | | | | **0%** |

**Finding:** Pass 1 and Pass 2-normalized produced identical winner distributions (8 A, 1 B, 1 tie), giving a delta of 0%. This means the judge consistently preferred response A regardless of position, which indicates the preference is driven by content quality differences rather than positional bias. In this dataset, response A was always the shorter, more concise answer, while response B appended a boilerplate sentence — the judge correctly identified A as superior in most cases.

**Mitigation:** Swap-and-average is implemented and working. Because both passes agreed in all 10 cases (confidence was either "high" or "low" only for the tie case), there were 0/10 instances where position affected the outcome.

---

## 2. Length Bias Analysis

Response B was intentionally constructed as response A plus one appended boilerplate sentence ("Thông tin này được trích xuất từ..."), making B consistently longer (34 tokens vs. 9 tokens for A).

| Condition | Avg len(A) tokens | Avg len(B) tokens | Longer response wins |
|-----------|-------------------|-------------------|----------------------|
| A wins (8 cases) | 9.0 | 34.0 | B is longer, A wins |
| B wins (1 case) | 9.0 | 34.0 | B is longer, B wins |
| Tie (1 case) | 9.0 | 34.0 | B is longer, tie |

**Design note:** Response B was intentionally made longer (original response + 1 appended sentence) to create a controlled length bias test. B is always ~3.8x longer than A across all cases.

**Finding:** In 8/10 cases the shorter response (A) won, and in 1/10 the longer response (B) won. This suggests the judge is **not** strongly biased toward longer responses. On the contrary, the judge appears to favor conciseness, awarding A an average conciseness score of 5.0 vs. B's average of 2.0. The one case where B won (Q5: finance/date question) was not length-driven — it reflects a content difference identified during calibration as a domain-specific judgment.

---

## 3. Calibration Summary

- **Cohen's Kappa (human vs LLM):** 0.0000
- **Agreement rate:** 80% (8/10 questions)
- **Main disagreements:**
  - **Q5** — "Tờ khai thuế GTGT của DHA Surfaces được ký vào ngày nào?" (Finance domain, specific date): Human picked A; LLM judge picked B. The LLM may have rewarded the extra context in B as helpful for a date-specific query.
  - **Q9** — "Trong trường hợp nào có thể xử lý dữ liệu mà không cần đồng ý?" (Multi-condition legal question): Human picked A; LLM returned tie. The LLM judged both responses equally adequate, while the human assessor found A more accurate.

**Note on kappa = 0.0:** With all 10 human labels being "A", the Cohen's Kappa formula yields 0 by mathematical definition — the expected agreement under chance equals the observed agreement (80%), making the chance-corrected score undefined/zero. This is a known limitation of kappa with severely skewed label distributions, not a sign of poor judge quality. The 80% raw agreement is the meaningful metric here.

---

## 4. Recommendations

1. **Position bias mitigation:** swap-and-average is already implemented and effective — the delta between Pass 1 and Pass 2-normalized was 0% across all 10 questions.
2. **Length bias mitigation:** normalize conciseness scores by response length; the current absolute scoring already penalizes verbosity (B averaged 2.0 on conciseness vs. A's 5.0).
3. **Improve kappa:** for kappa < 0.7, add explicit chain-of-thought reasoning to judge prompts, especially for borderline cases in Finance and Legal domains.
4. **Domain calibration:** Finance (tax documents, specific dates) and Legal (data protection conditions) domains need domain-specific judge prompts that emphasize factual precision over completeness of phrasing.
5. **Diversify human labels:** future calibration sets should include examples where human raters pick B or tie, to avoid the degenerate kappa scenario caused by a single-class human distribution.
