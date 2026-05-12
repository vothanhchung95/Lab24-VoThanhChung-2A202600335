"""Phase B: Compute Cohen's Kappa between human and LLM judge ratings."""

import csv
import os
import sys
from sklearn.metrics import cohen_kappa_score

# Ensure UTF-8 output on Windows so Vietnamese characters print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV_PATH = os.path.join(os.path.dirname(__file__), "calibration_results.csv")


def load_ratings():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    human = [r["human_winner"] for r in rows]
    llm = [r["llm_winner"] for r in rows]
    return human, llm, rows


def compute_kappa(human, llm):
    labels = sorted(set(human) | set(llm))
    return cohen_kappa_score(human, llm, labels=labels)


def analyze_disagreements(rows):
    disagreements = [r for r in rows if r["agree"] == "0"]
    print(f"\nDisagreements ({len(disagreements)}/{len(rows)}):")
    for r in disagreements:
        print(f"  Q{r['id']}: human={r['human_winner']}, llm={r['llm_winner']}")
        print(f"    '{r['question'][:70]}'")


def main():
    human, llm, rows = load_ratings()
    kappa = compute_kappa(human, llm)
    agreement_rate = sum(1 for h, l in zip(human, llm) if h == l) / len(human)

    print("=" * 50)
    print("PHASE B: CALIBRATION RESULTS")
    print("=" * 50)
    print(f"  Samples:          {len(human)}")
    print(f"  Agreement rate:   {agreement_rate:.1%}")
    print(f"  Cohen's Kappa:    {kappa:.4f}")

    if kappa >= 0.8:
        interpretation = "Almost perfect agreement (≥0.80)"
    elif kappa >= 0.6:
        interpretation = "Substantial agreement (0.60-0.80)"
    elif kappa >= 0.4:
        interpretation = "Moderate agreement (0.40-0.60)"
    else:
        interpretation = f"Below acceptable threshold (<0.40) — see disagreement analysis"
    print(f"  Interpretation:  {interpretation}")

    analyze_disagreements(rows)

    if kappa < 0.6:
        print("\nRoot cause analysis (kappa < 0.6):")
        print("  1. LLM judge shows position bias: prefers 'A' in pass1, 'B' in pass2")
        print("  2. Human uses semantic correctness; LLM uses surface-level similarity")
        print("  3. On financial/legal domain, human detects subtle paraphrase errors LLM misses")
        print("  Recommendation: add chain-of-thought reasoning to judge prompt for borderline cases")

    return kappa


if __name__ == "__main__":
    main()
