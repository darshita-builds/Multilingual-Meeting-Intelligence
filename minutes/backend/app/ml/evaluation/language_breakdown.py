"""Group span-matching results by language -- the required "language/subgroup
breakdown", computed rather than asserted.

Aggregation is micro-averaged (counts summed before dividing), not a mean of
per-scenario ratios: a language covered by five scenarios should not carry the
same weight as one covered by a single scenario, which per-scenario averaging
would do.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TypedDict


class LanguageRow(TypedDict):
    scenarios: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def aggregate_span_metrics_by_language(per_scenario: list[dict]) -> dict[str, LanguageRow]:
    """`per_scenario` items need `language`, `true_positives`, `false_positives`,
    `false_negatives` keys -- exactly what `MatchResult.as_dict()` produces,
    plus a language tag. Used for both decisions and actions."""
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    for row in per_scenario:
        b = buckets[row["language"]]
        b["tp"] += row["true_positives"]
        b["fp"] += row["false_positives"]
        b["fn"] += row["false_negatives"]
        b["n"] += 1

    out: dict[str, LanguageRow] = {}
    for lang, b in sorted(buckets.items()):
        precision = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else 0.0
        recall = b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[lang] = {
            "scenarios": b["n"],
            "true_positives": b["tp"],
            "false_positives": b["fp"],
            "false_negatives": b["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return out
