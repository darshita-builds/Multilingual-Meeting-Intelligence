"""Boundary-level evaluation for agenda segmentation.

Not full Pk / WindowDiff (the classic TextTiling metrics) -- those need a
tokenised, position-indexed reference segmentation, which this project's
annotated data does not have (the gold set records decision/action spans, not
a canonical topic segmentation of every scenario). Instead: boundary
precision/recall/F1 with a character-offset tolerance window -- a predicted
boundary counts as correct if it falls within `tolerance_chars` of a reference
boundary. This is a simplification, labelled as one rather than presented as
the standard metric it is not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentationMetrics:
    predicted_boundaries: int
    reference_boundaries: int
    true_positives: int

    @property
    def precision(self) -> float:
        if not self.predicted_boundaries:
            return 0.0
        return round(self.true_positives / self.predicted_boundaries, 4)

    @property
    def recall(self) -> float:
        if not self.reference_boundaries:
            return 0.0
        return round(self.true_positives / self.reference_boundaries, 4)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "predicted_boundaries": self.predicted_boundaries,
            "reference_boundaries": self.reference_boundaries,
            "true_positives": self.true_positives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def evaluate_boundaries(
    predicted_positions: list[int], reference_positions: list[int], tolerance_chars: int = 40
) -> SegmentationMetrics:
    remaining = list(reference_positions)
    true_positives = 0
    for position in predicted_positions:
        if not remaining:
            break
        closest = min(remaining, key=lambda r: abs(r - position))
        if abs(closest - position) <= tolerance_chars:
            true_positives += 1
            remaining.remove(closest)
    return SegmentationMetrics(
        predicted_boundaries=len(predicted_positions),
        reference_boundaries=len(reference_positions),
        true_positives=true_positives,
    )
