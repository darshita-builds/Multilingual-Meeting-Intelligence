"""Word Error Rate (WER) -- the standard STT accuracy metric.

    WER = (substitutions + deletions + insertions) / reference_word_count

Computed via word-level Levenshtein alignment, implemented from scratch (no
`jiwer` dependency) so this module runs with zero extra installs -- WER on a
hand-written pair of strings needs no ML library at all.

On real vs. synthetic data
---------------------------
This module computes WER correctly for ANY (reference, hypothesis) pair; it
does not know or care where the hypothesis came from. What it cannot supply is
real ASR output: `fixtures/meetings/` has no genuine speech (the `.wav` files
are a 2-second placeholder tone -- see `DATASET.md`'s "Known limitations"),
so there is no real WhisperSTT transcription to score yet. Until real
recordings exist, `run_evaluation.py` reports WER against a small,
**explicitly synthetic** set of hand-written (reference, simulated-hypothesis)
pairs in `evaluation/data/synthetic_wer_pairs.json`, built to exercise
realistic ASR error patterns (dropped filler words, homophone substitutions,
code-mixed script confusion) -- see that file's own `"provenance"` field.
Treat any WER number this module reports on that set as a demonstration that
the metric works, not as a measurement of WhisperSTT's real-world accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_wer(text: str) -> str:
    """Lowercase and strip punctuation -- conventional WER pre-processing, so
    'Friday.' and 'friday' are not scored as a substitution. Applied by
    `run_evaluation.py` before scoring; `word_error_rate` itself is pure and
    does not normalise for you, so it can also be used to score
    already-tokenised or intentionally case-sensitive input."""
    return _PUNCT.sub("", text.lower()).strip()


@dataclass
class WERResult:
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float

    def as_dict(self) -> dict:
        return {
            "reference_words": self.reference_words,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "wer": self.wer,
        }


def word_error_rate(reference: str, hypothesis: str) -> WERResult:
    """Word-level edit distance between `reference` and `hypothesis`.

    Tokenisation is whitespace splitting, which is adequate for English and
    for Devanagari (words are space-separated in both). It will over-count
    edits for a script with no word-separating whitespace; none of the
    languages this project targets (English, Hindi, Marathi, Bengali,
    Gujarati) have that property, so no segmenter is needed here.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    n, m = len(ref), len(hyp)

    if n == 0:
        return WERResult(
            reference_words=0, substitutions=0, deletions=0, insertions=m, wer=(1.0 if m else 0.0)
        )

    # dp[i][j] = edit distance between ref[:i] and hyp[:j]; op[i][j] records
    # which operation achieved that minimum, for backtracking below.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "eq"
                continue
            substitute = dp[i - 1][j - 1] + 1
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            best = min(substitute, delete, insert)
            dp[i][j] = best
            op[i][j] = "sub" if best == substitute else ("del" if best == delete else "ins")

    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and op[i][j] == "eq":
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and op[i][j] == "sub":
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or op[i][j] == "del"):
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    wer = (subs + dels + ins) / n
    return WERResult(
        reference_words=n, substitutions=subs, deletions=dels, insertions=ins, wer=round(wer, 4)
    )


def corpus_word_error_rate(pairs: list[tuple[str, str]]) -> WERResult:
    """Micro-averaged WER over several (reference, hypothesis) pairs -- total
    edits over total reference words, the standard corpus-level convention
    (not a mean of per-utterance WERs, which over-weights short utterances)."""
    total = WERResult(reference_words=0, substitutions=0, deletions=0, insertions=0, wer=0.0)
    for reference, hypothesis in pairs:
        r = word_error_rate(reference, hypothesis)
        total.reference_words += r.reference_words
        total.substitutions += r.substitutions
        total.deletions += r.deletions
        total.insertions += r.insertions
    if total.reference_words:
        total.wer = round(
            (total.substitutions + total.deletions + total.insertions) / total.reference_words, 4
        )
    return total
