"""WER: correctness of the from-scratch word-level edit distance."""

from __future__ import annotations

from backend.app.ml.evaluation.wer import (
    corpus_word_error_rate,
    normalize_for_wer,
    word_error_rate,
)


def test_identical_strings_have_zero_wer():
    result = word_error_rate("we have decided to proceed", "we have decided to proceed")
    assert result.wer == 0.0
    assert result.substitutions == result.deletions == result.insertions == 0


def test_single_substitution():
    result = word_error_rate("the decision is final", "the division is final")
    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 0
    assert result.wer == 0.25  # 1 error / 4 reference words


def test_single_deletion():
    result = word_error_rate("check the staging backups", "check staging backups")
    assert result.deletions == 1
    assert result.wer == 0.25


def test_single_insertion():
    result = word_error_rate("send the report", "send the the report")
    assert result.insertions == 1
    assert result.wer == round(1 / 3, 4)


def test_empty_reference_with_nonempty_hypothesis_is_all_insertions():
    result = word_error_rate("", "hello world")
    assert result.reference_words == 0
    assert result.insertions == 2
    assert result.wer == 1.0


def test_empty_reference_and_empty_hypothesis_is_zero_wer():
    result = word_error_rate("", "")
    assert result.wer == 0.0


def test_completely_different_hypothesis_scores_high_wer():
    result = word_error_rate("one two three", "four five six seven")
    assert result.wer > 1.0  # more insertions+substitutions than reference words


def test_corpus_wer_is_micro_averaged_not_a_mean_of_ratios():
    """A short utterance with 100% WER should not dominate a long, accurate one."""
    pairs = [
        ("a", "b"),  # 1/1 = 100% WER on a 1-word utterance
        (
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten",
        ),  # 0%
    ]
    corpus = corpus_word_error_rate(pairs)
    assert corpus.reference_words == 11
    assert corpus.wer == round(1 / 11, 4)


def test_normalize_for_wer_lowercases_and_strips_punctuation():
    assert normalize_for_wer("Friday.") == "friday"
    assert normalize_for_wer("It's final!") == "its final"
