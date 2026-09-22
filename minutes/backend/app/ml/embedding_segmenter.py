"""Embedding-based agenda segmentation. TextTiling-style boundary detection over
multilingual sentence embeddings.

Implements the contract described in the module's original docstring (see
`docs/architecture.md`'s "ML seam" section and README's "For Person A"
section): `EmbeddingAgendaSegmenter.segment()` returns contiguous
`AgendaBlockResult`s with a `confidence` that reflects how sharp the detected
topic boundary actually was, not a flat constant.

Algorithm
---------
1. Split the transcript into sentences, keeping character offsets so blocks can
   be traced back to their source span (`start_char`/`end_char`).
2. Embed every sentence with a multilingual `sentence-transformers` model
   (`paraphrase-multilingual-MiniLM-L12-v2` by default -- trained on 50+
   languages including Hindi, Marathi, Bengali and Gujarati, so Devanagari and
   code-mixed text are not out-of-distribution the way they would be for an
   English-only model).
3. Compute cosine similarity between every adjacent sentence pair (embeddings
   are L2-normalised, so a dot product IS the cosine similarity).
4. A boundary is placed where similarity dips into a local valley at least
   `std_multiplier` standard deviations below the transcript's mean similarity
   -- the same principle as classic TextTiling, just with embedding similarity
   in place of word overlap.
5. `min_sentences`/`max_sentences` keep blocks a reviewable size: a boundary
   too soon after the last one is ignored, and a block that runs long without
   a real boundary is force-cut (and honestly marked low-confidence, since that
   cut was not evidence of a topic shift).
6. Confidence is the *depth* of the similarity valley relative to its
   neighbours, passed through a saturating curve -- a sharp drop scores near
   0.95, a shallow one near 0.15. A flat transcript with no real boundaries
   still returns exactly one block spanning the whole text (never zero -- the
   platform treats zero blocks as a failed run).

Known limitations (see `docs/ml-evaluation.md`)
------------------------------------------------
* This is sentence-level TextTiling, not learned segmentation -- it has no
  notion of *what* a topic is, only where the embedding signal moves. A
  meeting with genuinely gradual topic drift (no participant ever says "next
  item") may under-segment.
* Boundary-level evaluation (precision/recall against annotated boundaries) is
  in `backend/app/ml/evaluation/segmentation_metrics.py`; only the corpus in
  `fixtures/meetings/` has been checked against it, and it is small and
  English-dominant (see the dataset card's own limitations section).

Sentence splitting (2026-09-22)
--------------------------------
Sentence boundaries come from `multilingual_cues.split_into_sentences()`, not
a local regex -- see that function's docstring for the full tiered strategy
(real STT segment boundaries, then punctuation, then a comma-based fallback
for punctuation-free ASR output like real Hindi transcripts). This module
does not decide when to fall back; it only optionally supplies `segments`
(accepted since this class's Protocol signature was written, but unused
until now -- no caller in this codebase currently passes them, so this stays
inert unless a caller starts to).
"""

from __future__ import annotations

import os
import statistics
from typing import Any

from backend.app.ml.base import AgendaBlockResult, MLServiceError, Segment, SegmentationResult
from backend.app.ml.multilingual_cues import Span, split_into_sentences
from backend.app.ml.sentence_embeddings import encode as embed_sentences

DEFAULT_MODEL = os.getenv("MOM_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_MIN_SENTENCES = int(os.getenv("MOM_SEGMENT_MIN_SENTENCES", "2"))
DEFAULT_MAX_SENTENCES = int(os.getenv("MOM_SEGMENT_MAX_SENTENCES", "10"))
DEFAULT_STD_MULTIPLIER = float(os.getenv("MOM_SEGMENT_STD_MULTIPLIER", "1.0"))

_Sentence = Span  # (text, start_char, end_char)


class EmbeddingAgendaSegmenter:
    """Topic segmentation by embedding similarity valleys."""

    name = "embedding-texttiling-v1"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        min_sentences: int = DEFAULT_MIN_SENTENCES,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        std_multiplier: float = DEFAULT_STD_MULTIPLIER,
    ) -> None:
        self.model_id = model_id
        self.min_sentences = max(1, min_sentences)
        self.max_sentences = max(self.min_sentences, max_sentences)
        self.std_multiplier = std_multiplier

    def segment(self, text: str, segments: list[Segment] | None = None) -> SegmentationResult:
        if not text or not text.strip():
            return SegmentationResult(
                blocks=[self._empty_block(text)], model_name=self._model_name()
            )

        sentences = split_into_sentences(text, segments)
        if len(sentences) <= 1:
            return SegmentationResult(
                blocks=[self._whole_text_block(sentences, text)], model_name=self._model_name()
            )

        try:
            embeddings = embed_sentences(self.model_id, [s[0] for s in sentences])
        except ImportError as exc:
            raise MLServiceError(
                "sentence-transformers is not installed; install requirements-ml.txt "
                "or set MOM_SEGMENTER_BACKEND=baseline"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive, model-library-specific
            raise MLServiceError(f"embedding segmentation failed: {exc}") from exc

        sims = [float(embeddings[i] @ embeddings[i + 1]) for i in range(len(sentences) - 1)]
        mean = statistics.fmean(sims)
        std = statistics.pstdev(sims) if len(sims) > 1 else 0.0
        threshold = mean - self.std_multiplier * std

        # sentence index -> confidence; a boundary falls AFTER this index.
        boundaries: dict[int, float] = {}
        since_last = 1
        for i, sim in enumerate(sims):
            since_last += 1
            is_real_boundary = sim < threshold and since_last > self.min_sentences
            is_forced_cut = since_last > self.max_sentences
            if is_real_boundary:
                boundaries[i] = self._confidence(sims, i)
                since_last = 0
            elif is_forced_cut:
                boundaries[i] = 0.3  # length cap, not a detected topic shift
                since_last = 0

        blocks: list[AgendaBlockResult] = []
        start_idx = 0
        for i in range(len(sentences)):
            is_last = i == len(sentences) - 1
            if i in boundaries or is_last:
                block_sentences = sentences[start_idx : i + 1]
                block_embeddings = embeddings[start_idx : i + 1]
                confidence = boundaries.get(i, 0.5)  # end-of-transcript close, not a detected cut
                blocks.append(
                    self._make_block(len(blocks), block_sentences, block_embeddings, confidence)
                )
                start_idx = i + 1

        return SegmentationResult(blocks=blocks, model_name=self._model_name())

    # ------------------------------------------------------------------ #

    def _model_name(self) -> str:
        return f"{self.name} ({self.model_id})"

    def _whole_text_block(self, sentences: list[_Sentence], text: str) -> AgendaBlockResult:
        if not sentences:
            return self._empty_block(text)
        sentence, start, end = sentences[0]
        return AgendaBlockResult(
            position=0,
            title=self._title_from_words(sentence),
            text=text[start:end] or sentence,
            start_char=start,
            end_char=end,
            confidence=0.5,
        )

    def _make_block(
        self,
        position: int,
        block_sentences: list[_Sentence],
        block_embeddings: Any,
        confidence: float,
    ) -> AgendaBlockResult:
        text = " ".join(s for s, _, _ in block_sentences)
        start_char = block_sentences[0][1]
        end_char = block_sentences[-1][2]
        title = self._representative_title(block_sentences, block_embeddings)
        return AgendaBlockResult(
            position=position,
            title=title,
            text=text,
            start_char=start_char,
            end_char=end_char,
            confidence=confidence,
        )

    def _representative_title(self, block_sentences: list[_Sentence], block_embeddings: Any) -> str:
        """The sentence closest to the block's centroid, truncated -- more
        representative of a multi-sentence block than always taking the first
        sentence, which classic TextTiling / the cue-phrase baseline both do."""
        if len(block_sentences) == 1:
            rep = block_sentences[0][0]
        else:
            centroid = block_embeddings.mean(axis=0)
            scores = block_embeddings @ centroid
            rep = block_sentences[int(scores.argmax())][0]
        return self._title_from_words(rep)

    @staticmethod
    def _title_from_words(sentence: str) -> str:
        title = " ".join(sentence.split()[:8])
        return (title[:1].upper() + title[1:]) if title else "Agenda item"

    @staticmethod
    def _empty_block(text: str) -> AgendaBlockResult:
        return AgendaBlockResult(
            position=0,
            title="Empty transcript",
            text=text or "",
            start_char=0,
            end_char=len(text or ""),
            confidence=0.0,
        )

    @staticmethod
    def _confidence(sims: list[float], i: int) -> float:
        """Valley depth relative to neighbours, through a saturating curve."""
        left = sims[i - 1] if i - 1 >= 0 else sims[i]
        right = sims[i + 1] if i + 1 < len(sims) else sims[i]
        neighbourhood = (left + right) / 2
        depth = max(0.0, neighbourhood - sims[i])
        confidence = depth / (depth + 0.15)
        return round(min(0.95, max(0.15, confidence)), 3)
