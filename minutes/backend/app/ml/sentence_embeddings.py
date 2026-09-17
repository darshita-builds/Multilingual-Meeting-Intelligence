"""Lazy, process-wide cache for `sentence-transformers` models.

`embedding_segmenter.py` and `transformer_extractor.py` both embed sentences
with the same multilingual model by default. Without a shared cache each would
load its own copy -- two ~470 MB models resident for one logical dependency.
This module is the one place that imports `sentence_transformers`, and it does
so lazily (inside the function, not at module import time) so that importing
this file -- or either module above -- never requires the dependency to be
installed. The registry's `ImportError` fallback only works if that import is
deferred to first use; see `backend/app/ml/registry.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from sentence_transformers import SentenceTransformer

_cache: dict[str, Any] = {}


def get_model(model_id: str) -> SentenceTransformer:
    """Load (once) and return a `SentenceTransformer`.

    Raises `ImportError` if `sentence-transformers` is not installed -- callers
    in this package catch that and re-raise as `MLServiceError`, which is what
    the platform's contract expects from an ML service.
    """
    if model_id in _cache:
        return _cache[model_id]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    _cache[model_id] = model
    return model


def encode(model_id: str, sentences: list[str]) -> np.ndarray:
    """Embed sentences, L2-normalised so a dot product is a cosine similarity."""
    model = get_model(model_id)
    return model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)


def reset_cache() -> None:
    """Test hook."""
    _cache.clear()
