"""Self-hosted arithmetic CAPTCHA (threat model: scripted registration spam).

Gated behind `settings.captcha_enabled` (default off, see `config.py`) so
turning it on never requires a third-party key or secret -- there is nothing
to configure beyond the flag itself, matching this project's rule that a
security control must not need a hardcoded credential to work.

Deliberately not cryptographic. The registration allow-list
(`services/approvals.py`) is the actual access control; this exists only to
raise the cost of a scripted `POST /auth/register` loop, the same
narrowly-scoped threat `register_ip_limiter` (`security/ratelimit.py`)
already partly covers. A simple arithmetic question, solved once and
consumed, is enough for that job.

Storage is in-process (same honest limitation as `RateLimiter`, see
`ratelimit.py`'s module docstring): correct for a single-instance deployment,
wrong the moment there are two API replicas. A challenge is single-use --
`verify()` deletes it on first use, whether the answer was right or wrong --
so a captured (id, answer) pair cannot be replayed.
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass

from backend.app.config import settings

__all__ = ["CaptchaChallenge", "CaptchaStore", "captcha_store"]


@dataclass
class CaptchaChallenge:
    id: str
    question: str
    answer: str
    expires_at: float


class CaptchaStore:
    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._ttl = ttl_seconds
        self._challenges: dict[str, CaptchaChallenge] = {}
        self._lock = threading.Lock()

    def issue(self) -> CaptchaChallenge:
        a, b = random.randint(1, 20), random.randint(1, 20)
        op, answer = random.choice([("+", a + b), ("-", a - b)])
        ttl = self._ttl if self._ttl is not None else settings.captcha_ttl_seconds
        challenge = CaptchaChallenge(
            id=str(uuid.uuid4()),
            question=f"What is {a} {op} {b}?",
            answer=str(answer),
            expires_at=time.monotonic() + ttl,
        )
        with self._lock:
            self._sweep()
            self._challenges[challenge.id] = challenge
        return challenge

    def verify(self, challenge_id: str, submitted_answer: str) -> bool:
        """Single-use: the challenge is gone after this call regardless of outcome."""
        with self._lock:
            challenge = self._challenges.pop(challenge_id, None)
        if challenge is None or challenge.expires_at < time.monotonic():
            return False
        return submitted_answer.strip() == challenge.answer

    def clear(self) -> None:
        """Test hook."""
        with self._lock:
            self._challenges.clear()

    def _sweep(self) -> None:
        """Evict expired challenges. Called with the lock already held."""
        now = time.monotonic()
        dead = [cid for cid, c in self._challenges.items() if c.expires_at < now]
        for cid in dead:
            del self._challenges[cid]


captcha_store = CaptchaStore()
