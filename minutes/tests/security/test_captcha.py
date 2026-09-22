"""CAPTCHA gate on /auth/register (security/captcha.py).

Off by default (`settings.captcha_enabled = False`), so the first group of
tests here is the regression check that matters most: nothing about
registration changes for a deployment that never turns this on.
"""

from __future__ import annotations

from backend.app.config import settings

PASSWORD = "a-long-enough-password"


def _register(client, email: str, **extra):
    return client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, **extra}
    )


def _solve(client) -> dict:
    challenge = client.get("/api/v1/auth/captcha").json()
    # Questions are always "What is A + B?" or "What is A - B?".
    _, expr = challenge["question"].split("What is ")
    op = "+" if "+" in expr else "-"
    a, b = (int(part) for part in expr.rstrip("?").split(op))
    answer = a + b if op == "+" else a - b
    return {"captcha_id": challenge["captcha_id"], "captcha_answer": str(answer)}


# --------------------------------------------------------------------------- #
# Default off -- no behaviour change for existing deployments/tests
# --------------------------------------------------------------------------- #


def test_disabled_by_default(client):
    assert settings.captcha_enabled is False


def test_registration_works_without_a_captcha_when_disabled(client, approve_email):
    approve_email("nocaptcha@example.com")
    resp = _register(client, "nocaptcha@example.com")
    assert resp.status_code == 201, resp.text


def test_policy_endpoint_reports_disabled(client):
    body = client.get("/api/v1/auth/registration-policy").json()
    assert body["captcha_required"] is False


# --------------------------------------------------------------------------- #
# Enabled
# --------------------------------------------------------------------------- #


def test_policy_endpoint_reports_enabled(client, captcha_enabled):
    body = client.get("/api/v1/auth/registration-policy").json()
    assert body["captcha_required"] is True


def test_challenge_is_a_simple_arithmetic_question(client, captcha_enabled):
    resp = client.get("/api/v1/auth/captcha")
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"].startswith("What is ")
    assert body["captcha_id"]
    assert body["expires_in"] == settings.captcha_ttl_seconds


def test_registration_without_a_solved_captcha_is_refused(client, captcha_enabled, approve_email):
    approve_email("needscaptcha@example.com")
    resp = _register(client, "needscaptcha@example.com")
    assert resp.status_code == 400
    assert "captcha" in resp.json()["detail"].lower()


def test_registration_with_a_correct_answer_succeeds(client, captcha_enabled, approve_email):
    approve_email("solved@example.com")
    resp = _register(client, "solved@example.com", **_solve(client))
    assert resp.status_code == 201, resp.text


def test_registration_with_a_wrong_answer_is_refused(client, captcha_enabled, approve_email):
    approve_email("wrong@example.com")
    challenge = client.get("/api/v1/auth/captcha").json()
    resp = _register(
        client,
        "wrong@example.com",
        captcha_id=challenge["captcha_id"],
        captcha_answer="not-a-number",
    )
    assert resp.status_code == 400


def test_a_challenge_cannot_be_replayed(client, captcha_enabled, approve_email):
    """Single-use: solving the same challenge twice must fail the second time."""
    approve_email("first@example.com")
    approve_email("second@example.com")
    solution = _solve(client)

    first = _register(client, "first@example.com", **solution)
    assert first.status_code == 201, first.text

    second = _register(client, "second@example.com", **solution)
    assert second.status_code == 400


def test_refusal_does_not_leak_allow_list_state(client, captcha_enabled):
    """A captcha failure and an allow-list refusal must look the same class of
    error regardless of whether the email would have been approved -- captcha
    is checked first, so an unapproved address never learns that from this
    response."""
    resp = _register(client, "unapproved-and-no-captcha@example.com")
    assert resp.status_code == 400


def test_captcha_failure_is_audited(client, captcha_enabled, db):
    from backend.app.models import AuditEvent

    _register(client, "audited@example.com")

    rows = db.query(AuditEvent).filter(AuditEvent.action == "auth.captcha_failed").all()
    assert rows, "a failed captcha left no evidence"


def test_registration_still_requires_allow_list_approval_after_captcha(client, captcha_enabled):
    """Solving the captcha is necessary, not sufficient -- the allow-list still applies."""
    resp = _register(client, "solved-but-unapproved@example.com", **_solve(client))
    assert resp.status_code == 403
