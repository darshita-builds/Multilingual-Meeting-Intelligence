"""Domain-specific minutes: generate -> edit -> approve -> gated export -> download.

Mirrors the shape of `test_pipeline_end_to_end.py`'s full-journey test, but for
the new generated-minutes feature: reuses the existing job/run/gate machinery
rather than a parallel one, and checks the same governance properties
(nothing exports without an approval, editing invalidates a stale approval).
"""

from __future__ import annotations

from backend.app.db import SessionLocal
from backend.app.models import GeneratedMinutes, Role


def _run_pipeline_to_ready(client, headers, job):
    """Get a job from 'uploaded' to having reviewed decisions/actions on record,
    exactly like the existing full-pipeline test does."""
    run = client.post(
        f"/api/v1/jobs/{job['id']}/runs", headers=headers, json={"mode": "agent"}
    ).json()
    gate = client.get(f"/api/v1/jobs/{job['id']}/minutes", headers=headers).json()[
        "pending_approvals"
    ][0]
    client.post(
        f"/api/v1/approvals/{gate['id']}/decide", headers=headers, json={"decision": "approved"}
    )
    return run


def test_generate_edit_approve_export_download(client, auth, uploaded_job):
    headers, _user = auth(Role.REVIEWER)
    job = uploaded_job(headers)
    _run_pipeline_to_ready(client, headers, job)

    # --- 1. review: the intended real workflow reviews items BEFORE ------- #
    # generating -- Domain Minutes only includes reviewer-verified
    # (approved/edited) items, so generating straight after the pipeline gate
    # (with everything still "proposed") would produce an empty document.
    minutes = client.get(f"/api/v1/jobs/{job['id']}/minutes", headers=headers).json()
    decisions = minutes["decisions"]
    actions = minutes["action_items"]
    assert decisions and actions, "fixture must produce at least one decision and one action"

    client.post(
        f"/api/v1/decisions/{decisions[0]['id']}/review",
        headers=headers,
        json={"status": "approved"},
    )
    edited_action_text = "Priya to send the finalised API contract to the vendor."
    client.post(
        f"/api/v1/actions/{actions[0]['id']}/review",
        headers=headers,
        json={"status": "edited", "text": edited_action_text},
    )

    # --- 2. generate -------------------------------------------------------- #
    generated = client.post(
        f"/api/v1/jobs/{job['id']}/generated-minutes",
        headers=headers,
        json={"domain": "technology"},
    )
    assert generated.status_code == 201, generated.text
    doc = generated.json()
    assert doc["domain"] == "technology"
    assert doc["status"] == "draft"
    assert len(doc["sections"]) >= 5
    assert doc["original_sections"] == doc["sections"]

    # The reviewed/edited content is actually present in the generated document.
    decisions_section = next(s for s in doc["sections"] if s["key"] == "decisions")
    actions_section = next(s for s in doc["sections"] if s["key"] == "actions")
    assert decisions[0]["text"] in decisions_section["body"], "approved decision must be included"
    assert edited_action_text in actions_section["body"], "edited action must be included"

    history = client.get(f"/api/v1/jobs/{job['id']}/generated-minutes", headers=headers).json()
    assert len(history) == 1

    minutes = client.get(f"/api/v1/jobs/{job['id']}/minutes", headers=headers).json()
    assert len(minutes["generated_minutes"]) == 1

    # --- 3. export before approval is refused outright --------------------- #
    premature = client.post(
        f"/api/v1/jobs/{job['id']}/generated-minutes/{doc['id']}/export",
        headers=headers,
        json={"format": "json"},
    )
    assert premature.status_code == 409, premature.text
    pending = client.get("/api/v1/approvals", headers=headers).json()
    assert not any(g["action"] == "export_minutes_document" for g in pending)

    # --- 4. edit ------------------------------------------------------------ #
    new_sections = [
        {**s, "body": s["body"] + "\n- Manually added note."} for s in doc["sections"]
    ]
    edited = client.patch(
        f"/api/v1/generated-minutes/{doc['id']}",
        headers=headers,
        json={"sections": new_sections},
    )
    assert edited.status_code == 200, edited.text
    assert "Manually added note." in edited.json()["sections"][0]["body"]
    assert edited.json()["status"] == "draft"

    # --- 5. approve ---------------------------------------------------------- #
    approved = client.post(
        f"/api/v1/generated-minutes/{doc['id']}/approve",
        headers=headers,
        json={"note": "Looks right."},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved"
    assert approved.json()["approved_by_id"] == _user.id

    # --- 6. editing an approved document reverts it to draft ---------------- #
    reverted = client.patch(
        f"/api/v1/generated-minutes/{doc['id']}",
        headers=headers,
        json={"sections": new_sections},
    )
    assert reverted.json()["status"] == "draft", "editing must invalidate a stale approval"
    assert reverted.json()["approved_at"] is None

    # re-approve for the export step below
    client.post(f"/api/v1/generated-minutes/{doc['id']}/approve", headers=headers, json={})

    # --- 7. export now opens a gate, same shape as the existing export flow - #
    export_req = client.post(
        f"/api/v1/jobs/{job['id']}/generated-minutes/{doc['id']}/export",
        headers=headers,
        json={"format": "json"},
    )
    assert export_req.status_code == 202, export_req.text
    export_body = export_req.json()
    assert export_body["status"] == "awaiting_approval"
    export_gate = export_body["approval"]
    assert export_gate["action"] == "export_minutes_document"
    assert export_gate["risk"] == "high"

    assert client.get(f"/api/v1/jobs/{job['id']}/exports", headers=headers).json() == []

    decided = client.post(
        f"/api/v1/approvals/{export_gate['id']}/decide",
        headers=headers,
        json={"decision": "approved"},
    )
    assert decided.status_code == 200, decided.text

    exports = client.get(f"/api/v1/jobs/{job['id']}/exports", headers=headers).json()
    assert len(exports) == 1
    assert exports[0]["format"] == "minutes_json"

    download = client.get(f"/api/v1/exports/{exports[0]['id']}/download", headers=headers)
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/json")
    assert "technology" in download.text

    trail = {
        e["action"] for e in client.get(f"/api/v1/jobs/{job['id']}/audit", headers=headers).json()
    }
    for expected in {
        "review.minutes.generated",
        "review.minutes.edited",
        "review.minutes.approved",
        "export.minutes_requested",
    }:
        assert expected in trail, f"missing audit action: {expected}"


def test_regenerate_keeps_history_and_supports_domain_change(client, auth, uploaded_job):
    headers, _ = auth(Role.REVIEWER)
    job = uploaded_job(headers)
    _run_pipeline_to_ready(client, headers, job)

    first = client.post(
        f"/api/v1/jobs/{job['id']}/generated-minutes",
        headers=headers,
        json={"domain": "technology"},
    ).json()
    second = client.post(
        f"/api/v1/jobs/{job['id']}/generated-minutes",
        headers=headers,
        json={"domain": "healthcare"},
    ).json()

    assert first["id"] != second["id"]
    history = client.get(f"/api/v1/jobs/{job['id']}/generated-minutes", headers=headers).json()
    assert len(history) == 2
    assert history[0]["id"] == second["id"], "history is newest first"
    domains = {h["domain"] for h in history}
    assert domains == {"technology", "healthcare"}


def test_viewer_role_cannot_generate_or_approve_or_export(client, auth, uploaded_job):
    """A viewer is refused by role alone, on their OWN job.

    (A viewer probing a job they do not own gets 404 instead, by the same
    IDOR-hiding design already used everywhere else via `OwnedJob` --
    that is not what this test is checking.) Starting a run already requires
    `RequireReviewer` too, so a viewer's own job never even reaches a
    transcript; the role dependency fires before the endpoint body runs, so a
    syntactically valid but non-existent minutes id is enough to prove it.
    """
    viewer_headers, _ = auth(Role.VIEWER)
    job = uploaded_job(viewer_headers)
    placeholder_id = "0" * 36

    assert (
        client.post(
            f"/api/v1/jobs/{job['id']}/generated-minutes",
            headers=viewer_headers,
            json={"domain": "business"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/v1/generated-minutes/{placeholder_id}/approve",
            headers=viewer_headers,
            json={},
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/v1/jobs/{job['id']}/generated-minutes/{placeholder_id}/export",
            headers=viewer_headers,
            json={"format": "csv"},
        ).status_code
        == 403
    )


def test_only_approved_and_edited_items_are_included_in_domain_minutes(client, auth, uploaded_job):
    """Rejected and still-proposed decisions/actions must not appear in a
    generated minutes document; approved and edited ones must. Nothing is
    deleted from the database -- only excluded from this rendering."""
    headers, _ = auth(Role.REVIEWER)
    job = uploaded_job(headers)
    _run_pipeline_to_ready(client, headers, job)

    minutes = client.get(f"/api/v1/jobs/{job['id']}/minutes", headers=headers).json()
    decisions = minutes["decisions"]
    actions = minutes["action_items"]
    assert len(decisions) > 2 and len(actions) > 2, (
        "fixture must produce enough items to exercise approved/edited/rejected/proposed"
    )

    client.post(
        f"/api/v1/decisions/{decisions[0]['id']}/review",
        headers=headers,
        json={"status": "approved"},
    )
    client.post(
        f"/api/v1/decisions/{decisions[1]['id']}/review",
        headers=headers,
        json={"status": "rejected"},
    )
    # decisions[2] is deliberately left untouched -- still "proposed".

    edited_text = "Priya to send the finalised API contract to the vendor."
    client.post(
        f"/api/v1/actions/{actions[0]['id']}/review",
        headers=headers,
        json={"status": "edited", "text": edited_text},
    )
    client.post(
        f"/api/v1/actions/{actions[1]['id']}/review",
        headers=headers,
        json={"status": "rejected"},
    )
    # actions[2] is deliberately left untouched -- still "proposed".

    generated = client.post(
        f"/api/v1/jobs/{job['id']}/generated-minutes",
        headers=headers,
        json={"domain": "business"},
    )
    assert generated.status_code == 201, generated.text
    doc = generated.json()

    decisions_section = next(s for s in doc["sections"] if s["key"] == "decisions")
    actions_section = next(s for s in doc["sections"] if s["key"] == "actions")

    assert decisions[0]["text"] in decisions_section["body"], "approved decision must be included"
    assert decisions[1]["text"] not in decisions_section["body"], "rejected decision must be excluded"
    assert decisions[2]["text"] not in decisions_section["body"], "still-proposed decision must be excluded"

    assert edited_text in actions_section["body"], "edited action must be included, with its edited text"
    assert actions[1]["text"] not in actions_section["body"], "rejected action must be excluded"
    assert actions[2]["text"] not in actions_section["body"], "still-proposed action must be excluded"

    # Nothing was deleted -- rejected/proposed items remain in the database,
    # available for audit/error analysis.
    minutes_after = client.get(f"/api/v1/jobs/{job['id']}/minutes", headers=headers).json()
    assert len(minutes_after["decisions"]) == len(decisions)
    assert len(minutes_after["action_items"]) == len(actions)
    assert any(d["status"] == "rejected" for d in minutes_after["decisions"])
    assert any(d["status"] == "proposed" for d in minutes_after["decisions"])


def test_deleting_a_job_cascades_generated_minutes(client, auth, uploaded_job):
    headers, _ = auth(Role.REVIEWER)
    job = uploaded_job(headers)
    _run_pipeline_to_ready(client, headers, job)
    doc = client.post(
        f"/api/v1/jobs/{job['id']}/generated-minutes",
        headers=headers,
        json={"domain": "education"},
    ).json()

    client.delete(f"/api/v1/jobs/{job['id']}", headers=headers)

    with SessionLocal() as session:
        assert session.get(GeneratedMinutes, doc["id"]) is None
