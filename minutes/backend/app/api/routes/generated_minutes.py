"""Domain-specific minutes: generate, edit, approve, export.

Generation is a pure re-rendering of data the platform already has (see
`services/minutes_templates.py`) -- no new extraction, no ML call. Review
follows the same human-in-the-loop principle already used for
`Decision`/`ActionItem` (draft -> approved, never auto-approved). Export
reuses the project's existing gated-export machinery
(`agent/orchestrator.py::request_minutes_export`) rather than a parallel path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, status

from backend.app.agent import orchestrator
from backend.app.api.deps import Audit, DbSession, OwnedJob, RequireReviewer
from backend.app.models import (
    ActionItem,
    AgendaBlock,
    Decision,
    GeneratedMinutes,
    Job,
    MinutesStatus,
    Role,
    Transcript,
)
from backend.app.schemas import (
    ApprovalOut,
    GeneratedMinutesApprove,
    GeneratedMinutesEdit,
    GeneratedMinutesOut,
    MeetingDomainIn,
    MinutesExportRequest,
)
from backend.app.services import minutes_templates
from backend.app.services.export import EXPORTABLE_STATUSES

router = APIRouter(tags=["generated-minutes"])


def _guard(db: DbSession, minutes_id: str, user) -> GeneratedMinutes:
    """Return the document if this user may act on it, else 404 (same
    ownership-hides-existence convention as `routes/minutes.py::_guard`)."""
    not_found = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="Minutes document not found."
    )
    doc = db.get(GeneratedMinutes, minutes_id)
    if doc is None:
        raise not_found
    job = db.get(Job, doc.job_id)
    if job is None or (job.owner_id != user.id and user.role is not Role.ADMIN):
        raise not_found
    return doc


@router.post(
    "/jobs/{job_id}/generated-minutes",
    response_model=GeneratedMinutesOut,
    status_code=status.HTTP_201_CREATED,
    summary="Generate domain-specific minutes from this meeting's existing data",
)
def generate_minutes(
    job: OwnedJob,
    payload: MeetingDomainIn,
    db: DbSession,
    user: RequireReviewer,
    audit: Audit,
) -> GeneratedMinutes:
    """Regenerating (a new domain, or a fresh pass) creates a new row; earlier
    generations stay in history rather than being overwritten."""
    transcript = db.query(Transcript).filter(Transcript.job_id == job.id).one_or_none()
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No transcript yet. Run the pipeline for this meeting first.",
        )

    agenda_blocks = (
        db.query(AgendaBlock)
        .filter(AgendaBlock.transcript_id == transcript.id)
        .order_by(AgendaBlock.position)
        .all()
    )
    # Only reviewer-verified items go into the final document -- the same
    # APPROVED/EDITED set `services/export.py` already treats as "reviewed and
    # trustworthy" for action-item export. A still-PROPOSED item has not been
    # looked at yet and a REJECTED one was explicitly ruled out; neither
    # belongs in a document meant to represent verified meeting information.
    # Nothing is deleted -- both remain in the database for audit/error
    # analysis, just excluded from this rendering.
    decisions = (
        db.query(Decision)
        .filter(Decision.job_id == job.id, Decision.status.in_(EXPORTABLE_STATUSES))
        .order_by(Decision.created_at)
        .all()
    )
    actions = (
        db.query(ActionItem)
        .filter(ActionItem.job_id == job.id, ActionItem.status.in_(EXPORTABLE_STATUSES))
        .order_by(ActionItem.created_at)
        .all()
    )

    sections = minutes_templates.generate_sections(
        payload.domain,
        job=job,
        transcript=transcript,
        agenda_blocks=agenda_blocks,
        decisions=decisions,
        actions=actions,
    )

    doc = GeneratedMinutes(
        job_id=job.id,
        domain=payload.domain,
        sections=sections,
        original_sections=sections,
        status=MinutesStatus.DRAFT,
        created_by_id=user.id,
    )
    db.add(doc)
    db.flush()

    audit.human(
        "review.minutes.generated",
        user_id=user.id,
        job_id=job.id,
        resource_type="generated_minutes",
        resource_id=doc.id,
        detail={"domain": payload.domain.value, "section_count": len(sections)},
    )
    return doc


@router.get(
    "/jobs/{job_id}/generated-minutes",
    response_model=list[GeneratedMinutesOut],
    summary="History of generated minutes for this meeting, newest first",
)
def list_generated_minutes(job: OwnedJob, db: DbSession) -> list[GeneratedMinutes]:
    return (
        db.query(GeneratedMinutes)
        .filter(GeneratedMinutes.job_id == job.id)
        .order_by(GeneratedMinutes.created_at.desc())
        .all()
    )


@router.patch(
    "/generated-minutes/{minutes_id}",
    response_model=GeneratedMinutesOut,
    summary="Edit a generated minutes document",
)
def edit_generated_minutes(
    minutes_id: Annotated[str, Path(min_length=36, max_length=36)],
    payload: GeneratedMinutesEdit,
    db: DbSession,
    user: RequireReviewer,
    audit: Audit,
) -> GeneratedMinutes:
    doc = _guard(db, minutes_id, user)

    was_approved = doc.status is MinutesStatus.APPROVED
    doc.sections = [s.model_dump() for s in payload.sections]
    doc.edited_by_id = user.id
    doc.edited_at = datetime.now(UTC)
    if was_approved:
        # Content changed after approval -- the approval no longer covers it,
        # same principle as the payload-fingerprint invalidation in agent/gates.py.
        doc.status = MinutesStatus.DRAFT
        doc.approved_by_id = None
        doc.approved_at = None
        doc.approval_note = None
    db.flush()

    audit.human(
        "review.minutes.edited",
        user_id=user.id,
        job_id=doc.job_id,
        resource_type="generated_minutes",
        resource_id=doc.id,
        detail={"reverted_from_approved": was_approved},
    )
    return doc


@router.post(
    "/generated-minutes/{minutes_id}/approve",
    response_model=GeneratedMinutesOut,
    summary="Approve a generated minutes document (required before export)",
)
def approve_generated_minutes(
    minutes_id: Annotated[str, Path(min_length=36, max_length=36)],
    payload: GeneratedMinutesApprove,
    db: DbSession,
    user: RequireReviewer,
    audit: Audit,
) -> GeneratedMinutes:
    doc = _guard(db, minutes_id, user)

    doc.status = MinutesStatus.APPROVED
    doc.approved_by_id = user.id
    doc.approved_at = datetime.now(UTC)
    doc.approval_note = payload.note
    db.flush()

    audit.human(
        "review.minutes.approved",
        user_id=user.id,
        job_id=doc.job_id,
        resource_type="generated_minutes",
        resource_id=doc.id,
        detail={"domain": doc.domain.value, "note": payload.note},
    )
    return doc


@router.post(
    "/jobs/{job_id}/generated-minutes/{minutes_id}/export",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request an export of an approved minutes document (opens a human approval gate)",
    responses={202: {"description": "Approval gate opened; no file written yet."}},
)
def request_minutes_export(
    job: OwnedJob,
    minutes_id: Annotated[str, Path(min_length=36, max_length=36)],
    payload: MinutesExportRequest,
    db: DbSession,
    user: RequireReviewer,
    audit: Audit,
) -> dict:
    doc = _guard(db, minutes_id, user)
    if doc.job_id != job.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Minutes document not found."
        )

    # Pre-flight, same reasoning as routes/exports.py: never open a gate for an
    # export that cannot succeed -- a document must be reviewer-approved first.
    if doc.status is not MinutesStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This minutes document is not yet approved. Approve it before exporting.",
        )

    run, gate = orchestrator.request_minutes_export(
        db, audit, job=job, generated_minutes=doc, actor_id=user.id, fmt=payload.format
    )

    return {
        "status": "awaiting_approval",
        "run_id": run.id,
        "approval": ApprovalOut.model_validate(gate).model_dump(mode="json"),
        "next_step": f"POST /api/v1/approvals/{gate.id}/decide",
    }
