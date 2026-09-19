"""Domain-specific meeting-minutes rendering.

NOT a new extraction pass. Every section here is a re-filing of data that
already exists on `Transcript`, `AgendaBlock`, `Decision` and `ActionItem` --
produced by Person A's pipeline and already reviewed through the existing
decision/action-item workflow. This module only decides which heading a real,
already-extracted item is shown under; it never invents a decision, an owner,
a deadline, a participant or a domain-specific fact that is not already on
one of those rows.

Two signals already computed upstream do double duty as routing rules here,
not classifications:

  * `confidence < LOW_CONFIDENCE` routes a decision/action into an
    "attention"-style section (Risks / Follow-up needed / etc.) -- the same
    signal the reviewer UI already highlights (see `Confidence` in
    `frontend/src/components/common.jsx`).
  * `owner_name` / `deadline` being `None` always renders literally as
    "Not specified" -- never guessed, matching the extractor's own contract
    ("if no pattern matches, the field is None -- never guessed").

Any section a domain asks for that the pipeline has no matching signal for at
all (e.g. exams/events, clinical specifics) renders an explicit sentence
saying so, rather than inventing content. This is the Healthcare requirement
("do not invent... display gracefully") applied to every domain, since the
data-integrity rule is the same regardless of which domain is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.models import MeetingDomain

LOW_CONFIDENCE = 0.5

FOOTER = (
    "Generated from meeting-derived data only. Fields not found in the "
    "transcript are marked 'Not specified'."
)


@dataclass
class Section:
    key: str
    heading: str
    body: str

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "heading": self.heading, "body": self.body}


# --------------------------------------------------------------------------- #
# Shared building blocks -- every domain composes from these
# --------------------------------------------------------------------------- #


def _overview_lines(job: Any, transcript: Any) -> list[str]:
    lines = [f"Meeting: {job.title or job.original_filename}"]
    if transcript is None:
        lines.append("No transcript is available for this meeting yet.")
        return lines

    duration = transcript.duration_seconds
    if duration:
        minutes, seconds = divmod(int(duration), 60)
        lines.append(f"Duration: {minutes:02d}:{seconds:02d}")
    languages = ", ".join(transcript.detected_languages or []) or "not detected"
    code_mixed = " (code-mixed)" if transcript.is_code_mixed else ""
    lines.append(f"Language(s): {languages}{code_mixed}")
    if transcript.diarization_enabled:
        speakers = len(
            {s.get("speaker") for s in (transcript.segments or []) if s.get("speaker")}
        )
        lines.append(f"Speakers identified: {speakers or 'not specified'}")
    return lines


def _topics_body(agenda_blocks: list[Any]) -> str:
    if not agenda_blocks:
        return "No agenda topics were segmented for this meeting."
    return "\n".join(f"- {b.title}" for b in sorted(agenda_blocks, key=lambda b: b.position))


def _decisions_body(decisions: list[Any], *, empty_message: str) -> str:
    if not decisions:
        return empty_message
    return "\n".join(f"- {d.text} (status: {d.status.value})" for d in decisions)


def _actions_body(actions: list[Any], *, empty_message: str) -> str:
    if not actions:
        return empty_message
    lines = []
    for a in actions:
        owner = a.owner_name or "Not specified"
        deadline = a.deadline or "Not specified"
        status = a.status.value
        lines.append(f"- {a.text} | Owner: {owner} | Deadline: {deadline} | Status: {status}")
    return "\n".join(lines)


def _attention_body(decisions: list[Any], actions: list[Any], *, empty_message: str) -> str:
    low = [d.text for d in decisions if d.confidence < LOW_CONFIDENCE]
    low += [a.text for a in actions if a.confidence < LOW_CONFIDENCE]
    if not low:
        return empty_message
    return "\n".join(f"- {text}" for text in low)


def _not_identified(what: str) -> str:
    return f"No {what} was identified by the extraction pipeline for this meeting."


# --------------------------------------------------------------------------- #
# Per-domain templates
# --------------------------------------------------------------------------- #


def _technology(job, transcript, agenda_blocks, decisions, actions) -> list[Section]:
    return [
        Section("overview", "Meeting Overview", "\n".join(_overview_lines(job, transcript))),
        Section("topics", "Technical Topics Discussed", _topics_body(agenda_blocks)),
        Section(
            "decisions",
            "Technical Decisions",
            _decisions_body(
                decisions, empty_message="No technical decisions were recorded for this meeting."
            ),
        ),
        Section(
            "actions",
            "Development Tasks & Owners",
            _actions_body(
                actions, empty_message="No development tasks were recorded for this meeting."
            ),
        ),
        Section(
            "risks",
            "Risks / Blockers (Low-Confidence Items)",
            _attention_body(
                decisions, actions, empty_message="No specific risks or blockers were flagged."
            ),
        ),
        Section(
            "next_steps",
            "Next Steps",
            "The development tasks listed above represent the agreed next steps for this project.",
        ),
        Section("footer", "Note", FOOTER),
    ]


def _education(job, transcript, agenda_blocks, decisions, actions) -> list[Section]:
    return [
        Section("overview", "Meeting Overview", "\n".join(_overview_lines(job, transcript))),
        Section("topics", "Subject / Course Discussion Topics", _topics_body(agenda_blocks)),
        Section(
            "decisions",
            "Decisions",
            _decisions_body(
                decisions, empty_message="No decisions were recorded for this meeting."
            ),
        ),
        Section(
            "actions",
            "Assignments & Responsibilities",
            _actions_body(
                actions, empty_message="No assignments were recorded for this meeting."
            ),
        ),
        Section("events", "Exams / Events Mentioned", _not_identified("exam or event reference")),
        Section(
            "followup",
            "Follow-up Needed & Next Steps",
            _attention_body(
                decisions, actions, empty_message="No items were flagged as needing follow-up."
            )
            + "\n\nSee Assignments & Responsibilities above for the full task list.",
        ),
        Section("footer", "Note", FOOTER),
    ]


def _healthcare(job, transcript, agenda_blocks, decisions, actions) -> list[Section]:
    disclaimer = (
        "This document contains only information extracted from the recorded "
        "meeting text. No clinical, diagnostic or patient-specific details are "
        "inferred or added beyond what participants actually said."
    )
    return [
        Section(
            "overview",
            "Meeting Overview",
            disclaimer + "\n\n" + "\n".join(_overview_lines(job, transcript)),
        ),
        Section(
            "topics", "Clinical / Operational Topics Discussed", _topics_body(agenda_blocks)
        ),
        Section(
            "decisions",
            "Decisions",
            _decisions_body(
                decisions, empty_message="No decisions were recorded for this meeting."
            ),
        ),
        Section(
            "actions",
            "Responsibilities & Follow-up Actions",
            _actions_body(
                actions, empty_message="No follow-up actions were recorded for this meeting."
            ),
        ),
        Section(
            "attention",
            "Items Needing Attention",
            _attention_body(
                decisions, actions, empty_message="No items were flagged as needing attention."
            ),
        ),
        Section(
            "next_steps",
            "Next Steps",
            "The responsibilities and follow-up actions above are the agreed next steps.",
        ),
        Section("footer", "Note", FOOTER),
    ]


def _business(job, transcript, agenda_blocks, decisions, actions) -> list[Section]:
    return [
        Section("overview", "Meeting Overview", "\n".join(_overview_lines(job, transcript))),
        Section("topics", "Key Business Topics", _topics_body(agenda_blocks)),
        Section(
            "decisions",
            "Business Decisions",
            _decisions_body(
                decisions, empty_message="No business decisions were recorded for this meeting."
            ),
        ),
        Section(
            "actions",
            "Action Items & Responsible Party",
            _actions_body(
                actions, empty_message="No action items were recorded for this meeting."
            ),
        ),
        Section(
            "risks",
            "Risks / Issues",
            _attention_body(
                decisions, actions, empty_message="No specific risks or issues were flagged."
            ),
        ),
        Section(
            "next_steps",
            "Follow-up Items & Next Steps",
            "The action items listed above represent the agreed follow-up items for this meeting.",
        ),
        Section("footer", "Note", FOOTER),
    ]


_TEMPLATES = {
    MeetingDomain.TECHNOLOGY: _technology,
    MeetingDomain.EDUCATION: _education,
    MeetingDomain.HEALTHCARE: _healthcare,
    MeetingDomain.BUSINESS: _business,
}


def generate_sections(
    domain: MeetingDomain,
    *,
    job: Any,
    transcript: Any,
    agenda_blocks: list[Any],
    decisions: list[Any],
    actions: list[Any],
) -> list[dict[str, str]]:
    """Render one domain's minutes from already-extracted meeting data.

    Every input is a real ORM row (or None for `transcript`); nothing here
    queries the database or calls any ML service.
    """
    builder = _TEMPLATES[domain]
    sections = builder(job, transcript, agenda_blocks or [], decisions or [], actions or [])
    return [s.to_dict() for s in sections]
