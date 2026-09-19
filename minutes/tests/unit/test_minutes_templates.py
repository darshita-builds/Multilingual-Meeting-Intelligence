"""Domain-minutes template engine: no DB, no ML -- pure re-filing of already-
extracted data. See `backend/app/services/minutes_templates.py`'s module
docstring for the design principle these tests check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.models import ItemStatus, MeetingDomain
from backend.app.services.minutes_templates import generate_sections


def _job(title="Sprint review"):
    return SimpleNamespace(title=title, original_filename="standup.wav")


def _transcript(**overrides):
    base = {
        "duration_seconds": 125.0,
        "detected_languages": ["en", "hi"],
        "is_code_mixed": True,
        "diarization_enabled": True,
        "segments": [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _block(position, title):
    return SimpleNamespace(position=position, title=title)


def _decision(text, confidence=0.8, status=ItemStatus.PROPOSED):
    return SimpleNamespace(text=text, confidence=confidence, status=status)


def _action(text, owner_name=None, deadline=None, confidence=0.8, status=ItemStatus.PROPOSED):
    return SimpleNamespace(
        text=text, owner_name=owner_name, deadline=deadline, confidence=confidence, status=status
    )


ALL_DOMAINS = [
    MeetingDomain.TECHNOLOGY,
    MeetingDomain.EDUCATION,
    MeetingDomain.HEALTHCARE,
    MeetingDomain.BUSINESS,
]


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_generates_a_section_per_domain_with_real_data(domain):
    sections = generate_sections(
        domain,
        job=_job(),
        transcript=_transcript(),
        agenda_blocks=[_block(0, "Payments module"), _block(1, "Infra migration")],
        decisions=[_decision("We decided to go with managed Postgres.")],
        actions=[_action("Send the API contract", owner_name="Priya", deadline="2026-10-05")],
    )

    keys = {s["key"] for s in sections}
    headings = [s["heading"] for s in sections]
    assert "overview" in keys
    assert len(headings) == len(set(headings)), "headings must not repeat within one document"

    overview = next(s for s in sections if s["key"] == "overview")
    assert "Sprint review" in overview["body"]

    topics = next(s for s in sections if "topic" in s["key"] or "Topics" in s["heading"])
    assert "Payments module" in topics["body"]
    assert "Infra migration" in topics["body"]

    actions_section = next(s for s in sections if s["key"] == "actions")
    assert "Priya" in actions_section["body"]
    assert "2026-10-05" in actions_section["body"]


def test_four_domains_produce_visibly_different_headings():
    kwargs = {
        "job": _job(),
        "transcript": _transcript(),
        "agenda_blocks": [_block(0, "Topic")],
        "decisions": [_decision("A decision was made.")],
        "actions": [_action("Do the thing")],
    }
    headings_by_domain = {
        domain: tuple(s["heading"] for s in generate_sections(domain, **kwargs))
        for domain in ALL_DOMAINS
    }
    # No two domains render the exact same heading sequence.
    assert len(set(headings_by_domain.values())) == len(ALL_DOMAINS)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_missing_owner_and_deadline_render_not_specified_never_guessed(domain):
    sections = generate_sections(
        domain,
        job=_job(),
        transcript=_transcript(),
        agenda_blocks=[],
        decisions=[],
        actions=[_action("Follow up with the vendor", owner_name=None, deadline=None)],
    )
    actions_section = next(s for s in sections if s["key"] == "actions")
    assert "Not specified" in actions_section["body"]
    assert "None" not in actions_section["body"]


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_empty_meeting_data_renders_graceful_messages_not_blank(domain):
    sections = generate_sections(
        domain, job=_job(), transcript=None, agenda_blocks=[], decisions=[], actions=[]
    )
    for section in sections:
        assert section["body"].strip(), f"section {section['key']!r} must never be blank"
    body_text = " ".join(s["body"] for s in sections)
    # Every domain must be honest about absent data rather than silent or invented.
    assert "No transcript is available" in body_text


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_low_confidence_items_are_routed_to_the_attention_section(domain):
    sections = generate_sections(
        domain,
        job=_job(),
        transcript=_transcript(),
        agenda_blocks=[],
        decisions=[_decision("A shaky decision.", confidence=0.2)],
        actions=[_action("A shaky task", confidence=0.1)],
    )
    attention = next(s for s in sections if s["key"] in ("risks", "attention", "followup"))
    assert "shaky decision" in attention["body"]
    assert "shaky task" in attention["body"]


def test_healthcare_never_invents_clinical_content():
    sections = generate_sections(
        MeetingDomain.HEALTHCARE,
        job=_job(),
        transcript=_transcript(),
        agenda_blocks=[],
        decisions=[],
        actions=[],
    )
    overview = next(s for s in sections if s["key"] == "overview")
    assert "No clinical, diagnostic or patient-specific details are inferred" in overview["body"]


def test_footer_states_the_data_integrity_rule_in_every_document():
    for domain in ALL_DOMAINS:
        sections = generate_sections(
            domain, job=_job(), transcript=_transcript(), agenda_blocks=[], decisions=[], actions=[]
        )
        footer = next(s for s in sections if s["key"] == "footer")
        assert "Generated from meeting-derived data only" in footer["body"]
