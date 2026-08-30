"""Guardrails: authorization (preventive), output contract (preventive), escalation."""

from __future__ import annotations

import datetime as _dt

import pytest

from cdd_agent.guardrails.authorization import (
    AgentRole,
    AuthorizationError,
    ToolAuthorization,
    ToolName,
)
from cdd_agent.guardrails.escalation import (
    Trigger,
    check_tier1_evidence,
    final_recommendation_review,
)
from cdd_agent.guardrails.output_contract import SchemaViolation, check_deck
from cdd_agent.schemas.common import Citation, ConfidenceTag, SourceKind
from cdd_agent.schemas.deck import Claim, Deck, Slide
from cdd_agent.schemas.evidence import EvidenceItem, EvidenceMatrix
from cdd_agent.schemas.risk import GapOwner, InformationGap, RiskRegister


# --------------------------------------------------------------- authorization
def test_intake_has_no_data_room_access(profile):
    auth = ToolAuthorization(profile)
    decision = auth.check(AgentRole.INTAKE, ToolName.DOCUMENT_RETRIEVAL)
    assert not decision.allowed
    assert "not authorized" in decision.reason


def test_synthesizer_cannot_retrieve_new_evidence(profile):
    auth = ToolAuthorization(profile)
    assert not auth.check(AgentRole.SYNTHESIZER, ToolName.DOCUMENT_RETRIEVAL).allowed


def test_auditor_cannot_commission_outreach(profile):
    """An auditor that generates its own evidence is auditing its own work."""
    auth = ToolAuthorization(profile)
    assert not auth.check(AgentRole.RISK_AUDITOR, ToolName.PRIMARY_RESEARCH).allowed
    assert auth.check(AgentRole.RISK_AUDITOR, ToolName.DOCUMENT_RETRIEVAL).allowed


def test_top5_customer_contact_blocked_pre_signing(profile):
    auth = ToolAuthorization(profile)
    ok = auth.check(AgentRole.ANALYST, ToolName.PRIMARY_RESEARCH, contact_type="customer")
    assert ok.allowed

    blocked = auth.check(
        AgentRole.ANALYST,
        ToolName.PRIMARY_RESEARCH,
        contact_type="customer",
        is_top5_customer=True,
    )
    assert not blocked.allowed
    assert "confirmatory" in blocked.reason


def test_competitor_contact_is_denied_by_intake(profile):
    auth = ToolAuthorization(profile)
    assert not auth.check(
        AgentRole.ANALYST, ToolName.PRIMARY_RESEARCH, contact_type="competitor"
    ).allowed


def test_below_the_line_blocks_all_outreach(profile):
    profile.access.above_the_line = False
    auth = ToolAuthorization(profile)
    for kind in ("customer", "expert"):
        decision = auth.check(
            AgentRole.ANALYST, ToolName.PRIMARY_RESEARCH, contact_type=kind
        )
        assert not decision.allowed
        assert "below-the-line" in decision.reason


def test_authorize_raises_rather_than_warning(profile):
    auth = ToolAuthorization(profile)
    with pytest.raises(AuthorizationError):
        auth.authorize(AgentRole.INTAKE, ToolName.DOCUMENT_RETRIEVAL)


def test_forbidden_tools_are_never_offered(profile):
    """The tool list itself is filtered, so there is nothing to be talked into."""
    auth = ToolAuthorization(profile)
    available = auth.available_tools(AgentRole.INTAKE)
    assert ToolName.DOCUMENT_RETRIEVAL not in available
    assert ToolName.PRIMARY_RESEARCH not in available


# ------------------------------------------------------------- output contract
def _citation() -> Citation:
    return Citation(
        source_kind=SourceKind.DATA_ROOM,
        source_file="Board_Deck.txt",
        locator="slide 1",
        document_date=_dt.date(2026, 5, 10),
    )


def test_claim_cannot_be_constructed_without_a_citation():
    with pytest.raises(ValueError, match="without a citation"):
        Claim(text="NRR is 118%", tag=ConfidenceTag.CONFIRMED)


def test_no_data_claim_may_stand_alone():
    claim = Claim(text="No renewal schedule was provided", tag=ConfidenceTag.NO_DATA)
    assert claim.citations == []


def test_confirmed_on_management_data_alone_is_a_violation():
    deck = Deck(
        engagement_id="e",
        created_by="Synthesizer",
        title="t",
        slides=[
            Slide(
                section_number=2,
                section_title="Market",
                so_what_headline="headline",
                claims=[
                    Claim(
                        text="NRR of 118% is sustainable",
                        tag=ConfidenceTag.CONFIRMED,
                        citations=[_citation()],
                    )
                ],
            )
        ],
    )
    report = check_deck(deck)
    assert not report.ok
    assert any("management-supplied data alone" in v for v in report.violations)


def test_removing_the_draft_notice_is_a_violation():
    deck = Deck(engagement_id="e", created_by="S", title="t", draft_notice="")
    assert any("draft-status notice" in v for v in check_deck(deck).violations)


def test_tier1_without_evidence_or_dated_gap_fails_the_gate(tree):
    matrix = EvidenceMatrix(engagement_id="e", created_by="test")
    deck = Deck(
        engagement_id="e",
        created_by="Synthesizer",
        title="t",
        slides=[
            Slide(section_number=1, section_title="Exec", so_what_headline="h", claims=[])
        ],
    )
    report = check_deck(deck, tree=tree, matrix=matrix, register=RiskRegister(
        engagement_id="e", created_by="test"
    ))
    assert not report.ok
    assert sum("below Partially Confirmed" in v for v in report.violations) == 4

    with pytest.raises(SchemaViolation):
        report.raise_if_violated()


def test_a_dated_gap_satisfies_the_gate(tree):
    matrix = EvidenceMatrix(engagement_id="e", created_by="test")
    register = RiskRegister(
        engagement_id="e",
        created_by="test",
        gaps=[
            InformationGap(
                id=f"GAP-{i}",
                engagement_id="e",
                created_by="Analyst",
                hypothesis_id=h.id,
                request="specific artifact",
                owner=GapOwner.MANAGEMENT,
                target_close_date=_dt.date.today() + _dt.timedelta(days=7),
                blocking=True,
            )
            for i, h in enumerate(tree.tier_1())
        ],
    )
    deck = Deck(
        engagement_id="e",
        created_by="Synthesizer",
        title="t",
        slides=[Slide(section_number=1, section_title="Exec", so_what_headline="h")],
    )
    report = check_deck(deck, tree=tree, matrix=matrix, register=register)
    assert report.ok


# ------------------------------------------------------------------ escalation
def test_weak_tier1_evidence_escalates(tree):
    matrix = EvidenceMatrix(engagement_id="e", created_by="test")
    escalations = check_tier1_evidence(tree, matrix)
    assert len(escalations) == 4
    assert all(e.trigger is Trigger.WEAK_TIER1_EVIDENCE and e.blocking for e in escalations)


def test_evidence_clears_the_escalation(tree):
    matrix = EvidenceMatrix(engagement_id="e", created_by="test")
    for h in tree.tier_1():
        matrix.add(
            EvidenceItem(
                id=f"EV-{h.id}",
                engagement_id="e",
                created_by="Analyst",
                hypothesis_id=h.id,
                claim="evidence",
                tag=ConfidenceTag.PARTIALLY_CONFIRMED,
                citations=[_citation()],
                source_kind=SourceKind.DATA_ROOM,
            )
        )
    assert check_tier1_evidence(tree, matrix) == []


def test_final_recommendation_review_is_unconditional():
    escalation = final_recommendation_review("e")
    assert escalation.trigger is Trigger.FINAL_RECOMMENDATION
    assert escalation.blocking
