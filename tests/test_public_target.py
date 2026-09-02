"""Listed targets under the three structures a buyer can actually pursue.

The interesting assertions here are not that public-company features exist, but that
they are *scoped*: a private deal must be untouched by any of it, and the three
public structures must not collapse into one. A minority holder who cannot compel a
decision and a bidder who will own the company outright are asking different
questions of the same filings, and the system should say so.
"""

from __future__ import annotations

import pytest

from cdd_agent.guardrails.authorization import AgentRole, ToolAuthorization, ToolName
from cdd_agent.knowledge.data_request_catalog import catalog_for, public_record_note
from cdd_agent.knowledge.outline import tailored_outline
from cdd_agent.knowledge.risk_taxonomy import applicable_categories
from cdd_agent.retrieval.chunking import classify_source_kind
from cdd_agent.schemas.common import SourceKind
from cdd_agent.schemas.deal_profile import (
    AccessConstraints,
    BuyerProfile,
    DealProfile,
    DealShape,
    InvestmentThesis,
    PublicMarketContext,
    SectorDefinition,
    TargetIdentification,
    TransactionStructure,
    VDRAccess,
)
from cdd_agent.schemas.risk import RiskCategory

MINORITY = TransactionStructure.PUBLIC_MINORITY_STAKE
CONTROL = TransactionStructure.PUBLIC_CONTROL_STAKE
PRIVATE = TransactionStructure.TAKE_PRIVATE


def shape(structure=MINORITY, public=True, strategic=False) -> DealShape:
    return DealShape(strategic_buyer=strategic, public_target=public,
                     structure=structure)


def profile(structure=PRIVATE, *, ticker="MRDN", mnpi=True, ack=True,
            issuer_contact=False, **public_kwargs) -> DealProfile:
    return DealProfile(
        engagement_id="atlas", created_by="test",
        target=TargetIdentification(
            legal_name="Meridian Data Systems", publicly_traded=True,
            transaction_structure=structure),
        public_market=PublicMarketContext(ticker=ticker, exchange="NASDAQ",
                                          **public_kwargs),
        sector=SectorDefinition(sub_sector="enterprise data-integration SaaS"),
        thesis=InvestmentThesis(one_sentence_thesis="Take Meridian private."),
        buyer=BuyerProfile(decision_criteria=["growth optionality"]),
        access=AccessConstraints(vdr_access=VDRAccess.VDR_LINK, mnpi_expected=mnpi,
                                 trading_restriction_acknowledged=ack,
                                 issuer_contact_permitted=issuer_contact),
    )


# --------------------------------------------------------------- deal structure
def test_the_three_structures_differ_on_control_and_listing():
    """The two facts everything else keys off."""
    assert not MINORITY.confers_control, "a minority holder compels nothing"
    assert MINORITY.retains_listing
    assert CONTROL.confers_control and CONTROL.retains_listing
    assert PRIVATE.confers_control and not PRIVATE.retains_listing
    assert PRIVATE.requires_shareholder_approval
    assert not MINORITY.requires_shareholder_approval


def test_a_listed_target_without_a_named_structure_cannot_start_phase_1():
    """Decomposing before the structure is known tests the wrong question."""
    p = profile(structure=TransactionStructure.UNKNOWN)
    ready, missing = p.is_ready_for_phase_1()
    assert not ready
    assert any("which public structure" in m for m in missing)


def test_a_private_deal_is_untouched_by_any_of_this():
    p = DealProfile(
        engagement_id="e", created_by="t",
        target=TargetIdentification(
            legal_name="Private Co", publicly_traded=False,
            transaction_structure=TransactionStructure.MAJORITY_BUYOUT),
        sector=SectorDefinition(sub_sector="SaaS"),
        thesis=InvestmentThesis(one_sentence_thesis="Buy it."),
        buyer=BuyerProfile(decision_criteria=["cash flow"]))
    assert not p.is_public_target
    ready, missing = p.is_ready_for_phase_1()
    assert ready, missing
    private = DealShape(structure=TransactionStructure.MAJORITY_BUYOUT)
    assert catalog_for(private) == ()
    listed = [c for c in applicable_categories(private) if "listed targets" in c.value]
    assert not listed, "a private deal must not be scored on listed-target risks"


# --------------------------------------------------------------------- MNPI
def test_the_data_room_does_not_open_until_the_restriction_is_acknowledged():
    """Reading a data room on a listed issuer puts the firm in possession of MNPI.

    That is not a decision the agent may take on the firm behalf, so the tools that
    would create the exposure are withheld rather than warned about.
    """
    auth = ToolAuthorization(profile(ack=False))
    for tool in (ToolName.DOCUMENT_RETRIEVAL, ToolName.STRUCTURED_COMPUTATION):
        decision = auth.check(AgentRole.ANALYST, tool)
        assert not decision.allowed
        assert "trading restriction" in decision.reason
    available = auth.available_tools(AgentRole.ANALYST)
    assert ToolName.DOCUMENT_RETRIEVAL not in available, "a blocked tool is not offered"
    # The public record is exactly what may be worked on before wall-crossing.
    assert ToolName.MARKET_SEARCH in available


def test_acknowledgement_unblocks_the_data_room():
    auth = ToolAuthorization(profile(ack=True))
    assert auth.check(AgentRole.ANALYST, ToolName.DOCUMENT_RETRIEVAL).allowed
    assert auth.mnpi_block is None


def test_issuer_contact_is_denied_but_independent_expert_calls_are_not():
    """Reg FD: the exposure of a leaky call lands on the asset being bought."""
    auth = ToolAuthorization(profile())
    denied = auth.check(AgentRole.ANALYST, ToolName.PRIMARY_RESEARCH,
                        contact_type="issuer investor relations")
    assert not denied.allowed
    assert "Reg FD" in denied.reason
    assert auth.check(AgentRole.ANALYST, ToolName.PRIMARY_RESEARCH,
                      contact_type="expert").allowed
    permitted = ToolAuthorization(profile(issuer_contact=True))
    assert permitted.check(AgentRole.ANALYST, ToolName.PRIMARY_RESEARCH,
                           contact_type="issuer").allowed


def test_a_private_target_is_not_subject_to_the_issuer_rule():
    p = profile()
    p.target.publicly_traded = False
    p.target.transaction_structure = TransactionStructure.MAJORITY_BUYOUT
    auth = ToolAuthorization(p)
    assert auth.check(AgentRole.ANALYST, ToolName.PRIMARY_RESEARCH,
                      contact_type="management").allowed


# ------------------------------------------------------------------ evidence
def test_sell_side_research_never_counts_as_triangulation():
    """Analysts are guided by the company, so consensus agreeing with the plan is
    one source counted twice - the failure the confidence schema exists to stop."""
    assert not SourceKind.SELL_SIDE_RESEARCH.is_independent
    assert not SourceKind.SELL_SIDE_RESEARCH.is_management_supplied
    assert SourceKind.PRIMARY_RESEARCH.is_independent


def test_a_filing_is_attested_but_still_the_issuer_own_account():
    assert SourceKind.PUBLIC_FILING.is_management_supplied
    assert SourceKind.PUBLIC_FILING.is_attested
    assert not SourceKind.PUBLIC_FILING.is_independent
    assert SourceKind.PUBLIC_FILING.is_public_record


@pytest.mark.parametrize("filename,expected", [
    ("Meridian_10-K_FY2025.txt", SourceKind.PUBLIC_FILING),
    ("Meridian_DEF-14A_proxy_2026.txt", SourceKind.PUBLIC_FILING),
    ("Q4_earnings-call.txt", SourceKind.PUBLIC_FILING),
    ("Wexford_analyst_note.txt", SourceKind.SELL_SIDE_RESEARCH),
    ("Project_Atlas_Board_Deck.txt", SourceKind.DATA_ROOM),
    ("MSA_TopAccounts.txt", SourceKind.DATA_ROOM),
])
def test_public_record_is_told_apart_from_confidential_material(filename, expected):
    """A data room routinely holds the last 10-K next to the board pack.

    They are not the same evidence: one is attested and creates no MNPI to read.
    Ambiguity resolves towards the more restrictive classification.
    """
    assert classify_source_kind(filename) is expected


# ------------------------------------------------------------------ scoping
@pytest.mark.parametrize("structure,expected", [
    (MINORITY, {RiskCategory.MARKET_EXPECTATIONS, RiskCategory.GOVERNANCE_CONTROL}),
    (CONTROL, {RiskCategory.MARKET_EXPECTATIONS, RiskCategory.GOVERNANCE_CONTROL,
               RiskCategory.DEAL_COMPLETION}),
    (PRIVATE, {RiskCategory.MARKET_EXPECTATIONS, RiskCategory.DEAL_COMPLETION}),
])
def test_risk_taxonomy_is_scoped_to_what_the_structure_can_raise(structure, expected):
    """A take-private has no continuing minority holders to answer to; a minority
    stake has no completion condition of its own to fail."""
    got = {c for c in applicable_categories(shape(structure)) if "listed" in c.value}
    assert got == expected


def test_the_outline_asks_each_structure_its_own_question():
    def elements(structure):
        return " ".join(
            e for s in tailored_outline("enterprise SaaS", "subscription/SaaS",
                                        shape(structure))
            for e in s.key_elements).lower()

    minority, control, take_private = map(elements, (MINORITY, CONTROL, PRIVATE))
    # Every listed structure argues with the market price.
    for text in (minority, control, take_private):
        assert "consensus" in text and "unaffected" in text
    # Only the minority holder has to live without control.
    assert "board seat" in minority and "board seat" not in take_private
    # Only the take-private removes the listing, and only it needs a vote.
    assert "delisting" in take_private and "delisting" not in minority
    assert "shareholder vote" in take_private
    # Only the control case keeps minority shareholders alongside.
    assert "minority shareholders" in control


def test_the_request_does_not_ask_management_for_what_they_published():
    where = public_record_note("Audited/reviewed financials (3-5 years)")
    assert "Annual report" in where
    assert public_record_note("Cohort retention data") == "", \
        "cohort retention is not in the filings - it must still be requested"


@pytest.mark.parametrize("structure,marker", [
    (MINORITY, "standstill"),
    (CONTROL, "Related-party"),
    (PRIVATE, "rollover"),
])
def test_each_structure_requests_what_only_it_needs(structure, marker):
    items = " ".join(i.item for i in catalog_for(shape(structure)))
    assert marker in items
    # And the public record is requested for every listed structure.
    assert "annual reports" in items
