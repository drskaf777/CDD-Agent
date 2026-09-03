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
from cdd_agent.schemas.deck import ExhibitStatus
from cdd_agent.schemas.risk import RiskCategory
from cdd_agent.synthesis.exhibits import FIGURE as FIGURE_RE

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


# --------------------------------------------------------------- critic scoring
def test_a_score_written_as_prose_is_recovered_not_discarded():
    """Observed live: the model put the reason in the score field for two criteria
    and the ValidationError ended the entire Phase-1 search."""
    from cdd_agent.agents.thesis_architect.critic import CriticVerdict

    v = CriticVerdict(buyer_criteria_coverage="4 - covers cash-flow stability",
                      four_question_alignment=3.5,
                      sub_sector_fit="5. strong fit for enterprise SaaS",
                      testability=4)
    assert v.buyer_criteria_coverage == 4.0
    assert v.sub_sector_fit == 5.0


def test_prose_with_no_score_is_still_rejected():
    """The prune threshold is 3.0 across a three-branch beam, so a guessed score
    silently decides which decomposition the engagement runs on."""
    from pydantic import ValidationError

    from cdd_agent.agents.thesis_architect.critic import CriticVerdict

    with pytest.raises(ValidationError):
        CriticVerdict(buyer_criteria_coverage="the tree is broad but shallow",
                      four_question_alignment=3, sub_sector_fit=3, testability=3)


def test_one_unscorable_branch_does_not_end_the_search(monkeypatch):
    """CrewAI raises inside kickoff, so the existing raw-string fallback never ran."""
    from cdd_agent.agents.thesis_architect import critic as critic_mod

    c = critic_mod.Critic.__new__(critic_mod.Critic)
    c.profile = profile()
    c.settings = type("S", (), {"offline": False, "crewai_tracing": False})()
    c.market_search = None
    monkeypatch.setattr(critic_mod.Critic, "_kb_context", lambda self, tree: "")

    class Boom:
        def __init__(self, *a, **k): pass
        def kickoff(self): raise ValueError("unparseable verdict")

    import sys
    import types
    fake = types.ModuleType("crewai")
    fake.Agent = lambda **k: None
    fake.Task = lambda **k: None
    fake.Crew = Boom
    monkeypatch.setitem(sys.modules, "crewai", fake)
    monkeypatch.setattr("cdd_agent.llm.models.get_crew_llm", lambda: None, raising=False)

    from cdd_agent.schemas.hypothesis import HypothesisTree
    tree = HypothesisTree(engagement_id="e", created_by="t", root_thesis="t",
                          branch_id="growth", framing_key="growth",
                          framing_label="growth-led")
    verdict = c._crew_verdict(tree)
    assert 1 <= verdict.buyer_criteria_coverage <= 5
    assert "could not be scored" in verdict.notes, "the degradation must be visible"


def test_a_quote_is_credited_to_the_document_it_came_from():
    """Live failure: a sentence from the earnings call was credited to the board deck.

    An evidence item routinely carries several citations, and naming the first one
    regardless of which chunk the sentence came out of is a sourcing error - the
    class of error this system exists to catch rather than commit.
    """
    from cdd_agent.schemas.common import Citation, ConfidenceTag
    from cdd_agent.schemas.evidence import EvidenceItem, EvidenceMatrix
    from cdd_agent.schemas.hypothesis import HypothesisTree
    from cdd_agent.schemas.risk import RiskRegister
    from cdd_agent.synthesis.exhibits import ExhibitContext

    matrix = EvidenceMatrix(engagement_id="e", created_by="t")
    matrix.add(EvidenceItem(
        id="EV-1", engagement_id="e", created_by="Analyst", hypothesis_id="H1",
        claim="Retrieved evidence bearing on H1: assorted context.",
        tag=ConfidenceTag.PARTIALLY_CONFIRMED, source_kind=SourceKind.DATA_ROOM,
        citations=[
            Citation(source_kind=SourceKind.DATA_ROOM, source_file="Board_Deck.txt",
                     locator="slide 1", quoted_text="The plan assumes recovery."),
            Citation(source_kind=SourceKind.PUBLIC_FILING,
                     source_file="Earnings_Call.txt", locator="page 2",
                     quoted_text="In fiscal 2025 we guided 21% and delivered 22%."),
        ]))
    ctx = ExhibitContext(
        tree=HypothesisTree(engagement_id="e", created_by="t", root_thesis="t",
                            branch_id="b", framing_key="growth",
                            framing_label="growth-led"),
        matrix=matrix, register=RiskRegister(engagement_id="e", created_by="t"),
        computation=None)
    (item, sentence), = ctx.statements("guided", limit=1)
    assert "guided 21%" in sentence
    assert item.citations[0].source_file == "Earnings_Call.txt", \
        "the quote must be credited to the document it was taken from"
    # The matrix itself is untouched, so the next exhibit sees the original order.
    assert matrix.items[0].citations[0].source_file == "Board_Deck.txt"


def test_method_references_are_never_quoted_as_findings_about_the_target():
    """The Knowledge Base holds the outline, the taxonomy, the request catalogue.

    Live run on real filings put "[Tier 2] Financial Records: Current-year budget vs."
    into a deck under "Management plan", cited to our own data-request catalogue. It
    reads as a finding about the company and is nothing of the kind.
    """
    from cdd_agent.schemas.common import Citation, ConfidenceTag
    from cdd_agent.schemas.evidence import EvidenceItem, EvidenceMatrix
    from cdd_agent.schemas.hypothesis import HypothesisTree
    from cdd_agent.schemas.risk import RiskRegister
    from cdd_agent.synthesis.exhibits import ExhibitContext

    matrix = EvidenceMatrix(engagement_id="e", created_by="t")
    matrix.add(EvidenceItem(
        id="EV-KB", engagement_id="e", created_by="Analyst", hypothesis_id="H1",
        claim="Retrieved evidence bearing on H1: reference material.",
        tag=ConfidenceTag.PARTIALLY_CONFIRMED, source_kind=SourceKind.KNOWLEDGE_BASE,
        citations=[Citation(source_kind=SourceKind.KNOWLEDGE_BASE,
                            source_file="kb_universal_data_request_catalog.txt",
                            locator="para 1",
                            quoted_text="Guidance of 20% growth is a Tier 2 item.")]))
    ctx = ExhibitContext(
        tree=HypothesisTree(engagement_id="e", created_by="t", root_thesis="t",
                            branch_id="b", framing_key="growth",
                            framing_label="growth-led"),
        matrix=matrix, register=RiskRegister(engagement_id="e", created_by="t"),
        computation=None)
    assert ctx.statements("guidance", figure=FIGURE_RE, limit=3) == []


def _ctx_with(quoted: str, kind=SourceKind.PUBLIC_FILING, filename="FRSH_10-K.txt"):
    from cdd_agent.schemas.common import Citation, ConfidenceTag
    from cdd_agent.schemas.evidence import EvidenceItem, EvidenceMatrix
    from cdd_agent.schemas.hypothesis import HypothesisTree
    from cdd_agent.schemas.risk import RiskRegister
    from cdd_agent.synthesis.exhibits import ExhibitContext

    matrix = EvidenceMatrix(engagement_id="e", created_by="t")
    matrix.add(EvidenceItem(
        id="EV-1", engagement_id="e", created_by="Analyst", hypothesis_id="H1",
        claim="Retrieved evidence bearing on H1: context.",
        tag=ConfidenceTag.PARTIALLY_CONFIRMED, source_kind=kind,
        citations=[Citation(source_kind=kind, source_file=filename,
                            locator="para 1", quoted_text=quoted)]))
    return ExhibitContext(
        tree=HypothesisTree(engagement_id="e", created_by="t", root_thesis="t",
                            branch_id="b", framing_key="growth",
                            framing_label="growth-led"),
        matrix=matrix, register=RiskRegister(engagement_id="e", created_by="t"),
        computation=None)


def test_accounting_estimates_are_not_analyst_estimates():
    """Every filing carries the phrase; matching it put a stock-compensation note
    into the deck under Published consensus."""
    from cdd_agent.synthesis.exhibits import CATALOGUE, consensus_vs_plan

    spec = next(s for s in CATALOGUE if s.key == "consensus_vs_plan")
    accounting = _ctx_with(
        "Significant assumptions and estimates used in preparing our consolidated "
        "financial statements include those related to the useful lives of 3 assets.")
    built = consensus_vs_plan(accounting, spec)
    assert built.status is ExhibitStatus.GAP, "accounting language is not consensus"

    real = _ctx_with("Consensus across the 15 covering analysts is $960.9 million.")
    built = consensus_vs_plan(real, spec)
    assert built.status is ExhibitStatus.EVIDENCED
    assert "960.9" in built.rows[0][1]
