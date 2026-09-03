"""The class of failure where an engagement artifacts stop describing one company.

These reproduce a real incident. A control labelled "Reload profile" loaded a demo
fixture over a Deal Profile Brief built from real SEC filings. Every count on the
dashboard stayed correct, the deck still rendered, and the next Phase-1 run
decomposed the wrong target against the right evidence. The reported symptom was
that the run felt slow.

What makes it detectable is that each artifact copies a value from the one above it
when it is built, so a divergence can only mean an upstream artifact was replaced.
"""

from __future__ import annotations

import pytest

from cdd_agent.guardrails.coherence import (
    EngagementIncoherent,
    check_engagement,
    raise_if_incoherent,
)
from cdd_agent.schemas.deal_profile import (
    BuyerProfile,
    DealProfile,
    InvestmentThesis,
    SectorDefinition,
    TargetIdentification,
)
from cdd_agent.schemas.deck import Deck
from cdd_agent.schemas.hypothesis import HypothesisTree


def profile(name="Freshworks Inc.", thesis="Take Freshworks private at a premium.",
            engagement="project-kanpur") -> DealProfile:
    return DealProfile(
        engagement_id=engagement, created_by="Intake Agent",
        target=TargetIdentification(legal_name=name),
        sector=SectorDefinition(sub_sector="enterprise SaaS"),
        thesis=InvestmentThesis(one_sentence_thesis=thesis),
        buyer=BuyerProfile(decision_criteria=["cash-flow stability"]))


def tree(thesis="Take Freshworks private at a premium.",
         engagement="project-kanpur") -> HypothesisTree:
    return HypothesisTree(
        engagement_id=engagement, created_by="Thesis Architect", root_thesis=thesis,
        branch_id="growth", framing_key="growth", framing_label="growth-led")


def test_a_coherent_engagement_reports_nothing():
    assert check_engagement("project-kanpur", profile=profile(), tree=tree()) == []


def test_swapping_the_profile_under_a_built_tree_is_caught():
    """The exact incident: the brief was replaced, the tree was not."""
    swapped = profile(name="Sentinel Secure Ltd",
                      thesis="Acquire Sentinel as a platform and cross-sell.")
    found = check_engagement("project-kanpur", profile=swapped, tree=tree())
    assert len(found) == 1
    assert found[0].artifact == "hypothesis tree"
    assert "different thesis" in found[0].detail
    # The remedy has to say not to proceed, because every count still looks right.
    assert "Do not advance" in found[0].remedy


def test_a_deck_about_another_company_is_caught():
    deck = Deck(engagement_id="project-kanpur", created_by="Synthesizer",
                title="Commercial Due Diligence - Sentinel Secure Ltd")
    found = check_engagement("project-kanpur", profile=profile(), deck=deck)
    assert [f.artifact for f in found] == ["deck"]


def test_an_artifact_filed_under_the_wrong_engagement_is_caught():
    found = check_engagement("project-kanpur", tree=tree(engagement="project-sentinel"))
    assert [f.artifact for f in found] == ["hypothesis tree"]
    assert "project-sentinel" in found[0].detail


def test_whitespace_and_case_do_not_count_as_disagreement():
    """A false positive here would train people to ignore the banner."""
    assert check_engagement(
        "project-kanpur",
        profile=profile(thesis="Take Freshworks private at a premium."),
        tree=tree(thesis="  take freshworks   private at a  premium.  "),
    ) == []


def test_raise_if_incoherent_stops_the_pipeline():
    with pytest.raises(EngagementIncoherent) as excinfo:
        raise_if_incoherent(
            "project-kanpur",
            profile=profile(name="Sentinel Secure Ltd", thesis="Something else."),
            tree=tree())
    assert "disagree about which company" in str(excinfo.value)


def test_the_store_refuses_a_cross_engagement_write(isolated_settings):
    """The structural half: an artifact knows which engagement it belongs to."""
    from cdd_agent.state.store import Collection, StateStore

    store = StateStore()
    foreign = tree(engagement="project-sentinel")
    with pytest.raises(ValueError, match="refusing to file"):
        store.put("project-kanpur", Collection.HYPOTHESIS_TREE, "current", foreign,
                  agent="Thesis Architect")


def test_phase_2_refuses_to_build_a_request_for_the_wrong_company(context, request):
    """Phase 2 turns the tree into weeks of data requests, so it stops here."""
    from cdd_agent.agents.analyst import Analyst

    # Build the tree from the brief as it stood, then swap the brief underneath it -
    # the order the real incident happened in.
    built = request.getfixturevalue("tree")
    context.profile.thesis.one_sentence_thesis = "A completely different thesis."
    with pytest.raises(EngagementIncoherent):
        Analyst(context).generate_data_request(built, save=False)


def test_a_trailing_full_stop_is_not_a_different_thesis():
    """The first version of this check fired on exactly this, on a real engagement.

    A guardrail that cries wolf over punctuation teaches people to click past it,
    which costs more than the check saves.
    """
    assert check_engagement(
        "project-sentinel",
        profile=profile(thesis="Acquire Sentinel and cross-sell into the base",
                        engagement="project-sentinel"),
        tree=tree(thesis="Acquire Sentinel and cross-sell into the base.",
                  engagement="project-sentinel"),
    ) == []


def test_the_report_shows_where_the_two_actually_differ():
    """Two long theses that agree for a paragraph and differ at the end used to be
    reported as two identical-looking quotes, which reads as a broken checker."""
    shared = "Acquire the target as a platform and cross-sell a new product line into "
    found = check_engagement(
        "project-kanpur",
        profile=profile(thesis=shared + "the existing installed base"),
        tree=tree(thesis=shared + "an entirely different customer set"))
    assert len(found) == 1
    detail = found[0].detail
    assert "they agree for" in detail
    assert "installed base" in detail and "different customer set" in detail
