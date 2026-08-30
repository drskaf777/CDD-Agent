"""The Phase-1 routing rules (Checkpoint 4.1 s 2.3, s 2.5).

These are the tests that matter most for the ToT step, because the routing decisions -
prune, escalate, ask - are the ones with no model in the loop and therefore no excuse
for being wrong.
"""

from __future__ import annotations

import pytest

from cdd_agent.agents.thesis_architect.beam_search import (
    apply_override,
    build_result,
    prune_and_route,
)
from cdd_agent.schemas.hypothesis import (
    CriticScore,
    FourQuestionCheck,
    Hypothesis,
    HypothesisTree,
)


def _branch(branch_id: str, avg: float, four_q: bool = True) -> HypothesisTree:
    tree = HypothesisTree(
        engagement_id="e",
        created_by="test",
        branch_id=branch_id,
        framing_label=f"{branch_id}-led",
        root_thesis="thesis",
        hypotheses=[Hypothesis(id=f"{branch_id}-H1", statement="claim", depth=1)],
    )
    check = FourQuestionCheck(
        market_growing=four_q,
        target_keeps_winning=four_q,
        unit_economics_hold=four_q,
        what_breaks_the_deal=four_q,
    )
    tree.score = CriticScore(
        four_question=check,
        buyer_criteria_coverage=avg,
        four_question_alignment=avg,
        sub_sector_fit=avg,
        testability=avg,
        criterion_notes={"sub_sector_fit": "does not test NRR"},
    )
    return tree


def test_selects_the_highest_scoring_branch():
    branches = [_branch("growth", 4.5), _branch("margin", 3.2), _branch("risk", 3.6)]
    routing = prune_and_route(branches)
    assert routing.outcome == "selected"
    assert routing.selected_branch_id == "growth"


def test_four_question_failure_is_a_hard_prune_and_is_not_overridable():
    branches = [_branch("growth", 4.8, four_q=False), _branch("risk", 3.9)]
    routing = prune_and_route(branches)
    assert routing.selected_branch_id == "risk"

    failed = branches[0]
    assert failed.pruned
    # Scoring highest does not save a framing that cannot test all four questions.
    assert not failed.prunable_pending_override
    assert "four-question check failed" in failed.prune_reason

    result = build_result("e", branches, routing)
    with pytest.raises(ValueError, match="four-question hard check"):
        apply_override(result, "growth", approved_by="MD")


def test_soft_prune_is_recoverable_and_logs_the_driving_criterion():
    branches = [_branch("growth", 4.2), _branch("margin", 2.4)]
    prune_and_route(branches)
    pruned = branches[1]
    assert pruned.pruned and pruned.prunable_pending_override
    # Checkpoint 4.1 s 2.5: the pruning step logs which criterion drove the low score.
    assert "weakest criterion" in pruned.prune_reason
    assert "does not test NRR" in pruned.prune_reason


def test_soft_pruned_branch_can_be_recovered_by_human_override():
    branches = [_branch("growth", 4.2), _branch("margin", 2.4)]
    routing = prune_and_route(branches)
    result = build_result("e", branches, routing)
    updated = apply_override(result, "margin", approved_by="Partner")
    assert updated.selected_branch_id == "margin"
    assert updated.outcome == "selected"
    selected = updated.selected()
    assert selected.human_approved and not selected.pruned


def test_ties_within_the_band_go_to_the_user_not_a_reranker():
    branches = [_branch("growth", 4.10), _branch("margin", 3.80), _branch("risk", 2.0)]
    routing = prune_and_route(branches)
    assert routing.outcome == "tie_escalated"
    assert set(routing.tied_branch_ids) == {"growth", "margin"}
    assert routing.selected_branch_id is None


def test_scores_outside_the_tie_band_do_not_escalate():
    branches = [_branch("growth", 4.10), _branch("margin", 3.40)]
    assert prune_and_route(branches).outcome == "selected"


def test_all_pruned_asks_a_clarifying_question_rather_than_taking_the_least_bad():
    branches = [_branch("growth", 1.5), _branch("margin", 2.0), _branch("risk", 2.2)]
    routing = prune_and_route(branches)
    assert routing.outcome == "all_pruned_clarification_needed"
    assert routing.selected_branch_id is None
    assert "under-specified" in routing.clarifying_question


def test_all_pruned_on_four_questions_names_what_was_untestable():
    branches = [_branch(b, 4.5, four_q=False) for b in ("growth", "margin", "risk")]
    routing = prune_and_route(branches)
    assert routing.outcome == "all_pruned_clarification_needed"
    assert "whether the market itself is growing" in routing.clarifying_question


def test_routing_refuses_to_run_before_scoring():
    tree = _branch("growth", 4.0)
    tree.score = None
    with pytest.raises(ValueError, match="before it was scored"):
        prune_and_route([tree])
