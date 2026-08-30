"""Decision Maker / Controller - beam search over candidate framings.

Checkpoint 4.1 s 2.3 chooses beam search (width 3) over BFS, DFS, and Monte Carlo, and
the reasons shape this implementation:

* BFS would expand every framing to full depth before comparing, wasting cost when
  evaluation is qualitative rather than a cheap function. Here, generation and scoring
  happen once per branch and nothing is deepened.
* DFS would commit to the first path - the premature commitment this step exists to avoid.
* Monte Carlo assumes cheap rollouts against a reward signal, and no simulator exists
  for "would this framing have found the real issues".

Cost is therefore fixed and one-time: 3 generations plus 3 evaluations per engagement,
not a recurring per-step cost like the ReAct loop's retrieval calls.

The routing rules are deterministic - this is orchestration logic, not a persona:
prune below threshold, escalate ties, ask a clarifying question when everything is pruned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cdd_agent.config import get_settings
from cdd_agent.schemas.hypothesis import HypothesisTree, ThesisSearchResult


@dataclass
class Routing:
    """Where the Controller sends the search after scoring."""

    outcome: str                       # selected | tie_escalated | all_pruned_...
    selected_branch_id: Optional[str] = None
    tied_branch_ids: list[str] = None  # type: ignore[assignment]
    clarifying_question: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tied_branch_ids is None:
            self.tied_branch_ids = []


def prune_and_route(branches: list[HypothesisTree]) -> Routing:
    """Apply the pruning threshold and route the result. Pure function, no I/O.

    Every branch must already carry a CriticScore.
    """
    settings = get_settings()
    if not branches:
        return Routing(
            outcome="all_pruned_clarification_needed",
            clarifying_question="No candidate framings were generated.",
        )

    survivors: list[HypothesisTree] = []
    for branch in branches:
        score = branch.score
        if score is None:
            raise ValueError(f"branch {branch.branch_id} was routed before it was scored")

        if not score.four_question.passed:
            unmapped = ", ".join(score.four_question.unmapped())
            branch.pruned = True
            # A hard-check failure is NOT recoverable by override: the four-question
            # test is a constraint on what counts as a diligence plan at all, not a
            # matter of taste (Checkpoint 4.1 s 2.5).
            branch.prunable_pending_override = False
            branch.prune_reason = (
                f"four-question check failed - unmapped: {unmapped}"
            )
            continue

        if score.average < settings.prune_threshold:
            branch.pruned = True
            # A soft-criterion prune stays recoverable, with the driving criterion
            # logged, so an unconventional but better framing can be recovered by
            # review instead of disappearing silently.
            branch.prunable_pending_override = True
            weakest = score.weakest_criterion()
            branch.prune_reason = (
                f"average {score.average:.2f} below {settings.prune_threshold:.1f} "
                f"threshold; weakest criterion: {weakest} "
                f"({score.criterion_notes.get(weakest, 'no note')})"
            )
            continue

        survivors.append(branch)

    if not survivors:
        return Routing(
            outcome="all_pruned_clarification_needed",
            clarifying_question=_clarifying_question(branches),
        )

    survivors.sort(key=lambda b: b.score.average, reverse=True)  # type: ignore[union-attr]
    best = survivors[0]
    tied = [
        b
        for b in survivors
        if abs(b.score.average - best.score.average) <= settings.tie_band  # type: ignore[union-attr]
    ]
    if len(tied) > 1:
        # Ties are not auto-resolved by reranking; both go to the deal team.
        return Routing(
            outcome="tie_escalated",
            tied_branch_ids=[b.branch_id for b in tied],
        )

    best.selected = True
    return Routing(outcome="selected", selected_branch_id=best.branch_id)


def _clarifying_question(branches: list[HypothesisTree]) -> str:
    """When everything is pruned, ask - do not settle for the least-bad option.

    Checkpoint 4.1 s 2.3: "If all three fail, the system doesn't default to the
    least-bad option; it flags the thesis as under-specified and returns a clarifying
    question."
    """
    unmapped: set[str] = set()
    for b in branches:
        if b.score and not b.score.four_question.passed:
            unmapped |= set(b.score.four_question.unmapped())

    if unmapped:
        readable = {
            "market_growing": "whether the market itself is growing",
            "target_keeps_winning": "how the target keeps winning share",
            "unit_economics_hold": "how the unit economics behave",
            "what_breaks_the_deal": "what would break the deal",
        }
        missing = ", ".join(readable.get(u, u) for u in sorted(unmapped))
        return (
            "Every candidate framing failed the four-question screen on the same "
            f"ground: none of them could test {missing}. The thesis as stated does not "
            "say enough to build a testable plan around. What specifically is the "
            "value-creation mechanism, and which assumption in the model is most "
            "load-bearing?"
        )
    return (
        "Every candidate framing scored below the quality threshold. The thesis appears "
        "under-specified: what is the value-creation mechanism, which two or three model "
        "assumptions is the recommendation most sensitive to, and what does this buyer "
        "most need to be true?"
    )


def build_result(
    engagement_id: str, branches: list[HypothesisTree], routing: Routing
) -> ThesisSearchResult:
    """Assemble the persisted record. Pruned branches are retained with their reasons."""
    return ThesisSearchResult(
        engagement_id=engagement_id,
        created_by="Thesis Architect / Controller",
        branches=branches,
        selected_branch_id=routing.selected_branch_id,
        outcome=routing.outcome,
        tied_branch_ids=routing.tied_branch_ids,
        clarifying_question=routing.clarifying_question,
        escalation_message=_escalation_message(routing),
    )


def _escalation_message(routing: Routing) -> Optional[str]:
    if routing.outcome == "tie_escalated":
        return (
            f"Framings {', '.join(routing.tied_branch_ids)} scored within the tie band. "
            "Choose one before Phase 2 data requests are generated."
        )
    if routing.outcome == "all_pruned_clarification_needed":
        return routing.clarifying_question
    return None


def apply_override(
    result: ThesisSearchResult, branch_id: str, *, approved_by: str
) -> ThesisSearchResult:
    """Recover a soft-pruned branch, or resolve a tie, by human choice.

    Refuses to revive a branch pruned on the four-question hard check - that prune is
    not a matter of preference.
    """
    branch = next((b for b in result.branches if b.branch_id == branch_id), None)
    if branch is None:
        raise ValueError(f"no branch {branch_id!r} in this search")
    if branch.pruned and not branch.prunable_pending_override:
        raise ValueError(
            f"branch {branch_id!r} was pruned on the four-question hard check "
            f"({branch.prune_reason}) and cannot be selected by override"
        )
    for b in result.branches:
        b.selected = False
    branch.pruned = False
    branch.selected = True
    branch.human_approved = True
    branch.created_by = f"{branch.created_by} (override by {approved_by})"
    result.selected_branch_id = branch_id
    result.outcome = "selected"
    result.escalation_message = None
    return result
