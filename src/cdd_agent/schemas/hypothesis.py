"""Hypothesis Tree artifacts and the Tree-of-Thought search record.

Checkpoint 4.1: a "thought" is one candidate hypothesis; a node is a hypothesis at a
position; a branch is one complete candidate decomposition. Depth 0 is the root thesis,
depth 1 the Tier-1 framing (3-5 hypotheses), depth 2 the supporting assumptions.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from cdd_agent.schemas.common import ConfidenceTag, Stamped, Tier


class Hypothesis(BaseModel):
    """A testable claim plus its rationale, carrying an Evidence-Matrix placeholder."""

    id: str
    statement: str = Field(description="A testable claim, not a topic heading.")
    rationale: str = ""
    tier: Tier = Tier.DEAL_CRITICAL
    depth: int = Field(default=1, ge=0, le=2)
    parent_id: Optional[str] = None
    # Which of the four questions this hypothesis tests. Used by the Critic's hard check.
    four_question_ref: Optional[str] = None
    confidence: ConfidenceTag = ConfidenceTag.NO_DATA
    evidence_ids: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(
        default_factory=list,
        description="What would move this to Confirmed or Contradicted. Seeds Phase 2.",
    )

    @property
    def is_tier_1(self) -> bool:
        return self.tier is Tier.DEAL_CRITICAL


class FourQuestionCheck(BaseModel):
    """Bain's four-question test as a hard pass/fail constraint (Checkpoint 4.1 s 2.3).

    A framing that leaves any question unmapped is pruned outright and is *not*
    recoverable by user override - unlike a soft-criterion prune.
    """

    market_growing: bool = False
    target_keeps_winning: bool = False
    unit_economics_hold: bool = False
    what_breaks_the_deal: bool = False

    @property
    def passed(self) -> bool:
        return all(
            (self.market_growing, self.target_keeps_winning,
             self.unit_economics_hold, self.what_breaks_the_deal)
        )

    def unmapped(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if not v]


class CriticScore(BaseModel):
    """The Critic's assessment of one branch. Soft criteria are scored 1-5."""

    four_question: FourQuestionCheck = Field(default_factory=FourQuestionCheck)
    buyer_criteria_coverage: float = Field(default=0.0, ge=0.0, le=5.0)
    four_question_alignment: float = Field(default=0.0, ge=0.0, le=5.0)
    sub_sector_fit: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description="Checked against the Knowledge-Base Index, not unaided judgment.",
    )
    testability: float = Field(default=0.0, ge=0.0, le=5.0)
    notes: str = ""
    # Recorded so a prune is explainable, per the Checkpoint 4.1 s 2.5 mitigation.
    criterion_notes: dict[str, str] = Field(default_factory=dict)

    @property
    def average(self) -> float:
        vals = (self.buyer_criteria_coverage, self.four_question_alignment,
                self.sub_sector_fit, self.testability)
        return round(sum(vals) / len(vals), 3)

    def weakest_criterion(self) -> str:
        scores = {
            "buyer_criteria_coverage": self.buyer_criteria_coverage,
            "four_question_alignment": self.four_question_alignment,
            "sub_sector_fit": self.sub_sector_fit,
            "testability": self.testability,
        }
        return min(scores, key=lambda k: scores[k])


class HypothesisTree(Stamped):
    """One complete candidate decomposition - a branch of the ToT search."""

    engagement_id: str
    branch_id: str
    framing_label: str = Field(description="e.g. growth-led, margin-led, risk-led.")
    root_thesis: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)

    score: Optional[CriticScore] = None
    selected: bool = False
    pruned: bool = False
    prune_reason: Optional[str] = None
    # Checkpoint 4.1 s 2.5: soft-criterion prunes stay recoverable by human review;
    # a four-question failure is a hard prune and is not marked recoverable.
    prunable_pending_override: bool = False
    human_approved: bool = False

    def tier_1(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.is_tier_1 and h.depth == 1]

    def children_of(self, hypothesis_id: str) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.parent_id == hypothesis_id]

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return next((h for h in self.hypotheses if h.id == hypothesis_id), None)

    @model_validator(mode="after")
    def _check_shape(self) -> "HypothesisTree":
        for h in self.hypotheses:
            if h.parent_id and not any(o.id == h.parent_id for o in self.hypotheses):
                raise ValueError(f"hypothesis {h.id} references missing parent {h.parent_id}")
        return self


class ThesisSearchResult(Stamped):
    """The full record of one Phase-1 beam search.

    Pruned branches persist alongside the winner, tagged with their exclusion reason,
    so an unconventional-but-better framing can be recovered by review rather than
    disappearing silently (Checkpoint 4.1 s 2.5).
    """

    engagement_id: str
    branches: list[HypothesisTree] = Field(default_factory=list)
    selected_branch_id: Optional[str] = None
    outcome: str = Field(
        default="selected",
        description="selected | tie_escalated | all_pruned_clarification_needed",
    )
    escalation_message: Optional[str] = None
    tied_branch_ids: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None

    def selected(self) -> Optional[HypothesisTree]:
        if not self.selected_branch_id:
            return None
        return next(
            (b for b in self.branches if b.branch_id == self.selected_branch_id), None
        )

    def requires_human(self) -> bool:
        return self.outcome != "selected"
