"""Corrective layer: the five human-intervention triggers (Checkpoint 6.1).

The triggers are deliberately narrow. Over-triggering breeds alert fatigue and a
reviewer who rubber-stamps instead of reviewing, so this module implements exactly the
five points the checkpoint names and nothing else:

1. Any Tier-1 hypothesis below Partially Confirmed when Synthesis is requested.
2. A Phase-1 tie within 0.5 points, or all three candidates pruned.
3. A Risk Auditor conflict flag (multiple live versions of one source).
4. Any action that would exceed intake's NDA/access constraints - a hard block.
5. Every final go/no-go recommendation, which is always human-reviewed.

Trigger 5 is unconditional by construction: it is not a check that can pass.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from cdd_agent.schemas.common import ConfidenceTag
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree, ThesisSearchResult
from cdd_agent.schemas.risk import RiskRegister


class Trigger(str, Enum):
    WEAK_TIER1_EVIDENCE = "weak_tier1_evidence"
    PHASE1_TIE_OR_ALL_PRUNED = "phase1_tie_or_all_pruned"
    SOURCE_CONFLICT = "source_conflict"
    ACCESS_BOUNDARY = "access_boundary"
    FINAL_RECOMMENDATION = "final_recommendation"


@dataclass
class Escalation:
    trigger: Trigger
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    blocking: bool = True
    raised_by: str = "Controller"
    raised_at: _dt.datetime = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    def render(self) -> str:
        mark = "BLOCKS" if self.blocking else "notice"
        return f"[{mark}] {self.trigger.value}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger.value,
            "message": self.message,
            "detail": self.detail,
            "blocking": self.blocking,
            "raised_by": self.raised_by,
            "raised_at": self.raised_at.isoformat(),
        }


def check_tier1_evidence(
    tree: HypothesisTree,
    matrix: EvidenceMatrix,
    register: Optional[RiskRegister] = None,
) -> list[Escalation]:
    """Trigger 1. Run when Synthesis is requested, not continuously.

    A Tier-1 hypothesis with No Data is acceptable *only* if it carries an explicit,
    dated information gap - the design's stopping condition, restated.
    """
    dated = {
        g.hypothesis_id
        for g in (register.gaps if register else [])
        if g.hypothesis_id and g.target_close_date is not None
    }
    out: list[Escalation] = []
    for h in tree.tier_1():
        if matrix.rating(h.id) is ConfidenceTag.NO_DATA and h.id not in dated:
            out.append(
                Escalation(
                    trigger=Trigger.WEAK_TIER1_EVIDENCE,
                    message=(
                        f"Tier-1 hypothesis {h.id} is below Partially Confirmed with no "
                        f"dated information gap: {h.statement}"
                    ),
                    detail={"hypothesis_id": h.id, "statement": h.statement},
                )
            )
    return out


def check_phase1(result: ThesisSearchResult) -> list[Escalation]:
    """Trigger 2. Ties are not auto-resolved by reranking; all-pruned asks a question."""
    if result.outcome == "tie_escalated":
        tied = ", ".join(result.tied_branch_ids)
        return [
            Escalation(
                trigger=Trigger.PHASE1_TIE_OR_ALL_PRUNED,
                message=(
                    f"Candidate framings {tied} score within 0.5 points. Choose one "
                    "rather than letting the Controller pick."
                ),
                detail={"tied_branch_ids": result.tied_branch_ids},
            )
        ]
    if result.outcome == "all_pruned_clarification_needed":
        return [
            Escalation(
                trigger=Trigger.PHASE1_TIE_OR_ALL_PRUNED,
                message=(
                    "All three framings were pruned. The thesis appears under-specified: "
                    f"{result.clarifying_question}"
                ),
                detail={"clarifying_question": result.clarifying_question},
            )
        ]
    return []


def check_source_conflicts(register: RiskRegister) -> list[Escalation]:
    """Trigger 3. Raised by the Risk Auditor, surfaced by the Controller."""
    return [
        Escalation(
            trigger=Trigger.SOURCE_CONFLICT,
            message=f"Conflicting versions of one source: {conflict}",
            detail={"conflict": conflict},
            raised_by="Risk Auditor",
        )
        for conflict in register.source_conflicts
    ]


def access_boundary(reason: str, attempted: str) -> Escalation:
    """Trigger 4. A hard block - recorded, never downgraded to a warning."""
    return Escalation(
        trigger=Trigger.ACCESS_BOUNDARY,
        message=f"Blocked: {attempted}. {reason}",
        detail={"attempted_action": attempted, "reason": reason},
        blocking=True,
    )


def final_recommendation_review(engagement_id: str) -> Escalation:
    """Trigger 5. Unconditional: every go/no-go is human-reviewed before the IC."""
    return Escalation(
        trigger=Trigger.FINAL_RECOMMENDATION,
        message=(
            "Draft complete. The go/no-go recommendation requires partner/MD review "
            "before it reaches the investment committee."
        ),
        detail={"engagement_id": engagement_id},
        blocking=True,
    )
