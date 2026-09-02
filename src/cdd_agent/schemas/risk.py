"""Risk Register and Outstanding Information Log - the Phase-5 continuous artifact."""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from cdd_agent.schemas.common import Stamped


class RiskCategory(str, Enum):
    """The standing risk taxonomy, design specification s VII.A.

    Coverage of this enum per deal is an evaluation metric in its own right
    (Checkpoint 6.1: "the share of the standing risk taxonomy actually evaluated
    per deal, not just what the model happened to surface").
    """

    REVENUE_QUALITY = "Revenue quality & concentration"
    GROWTH_SUSTAINABILITY = "Growth sustainability"
    COMPETITIVE_DISRUPTION = "Competitive disruption"
    UNIT_ECONOMICS = "Unit economics / margin"
    KEY_PERSON = "Key-person / management"
    RETENTION = "Customer & employee retention"
    REGULATORY = "Regulatory / compliance"
    DATA_ROOM_INTEGRITY = "Data-room integrity"
    INTEGRATION_SYNERGY = "Integration / synergy (strategic buyers)"
    # Listed targets only. Screened conditionally so a private deal is never
    # marked down on coverage for categories that cannot apply to it.
    MARKET_EXPECTATIONS = "Public-market expectations (listed targets)"
    GOVERNANCE_CONTROL = "Governance & control rights (listed targets)"
    DEAL_COMPLETION = "Deal completion & approvals (listed targets)"


class RiskStatus(str, Enum):
    OPEN = "open"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    CARRIED_TO_CONFIRMATORY = "carried to confirmatory diligence"


class RiskItem(Stamped):
    id: str
    engagement_id: str
    category: RiskCategory
    description: str
    severity: int = Field(ge=1, le=5)
    likelihood: int = Field(ge=1, le=5)
    hypothesis_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: RiskStatus = RiskStatus.OPEN
    # Design spec s VIII: findings resting solely on management data are flagged here.
    management_data_only: bool = False

    @property
    def score(self) -> int:
        return self.severity * self.likelihood


class GapOwner(str, Enum):
    DEAL_TEAM = "deal team"
    MANAGEMENT = "management"
    THIRD_PARTY_EXPERT = "third-party expert"


class InformationGap(Stamped):
    """A specific, addressed follow-up request - never a generic "more data needed".

    Design spec s VII.B: each gap carries a suggested owner and a target-resolution
    date tied to the IC deadline captured at intake.
    """

    id: str
    engagement_id: str
    outline_section: Optional[int] = None
    hypothesis_id: Optional[str] = None
    request: str = Field(description="The specific artifact being asked for.")
    owner: GapOwner = GapOwner.MANAGEMENT
    target_close_date: Optional[_dt.date] = None
    blocking: bool = Field(default=False, description="True when it blocks a Tier-1 rating.")
    carried_to_confirmatory: bool = Field(
        default=False,
        description="Cannot be resolved pre-signing (e.g. reference calls restricted "
        "pre-LOI). Explicitly carried forward rather than silently dropped.",
    )
    resolved: bool = False


class RiskRegister(Stamped):
    engagement_id: str
    risks: list[RiskItem] = Field(default_factory=list)
    gaps: list[InformationGap] = Field(default_factory=list)
    # Set by the Risk Auditor when it finds multiple live versions of one source.
    source_conflicts: list[str] = Field(default_factory=list)

    def ranked(self) -> list[RiskItem]:
        return sorted(self.risks, key=lambda r: (-r.score, r.category.value))

    def open_blocking_gaps(self) -> list[InformationGap]:
        return [g for g in self.gaps if g.blocking and not g.resolved]

    def categories_evaluated(self) -> set[RiskCategory]:
        return {r.category for r in self.risks}

    def coverage(self, applicable: "list[RiskCategory] | None" = None) -> float:
        """Risk Register coverage metric (Checkpoint 6.1).

        `applicable` narrows the denominator - integration/synergy applies only to
        strategic buyers, so counting it against a sponsor deal understates coverage.
        The register cannot know the buyer type, so the caller supplies the set.
        """
        pool = list(applicable) if applicable else list(RiskCategory)
        evaluated = self.categories_evaluated()
        if not pool:
            return 1.0
        return len([c for c in pool if c in evaluated]) / len(pool)
