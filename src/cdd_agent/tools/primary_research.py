"""Independent Primary Research tool - Checkpoint 2.1, third tool category.

This addresses a retrieval limitation of a different kind: management-curated reference
calls produce satisfaction data running meaningfully higher than independently-recruited
samples, and a model cannot detect that bias from the transcripts alone. Routing
outreach through an independent-sourcing step, rather than accepting whatever contact
list the data room supplies, is what keeps the outside-in evidence standard real.

Two things this module does *not* do, deliberately:

* It does not contact anyone by itself. Commissioning a real interview is an outward-
  facing action; this returns a scoped, authorized request for the deal team to place,
  and records the methodology disclosure that Section 4 of the outline requires.
* It does not accept a management-supplied contact list as an independent sample. A
  request built from one is tagged as such and loses its independence weighting.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

from cdd_agent.guardrails.authorization import (
    AgentRole,
    ToolAuthorization,
    ToolName,
)
from cdd_agent.schemas.common import Citation, SourceKind


@dataclass
class ResearchRequest:
    """A commissioned, authorized interview programme."""

    id: str
    hypothesis_id: str
    contact_type: str            # customer | expert | former employee
    objective: str
    target_sample_size: int
    screening_criteria: list[str] = field(default_factory=list)
    independently_sourced: bool = True
    excludes_management_list: bool = True
    authorized_note: str = ""
    requested_at: _dt.datetime = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    def methodology_disclosure(self) -> str:
        """The disclosure Section 4 of the enhanced outline requires."""
        source = (
            "independently recruited; management-supplied contact list excluded"
            if self.independently_sourced and self.excludes_management_list
            else "sourced from a management-supplied list - NOT independent"
        )
        return (
            f"n={self.target_sample_size} {self.contact_type} interviews, {source}. "
            f"Screening: {'; '.join(self.screening_criteria) or 'none stated'}."
        )


@dataclass
class InterviewNote:
    """A returned interview, ready to enter the Evidence Matrix as independent evidence."""

    request_id: str
    respondent_label: str        # anonymised; no personal data enters the store
    contact_type: str
    date: _dt.date
    summary: str
    independently_sourced: bool = True

    def to_citation(self) -> Citation:
        return Citation(
            source_kind=SourceKind.PRIMARY_RESEARCH,
            source_file=f"interview::{self.request_id}",
            locator=f"{self.respondent_label} ({self.contact_type})",
            document_date=self.date,
            quoted_text=self.summary[:2000],
        )


class PrimaryResearchTool:
    """Commissions independently-sourced customer and expert interviews.

    Authorization is checked on every call. A blocked call raises; it does not return
    a degraded result, because a quietly-skipped interview would leave the deck looking
    outside-in when it is not.
    """

    name = "primary_research"
    description = (
        "Commission independently-sourced customer or industry-expert interviews to "
        "test a hypothesis. Subject to intake Category F access constraints."
    )

    def __init__(
        self,
        authorization: ToolAuthorization,
        role: AgentRole = AgentRole.ANALYST,
    ) -> None:
        self.authorization = authorization
        self.role = role
        self._counter = 0

    def commission(
        self,
        *,
        hypothesis_id: str,
        contact_type: str,
        objective: str,
        target_sample_size: int = 8,
        screening_criteria: Optional[list[str]] = None,
        is_top5_customer: bool = False,
        from_management_list: bool = False,
    ) -> ResearchRequest:
        decision = self.authorization.check(
            self.role,
            ToolName.PRIMARY_RESEARCH,
            contact_type=contact_type,
            is_top5_customer=is_top5_customer,
        )
        decision.raise_if_denied()

        self._counter += 1
        return ResearchRequest(
            id=f"PR-{self._counter:03d}",
            hypothesis_id=hypothesis_id,
            contact_type=contact_type,
            objective=objective,
            target_sample_size=target_sample_size,
            screening_criteria=screening_criteria or [],
            independently_sourced=not from_management_list,
            excludes_management_list=not from_management_list,
            authorized_note=decision.reason,
        )

    def can_commission(self, contact_type: str, is_top5_customer: bool = False) -> bool:
        return self.authorization.check(
            self.role,
            ToolName.PRIMARY_RESEARCH,
            contact_type=contact_type,
            is_top5_customer=is_top5_customer,
        ).allowed
