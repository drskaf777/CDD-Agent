"""Evidence Matrix: hypothesis -> data -> confidence rating.

The Observation step of the ReAct loop writes here. The tag, not the raw tool output,
is what carries forward (Checkpoint 2.1).
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from pydantic import Field, model_validator

from cdd_agent.schemas.common import Citation, ConfidenceTag, SourceKind, Stamped


class EvidenceItem(Stamped):
    """One observation, tagged back to the hypothesis that motivated it."""

    id: str
    engagement_id: str
    hypothesis_id: str
    claim: str
    tag: ConfidenceTag
    citations: list[Citation] = Field(default_factory=list)
    source_kind: SourceKind
    query: Optional[str] = Field(default=None, description="The Action step that produced it.")
    superseded_by: Optional[str] = None
    conflicts_with: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _grounding(self) -> "EvidenceItem":
        """No-Data is the only tag that may stand without a citation.

        This is the schema-level half of the output constraint in Checkpoint 6.1:
        an unqualified assertion is a schema violation, not a style choice.
        """
        if self.tag is not ConfidenceTag.NO_DATA and not self.citations:
            raise ValueError(
                f"evidence {self.id} tagged {self.tag.value} without a citation"
            )
        return self

    @property
    def is_independent(self) -> bool:
        return self.source_kind.is_independent


class EvidenceMatrix(Stamped):
    engagement_id: str
    items: list[EvidenceItem] = Field(default_factory=list)

    def for_hypothesis(self, hypothesis_id: str) -> list[EvidenceItem]:
        return [i for i in self.items if i.hypothesis_id == hypothesis_id]

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def rating(self, hypothesis_id: str) -> ConfidenceTag:
        """Roll evidence up to a single tag for a hypothesis.

        Contradicted dominates: a contradicting finding is the most decision-relevant
        state and must not be averaged away by supporting items. Otherwise the strongest
        tag present wins, except that management-supplied evidence alone cannot reach
        Confirmed - it degrades to Partially Confirmed (design spec s VIII, bias
        disclosure; the outside-in evidence standard in s I).
        """
        items = self.for_hypothesis(hypothesis_id)
        if not items:
            return ConfidenceTag.NO_DATA
        tags = {i.tag for i in items}
        if ConfidenceTag.CONTRADICTED in tags:
            return ConfidenceTag.CONTRADICTED
        if ConfidenceTag.CONFIRMED in tags:
            supporting = [i for i in items if i.tag is ConfidenceTag.CONFIRMED]
            if any(i.is_independent for i in supporting):
                return ConfidenceTag.CONFIRMED
            return ConfidenceTag.PARTIALLY_CONFIRMED
        if ConfidenceTag.PARTIALLY_CONFIRMED in tags:
            return ConfidenceTag.PARTIALLY_CONFIRMED
        return ConfidenceTag.NO_DATA

    def triangulated(self, hypothesis_id: str) -> bool:
        """True when at least one independent source supports the hypothesis."""
        return any(i.is_independent for i in self.for_hypothesis(hypothesis_id))

    def latest_citation_date(self, hypothesis_id: str) -> Optional[_dt.date]:
        dates = [
            c.document_date
            for i in self.for_hypothesis(hypothesis_id)
            for c in i.citations
            if c.document_date
        ]
        return max(dates) if dates else None
