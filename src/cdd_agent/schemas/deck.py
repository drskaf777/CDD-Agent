"""Draft presentation artifacts.

The output contract lives in the schema: a Claim cannot be constructed without a
confidence tag, and cannot carry a non-"No Data" tag without a citation. That makes
an unqualified assertion a validation error rather than something a reviewer has to
catch by reading (Checkpoint 6.1, output constraints).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from cdd_agent.schemas.common import Citation, ConfidenceTag, Stamped


class Claim(BaseModel):
    text: str
    tag: ConfidenceTag
    citations: list[Citation] = Field(default_factory=list)
    hypothesis_id: Optional[str] = None
    management_data_only: bool = False

    @model_validator(mode="after")
    def _must_be_grounded(self) -> "Claim":
        if self.tag is not ConfidenceTag.NO_DATA and not self.citations:
            raise ValueError(f"claim without a citation: {self.text[:80]!r}")
        return self

    def render(self) -> str:
        cites = "; ".join(c.short() for c in self.citations) or "no source - gap logged"
        flag = " [management data only]" if self.management_data_only else ""
        return f"{self.text} [{self.tag.value}]{flag} ({cites})"


class Exhibit(BaseModel):
    """A computed table or chart spec. Numbers come from the computation tool."""

    title: str
    kind: str = Field(description="table | bridge | cohort | sensitivity | matrix")
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    note: str = ""


class Slide(BaseModel):
    section_number: int
    section_title: str
    so_what_headline: str = Field(
        description="An evidence-backed assertion, not a topic label."
    )
    claims: list[Claim] = Field(default_factory=list)
    exhibits: list[Exhibit] = Field(default_factory=list)


class Deck(Stamped):
    engagement_id: str
    title: str
    draft_notice: str = (
        "DRAFT - structured working draft for partner/MD review. Not an IC "
        "recommendation. Every go/no-go is human-reviewed before it reaches the IC."
    )
    slides: list[Slide] = Field(default_factory=list)

    def all_claims(self) -> list[Claim]:
        return [c for s in self.slides for c in s.claims]

    def groundedness(self) -> float:
        """Share of claims carrying at least one citation (Checkpoint 6.1 metric)."""
        claims = self.all_claims()
        if not claims:
            return 1.0
        return sum(1 for c in claims if c.citations) / len(claims)
