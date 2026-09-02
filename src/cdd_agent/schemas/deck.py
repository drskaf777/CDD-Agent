"""Draft presentation artifacts.

The output contract lives in the schema: a Claim cannot be constructed without a
confidence tag, and cannot carry a non-"No Data" tag without a citation. That makes
an unqualified assertion a validation error rather than something a reviewer has to
catch by reading (Checkpoint 6.1, output constraints).
"""

from __future__ import annotations

from enum import Enum
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


class ExhibitStatus(str, Enum):
    """Where an exhibit's content came from - the same discipline as a Claim's tag.

    A standard CDD deck has a known set of exhibits. Rendering one with invented
    numbers because the data room lacked the inputs is precisely the failure this
    project exists to prevent, so an exhibit that cannot be built says so and carries
    the request that would build it.
    """

    COMPUTED = "computed"      # arithmetic over parsed data-room tables
    EVIDENCED = "evidenced"    # assembled from cited evidence in the matrix
    GAP = "gap"                # inputs absent - rendered as a request, not a guess


class Series(BaseModel):
    """One numeric series for a chart. Labels and values stay aligned by index."""

    name: str = ""
    labels: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    unit: str = ""

    @model_validator(mode="after")
    def _aligned(self) -> "Series":
        if self.labels and len(self.labels) != len(self.values):
            raise ValueError(
                f"series {self.name!r} has {len(self.labels)} labels for "
                f"{len(self.values)} values"
            )
        return self


class Exhibit(BaseModel):
    """A computed table or chart spec. Numbers come from the computation tool."""

    key: str = ""
    title: str
    kind: str = Field(
        description="table | bar | line | scatter | waterfall | heatmap | matrix"
    )
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    note: str = ""
    # Chart data, when the exhibit is drawable rather than only tabular.
    series: list[Series] = Field(default_factory=list)
    status: ExhibitStatus = ExhibitStatus.EVIDENCED
    # What is missing, and what would close it. Set only when status is GAP.
    gap_request: str = ""

    @model_validator(mode="after")
    def _gap_is_explained(self) -> "Exhibit":
        if self.status is ExhibitStatus.GAP and not self.gap_request:
            raise ValueError(
                f"exhibit {self.title!r} is a gap but names no request that would close it"
            )
        if self.status is not ExhibitStatus.GAP and self.gap_request:
            raise ValueError(
                f"exhibit {self.title!r} carries a gap request but is not a gap"
            )
        return self


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
        """Share of *assertive* claims carrying a citation (Checkpoint 6.1 metric).

        No Data claims are excluded from the denominator, matching
        `cdd_agent.evaluation.metrics.groundedness`. A logged gap is a correct output,
        not an ungrounded claim; counting it as a miss would create pressure to assert
        rather than to log, which is exactly backwards. Two definitions of one metric
        is worse than either definition.
        """
        assertive = [c for c in self.all_claims() if c.tag is not ConfidenceTag.NO_DATA]
        if not assertive:
            return 1.0
        return sum(1 for c in assertive if c.citations) / len(assertive)
