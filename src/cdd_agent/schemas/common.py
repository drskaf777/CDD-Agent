"""Vocabulary shared across every artifact.

The four-way confidence schema is the spine of the whole system: the ReAct
Observation step emits it (Checkpoint 2.1), retrieval maps its own failure mode
onto it (Checkpoint 3.1 s 5), the ToT placeholders use it (Checkpoint 4.1 s 2.2),
and the Synthesizer's output contract requires it on every claim (Checkpoint 6.1).
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfidenceTag(str, Enum):
    CONFIRMED = "Confirmed"
    PARTIALLY_CONFIRMED = "Partially Confirmed"
    CONTRADICTED = "Contradicted"
    NO_DATA = "No Data"

    @property
    def is_at_least_partial(self) -> bool:
        """Synthesis gate: Tier-1 hypotheses must clear this or carry a dated gap."""
        return self in (ConfidenceTag.CONFIRMED, ConfidenceTag.PARTIALLY_CONFIRMED,
                        ConfidenceTag.CONTRADICTED)


class Tier(int, Enum):
    """Data-request tiering, design specification s V."""

    DEAL_CRITICAL = 1   # blocking: no hypothesis is Confirmed/Contradicted without it
    DEPTH_BUILDING = 2  # needed for presentation-ready depth
    ENRICHMENT = 3      # strengthens an exhibit, never blocking


class SourceKind(str, Enum):
    """Where a piece of evidence came from.

    Kept distinct because the design weights them differently: management-supplied
    material is what is being tested, so it can never be the sole support for a
    Confirmed tag (design spec s VIII, bias disclosure).
    """

    DATA_ROOM = "data_room"                # management-supplied, unstructured
    STRUCTURED_DATA = "structured_data"    # management-supplied, parsed to schema
    COMPUTATION = "computation"            # derived by the computation tool
    PRIMARY_RESEARCH = "primary_research"  # independently sourced interviews
    KNOWLEDGE_BASE = "knowledge_base"      # cross-engagement / external market data
    # --- Listed targets ---
    PUBLIC_FILING = "public_filing"        # 10-K/20-F, 10-Q, 8-K, proxy, transcript
    SELL_SIDE_RESEARCH = "sell_side_research"  # broker notes, published consensus
    INTAKE = "intake"                      # stated by the buyer in the Deal Profile

    @property
    def is_management_supplied(self) -> bool:
        # A filing is still the issuer's own account of itself. Audit and officer
        # certification raise its evidential weight - see is_attested - but they do
        # not make it independent of the party whose plan is being tested.
        return self in (SourceKind.DATA_ROOM, SourceKind.STRUCTURED_DATA,
                        SourceKind.PUBLIC_FILING)

    @property
    def is_independent(self) -> bool:
        """Sources that can triangulate a management claim.

        Sell-side research is deliberately excluded. Analysts build their models from
        company guidance and management access, so consensus agreeing with the plan
        is mostly an echo of it. Treating it as corroboration would manufacture
        triangulation out of the same source twice - the exact failure the four-way
        confidence schema exists to prevent.
        """
        return self in (SourceKind.PRIMARY_RESEARCH, SourceKind.KNOWLEDGE_BASE)

    @property
    def is_buyer_asserted(self) -> bool:
        """Stated by the client at intake and not yet verified against anything.

        Traceable - the Deal Profile Brief is a stored artifact with an author and a
        timestamp - but corroborated by nobody, so it counts neither as management
        data nor as independent triangulation.
        """
        return self is SourceKind.INTAKE

    @property
    def is_attested(self) -> bool:
        """Filed under audit and officer certification, with legal liability attached.

        Weaker than independent, stronger than a board deck: the classic misses -
        segment mix, non-GAAP adjustments - live inside filings, but nobody signs a
        10-K casually.
        """
        return self is SourceKind.PUBLIC_FILING

    @property
    def is_public_record(self) -> bool:
        """Available without the data room, so it creates no MNPI to read."""
        return self in (SourceKind.PUBLIC_FILING, SourceKind.SELL_SIDE_RESEARCH,
                        SourceKind.KNOWLEDGE_BASE)


class Citation(BaseModel):
    """A pointer back to the exact passage or computation a claim rests on.

    Persisted to the retrieval citation log in Long-Term Memory (Architecture
    v6.7, slide 1) so a later session can see not just what was concluded but
    which chunk it was grounded in.
    """

    model_config = ConfigDict(frozen=True)

    source_kind: SourceKind
    source_file: str
    locator: str = Field(description="Clause, slide number, sheet!range, or Q&A turn.")
    chunk_id: Optional[str] = None
    document_date: Optional[_dt.date] = None
    document_tier: Optional[Tier] = None
    quoted_text: str = Field(default="", max_length=2000)
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def short(self) -> str:
        date = f", {self.document_date.isoformat()}" if self.document_date else ""
        return f"{self.source_file} ({self.locator}{date})"


class OutlineSection(BaseModel):
    """One section of the enhanced master outline (design spec s IV.A)."""

    model_config = ConfigDict(frozen=True)

    number: int
    title: str
    key_elements: tuple[str, ...]
    is_new: bool = False
    is_enhanced: bool = False


def utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class Stamped(BaseModel):
    """Mixin for artifacts that must be attributable and timestamped.

    Guardrail, Checkpoint 6.1: every write to the shared state store is
    timestamped and attributed to an agent, making the run auditable after the fact.
    """

    created_at: _dt.datetime = Field(default_factory=utcnow)
    created_by: str = Field(default="system", description="Agent role that wrote this.")

    @field_validator("created_by")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("created_by must name the writing agent")
        return v
