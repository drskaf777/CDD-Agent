"""The two RAG tools - Checkpoint 3.1 s 2 and Architecture v6.7 slide 1.

* ``DocumentRetrievalTool`` queries the engagement-scoped Data-Room Index.
* ``MarketSearchTool`` queries the cross-engagement Knowledge-Base Index.

Both return top-k chunks with source citations directly into the Observation step, and
both write the retrieved citation to the citation log so a later session can see which
passage a conclusion was grounded in.

Neither tool ever returns a bare below-floor match. Checkpoint 3.1 s 5 maps that case
onto the four-way schema instead: an explicit, auditable No Data beats a
plausible-looking wrong answer with a citation attached to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from cdd_agent.retrieval.indexes import (
    DataRoomIndex,
    KnowledgeBaseIndex,
    RetrievalResult,
)
from cdd_agent.schemas.common import Citation, ConfidenceTag, SourceKind, Tier


@dataclass
class RetrievalObservation:
    """What the Observation step receives: chunks, citations, and a provisional tag."""

    query: str
    citations: list[Citation]
    passages: list[str]
    provisional_tag: ConfidenceTag
    note: str = ""
    superseded_filtered: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.superseded_filtered is None:
            self.superseded_filtered = []

    def render(self) -> str:
        if not self.citations:
            return f"No Data - {self.note}"
        lines = [f"{len(self.citations)} passage(s) retrieved for {self.query!r}:"]
        # strict: a citation and its passage are a pair. Zipping to the shortest
        # would print one chunk under another chunk's citation, which is precisely
        # the mis-attribution this system exists to prevent.
        for cite, passage in zip(self.citations, self.passages, strict=True):
            snippet = " ".join(passage.split())[:300]
            lines.append(f"  - {cite.short()} [sim {cite.similarity:.2f}]: {snippet}")
        if self.superseded_filtered:
            lines.append(f"  (filtered as superseded: {', '.join(self.superseded_filtered)})")
        return "\n".join(lines)


def _to_observation(
    result: RetrievalResult, source_kind: SourceKind
) -> RetrievalObservation:
    if result.is_empty:
        note = (
            "every candidate scored below the similarity floor - tagged No Data rather "
            "than returned as a low-confidence match"
            if result.below_floor
            else "no matching passages in the index"
        )
        return RetrievalObservation(
            query=result.query,
            citations=[],
            passages=[],
            provisional_tag=ConfidenceTag.NO_DATA,
            note=note,
            superseded_filtered=result.filtered_superseded,
        )
    return RetrievalObservation(
        query=result.query,
        citations=[c.to_citation(source_kind) for c in result.chunks],
        passages=[c.text for c in result.chunks],
        # Retrieval never asserts Confirmed on its own: the Observation step tags the
        # evidence after reading it. Partially Confirmed is the strongest a raw
        # retrieval can be, by design.
        provisional_tag=ConfidenceTag.PARTIALLY_CONFIRMED,
        superseded_filtered=result.filtered_superseded,
    )


class DocumentRetrievalTool:
    """Semantic query over the deal's own unstructured documents."""

    name = "document_retrieval"
    description = (
        "Semantic search over this engagement's data-room documents (contracts, board "
        "decks, transcripts). Use several narrow, targeted queries rather than one "
        "broad one. Returns passages with citations, or No Data."
    )

    def __init__(self, engagement_id: str, index: Optional[DataRoomIndex] = None) -> None:
        self.engagement_id = engagement_id
        self.index = index or DataRoomIndex(engagement_id)

    def __call__(
        self,
        query: str,
        *,
        doc_tier: Optional[Tier] = None,
        doc_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> RetrievalObservation:
        where: dict[str, Any] = {}
        if doc_tier is not None:
            where["doc_tier"] = int(doc_tier)
        if doc_type:
            where["doc_type"] = doc_type
        # Chroma requires an explicit operator when filtering on more than one field.
        chroma_where: Optional[dict[str, Any]]
        if len(where) > 1:
            chroma_where = {"$and": [{k: v} for k, v in where.items()]}
        else:
            chroma_where = where or None
        result = self.index.query(query, k=k, where=chroma_where)
        return _to_observation(result, SourceKind.DATA_ROOM)


class MarketSearchTool:
    """Semantic query over the cross-engagement Knowledge-Base Index."""

    name = "market_search"
    description = (
        "Search the agent's persistent knowledge base: sub-sector diagnostic "
        "frameworks, the standing risk taxonomy, redacted prior findings, and external "
        "market and precedent-transaction data. Never contains client-confidential text."
    )

    def __init__(self, index: Optional[KnowledgeBaseIndex] = None) -> None:
        self.index = index or KnowledgeBaseIndex()

    def __call__(
        self,
        query: str,
        *,
        sub_sector: Optional[str] = None,
        topic: Optional[str] = None,
        k: Optional[int] = None,
    ) -> RetrievalObservation:
        where: dict[str, Any] = {}
        if sub_sector:
            where["sub_sector"] = sub_sector
        if topic:
            where["topic"] = topic
        chroma_where: Optional[dict[str, Any]]
        if len(where) > 1:
            chroma_where = {"$and": [{k: v} for k, v in where.items()]}
        else:
            chroma_where = where or None
        result = self.index.query(
            query, k=k, where=chroma_where, drop_superseded=False
        )
        return _to_observation(result, SourceKind.KNOWLEDGE_BASE)
