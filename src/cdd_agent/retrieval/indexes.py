"""The two vector indexes - Checkpoint 3.1 s 2.

They are separate, not one index with a filter column, because the two bodies of
evidence have different confidentiality and lifecycle requirements:

* ``DataRoomIndex`` is engagement-scoped, access-controlled, and purged or archived at
  close per the NDA constraints captured at intake (Category F).
* ``KnowledgeBaseIndex`` is cross-engagement and persistent - sub-sector frameworks,
  the standing risk taxonomy, redacted prior findings, and external market data. This
  is where judgment compounds across deals.

Retrieval is hybrid: metadata filtering happens *before* ranking, so a query can never
return a chunk from the wrong deal or a superseded draft even when it is semantically
close. A result below the similarity floor is reported as No Data rather than returned
as a low-confidence match - the "confident near-miss" mitigation from Checkpoint 3.1 s 5.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from cdd_agent.config import get_settings
from cdd_agent.retrieval.chunking import Chunk
from cdd_agent.retrieval.embeddings import get_embedding_function
from cdd_agent.schemas.common import Citation, SourceKind, Tier

KNOWLEDGE_BASE_COLLECTION = "knowledge_base"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    locator: str
    similarity: float
    metadata: dict[str, Any]

    @property
    def document_date(self) -> Optional[_dt.date]:
        raw = str(self.metadata.get("document_date") or "")
        try:
            return _dt.date.fromisoformat(raw) if raw else None
        except ValueError:
            return None

    def to_citation(self, source_kind: SourceKind) -> Citation:
        tier_raw = self.metadata.get("doc_tier")
        return Citation(
            source_kind=source_kind,
            source_file=self.source_file,
            locator=self.locator,
            chunk_id=self.chunk_id,
            document_date=self.document_date,
            document_tier=Tier(int(tier_raw)) if tier_raw else None,
            quoted_text=self.text[:2000],
            similarity=round(self.similarity, 4),
        )


@dataclass
class RetrievalResult:
    """What a query returned, plus why it returned that.

    ``below_floor`` is carried explicitly so the caller can tag a hypothesis No Data
    with a reason, instead of silently receiving an empty list.
    """

    query: str
    chunks: list[RetrievedChunk]
    below_floor: bool = False
    filtered_superseded: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.filtered_superseded is None:
            self.filtered_superseded = []

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class _BaseIndex:
    source_kind: SourceKind = SourceKind.DATA_ROOM

    def __init__(self, collection_name: str) -> None:
        settings = get_settings()
        settings.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ writes
    def add(self, chunks: Sequence[Chunk], extra_metadata: dict[str, Any] | None = None) -> int:
        if not chunks:
            return 0
        metadatas = []
        for c in chunks:
            md = {k: v for k, v in c.metadata.items()}
            md["locator"] = c.locator
            if extra_metadata:
                md.update(extra_metadata)
            metadatas.append({k: _scalar(v) for k, v in md.items()})
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def drop(self) -> None:
        self._client.delete_collection(self.collection_name)

    # ------------------------------------------------------------------- reads
    def query(
        self,
        text: str,
        *,
        k: Optional[int] = None,
        where: Optional[dict[str, Any]] = None,
        similarity_floor: Optional[float] = None,
        dedupe_by_source: bool = True,
        drop_superseded: bool = True,
    ) -> RetrievalResult:
        settings = get_settings()
        k = k or settings.top_k
        floor = settings.similarity_floor if similarity_floor is None else similarity_floor

        if self._collection.count() == 0:
            return RetrievalResult(query=text, chunks=[], below_floor=True)

        # Over-fetch so de-duplication and supersession filtering still leave k results.
        raw = self._collection.query(
            query_texts=[text],
            n_results=min(max(k * 4, k), max(self._collection.count(), 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        candidates: list[RetrievedChunk] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            candidates.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc or "",
                    source_file=str(meta.get("source_file", cid.split("::")[0])),
                    locator=str(meta.get("locator", "")),
                    similarity=_cosine_similarity(dist),
                    metadata=dict(meta),
                )
            )

        superseded: list[str] = []
        if drop_superseded:
            candidates, superseded = _drop_superseded(candidates)

        kept = [c for c in candidates if c.similarity >= floor]
        below_floor = bool(candidates) and not kept

        if dedupe_by_source:
            seen: set[str] = set()
            deduped: list[RetrievedChunk] = []
            for c in kept:
                if c.source_file in seen:
                    continue
                seen.add(c.source_file)
                deduped.append(c)
            kept = deduped

        return RetrievalResult(
            query=text,
            chunks=kept[:k],
            below_floor=below_floor,
            filtered_superseded=superseded,
        )


class DataRoomIndex(_BaseIndex):
    """Engagement-scoped. One Chroma collection per deal - a hard namespace boundary.

    Using a separate collection rather than a metadata filter means a coding error in
    a `where` clause cannot leak one engagement's documents into another's retrieval.
    """

    source_kind = SourceKind.DATA_ROOM

    def __init__(self, engagement_id: str) -> None:
        self.engagement_id = engagement_id
        super().__init__(f"dataroom__{_safe(engagement_id)}")

    def add(self, chunks: Sequence[Chunk], extra_metadata: dict[str, Any] | None = None) -> int:
        md = {"engagement_id": self.engagement_id}
        md.update(extra_metadata or {})
        return super().add(chunks, md)

    def purge(self) -> None:
        """Teardown at engagement close, per intake Category F retention policy."""
        self.drop()


class KnowledgeBaseIndex(_BaseIndex):
    """Cross-engagement and persistent. Never holds client-confidential text."""

    source_kind = SourceKind.KNOWLEDGE_BASE

    def __init__(self) -> None:
        super().__init__(KNOWLEDGE_BASE_COLLECTION)

    def add_reference(
        self, chunks: Sequence[Chunk], *, topic: str, sub_sector: str = ""
    ) -> int:
        return super().add(chunks, {"topic": topic, "sub_sector": sub_sector})


# --------------------------------------------------------------------- helpers
def _cosine_similarity(distance: float | None) -> float:
    """Chroma returns cosine *distance*; the design speaks in similarity."""
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _drop_superseded(
    candidates: Iterable[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[str]]:
    """Keep only the most-recent-dated version within each version group.

    This is the first of the three "confident near-miss" mitigations: supersession is
    part of the filter rather than something similarity ranking is trusted to get
    right. Undated documents are never used to supersede a dated one.
    """
    items = list(candidates)
    latest: dict[str, _dt.date] = {}
    for c in items:
        group = str(c.metadata.get("version_group") or c.source_file)
        d = c.document_date
        if d and (group not in latest or d > latest[group]):
            latest[group] = d

    kept: list[RetrievedChunk] = []
    dropped: list[str] = []
    for c in items:
        group = str(c.metadata.get("version_group") or c.source_file)
        d = c.document_date
        if group in latest and d is not None and d < latest[group]:
            dropped.append(f"{c.source_file} ({d.isoformat()}, superseded)")
            continue
        kept.append(c)
    return kept, dropped


def _safe(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-")
    return (out or "engagement")[:50]


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
