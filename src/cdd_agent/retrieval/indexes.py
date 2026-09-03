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
        # The index says where the chunk was stored; the document says what it is. A
        # 10-K filed in the data room is still a public filing, and citing it as
        # confidential management material would misstate both its weight and the
        # MNPI position.
        stored = str(self.metadata.get("source_kind") or "")
        try:
            resolved = SourceKind(stored) if stored else source_kind
        except ValueError:
            resolved = source_kind
        return Citation(
            source_kind=resolved,
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
        # Imported here rather than at module scope so the package stays importable
        # without a vector store installed - the computation and schema layers do not
        # need one, and neither do most of the tests.
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = get_settings()
        settings.ensure_dirs()
        _check_index_compatibility(settings.chroma_dir, chromadb.__version__)
        try:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        except BaseException as exc:
            # Chroma's Rust bindings raise a PanicException, which derives from
            # BaseException and would otherwise escape every normal handler as an
            # unreadable stack trace. The overwhelmingly common cause is an index
            # written by a different chromadb, so translate it - but only after
            # confirming it is not an ordinary error we should let through.
            if type(exc).__name__ != "PanicException":
                raise
            raise _mismatch_error(settings.chroma_dir, chromadb.__version__) from exc
        _record_index_version(settings.chroma_dir, chromadb.__version__)
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
        # strict: these four arrays are one query result split four ways. If the
        # backend ever returns them ragged, silently zipping to the shortest would
        # drop retrieved chunks without a trace - the failure would look like a
        # thin data room rather than a bug.
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
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
_VERSION_MARKER = ".chromadb-version"


class IndexVersionMismatch(RuntimeError):
    """The persisted index was written by a different chromadb than the one loaded."""


def _mismatch_error(chroma_dir: Any, running_version: str) -> IndexVersionMismatch:
    """Build the explanation shown when the persisted index cannot be opened."""
    from pathlib import Path

    marker = Path(chroma_dir) / _VERSION_MARKER
    written_by = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    # The marker records the last version that opened the index, which is not always
    # the one that wrote it - so never claim a version that contradicts the failure.
    provenance = (
        f"was written by chromadb {written_by}"
        if written_by and written_by != running_version
        else "was written by a different chromadb"
    )
    return IndexVersionMismatch(
        "\n".join(
            [
                f"The vector index at {chroma_dir} {provenance}, but chromadb "
                f"{running_version} is loaded. Chroma cannot read its own persisted "
                "format across versions.",
                "",
                "Delete the index directory and rebuild it:",
                f"    rm -r {chroma_dir}",
                "    cdd seed-kb",
                "    cdd ingest <engagement> <data-room>",
                "",
                "This usually means two environments share one data directory - "
                "CrewAI pins chromadb~=1.1.0, so a [critic] install resolves an older "
                "chromadb than a plain one. Give each interpreter its own "
                "CDD_CHROMA_DIR to keep both.",
            ]
        )
    )


def _check_index_compatibility(chroma_dir: Any, running_version: str) -> None:
    """Refuse an index whose recorded chromadb version differs from the running one.

    This catches the case cheaply and by name. It cannot catch everything: an index
    created before this check carries no marker, which is why opening the client is
    also wrapped - see `_BaseIndex.__init__`.
    """
    from pathlib import Path

    marker = Path(chroma_dir) / _VERSION_MARKER
    if not marker.exists():
        return
    if marker.read_text(encoding="utf-8").strip() != running_version:
        raise _mismatch_error(chroma_dir, running_version)


def _record_index_version(chroma_dir: Any, running_version: str) -> None:
    """Stamp the directory after a successful open, so the next mismatch is named."""
    from pathlib import Path

    try:
        (Path(chroma_dir) / _VERSION_MARKER).write_text(running_version, encoding="utf-8")
    except OSError:
        # A read-only or racing index directory is not a reason to fail the run.
        pass


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
