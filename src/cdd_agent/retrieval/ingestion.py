"""Phase-3 ingestion: classify, index, extract, normalize.

Design specification s VI, steps 1-2, and Architecture v6.7 slide 1:

* Unstructured files (contracts, decks, transcripts) are chunked, embedded, and
  written to the Data-Room Index.
* Structured files (financials, CRM exports, cohort tables) are parsed to a common
  schema and queried directly - deliberately *not* embedded, because exact-match and
  aggregation beat semantic similarity for numeric lookups (Checkpoint 3.1 s 4).

The index it produces is auditable by the user: every file gets a classification, a
tier, and a date, and files that could not be dated are reported rather than assumed
current.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from cdd_agent.retrieval.chunking import SourceDocument, chunk_document
from cdd_agent.schemas.common import Tier

if TYPE_CHECKING:
    from cdd_agent.retrieval.indexes import DataRoomIndex

UNSTRUCTURED_SUFFIXES = {".txt", ".md", ".text"}
STRUCTURED_SUFFIXES = {".csv", ".tsv"}
JSON_SUFFIXES = {".json"}

_DATE_IN_NAME = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
_YEAR_MONTH = re.compile(r"(20\d{2})[-_]?(\d{2})(?!\d)")

_TYPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("contract", ("contract", "msa", "agreement", "sow", "order form", "terms")),
    ("transcript", ("transcript", "interview", "call notes", "expert")),
    ("deck", ("deck", "presentation", "board", "mgmt", "cim", "teaser")),
)

_TIER_MARKERS: tuple[tuple[Tier, tuple[str, ...]], ...] = (
    (Tier.DEAL_CRITICAL,
     ("audited", "financial", "contract", "msa", "customer", "arr", "revenue",
      "business plan", "model", "concentration", "cohort")),
    (Tier.DEPTH_BUILDING,
     ("crm", "pipeline", "org chart", "cost", "kpi", "nps", "churn", "roadmap",
      "compliance", "turnover")),
)


@dataclass
class StructuredTable:
    """A parsed tabular file, normalised to a common shape.

    Kept as rows of strings plus a typed numeric view so the computation tool can do
    real arithmetic while citations still point at a concrete sheet and column.
    """

    source_file: str
    name: str
    columns: list[str]
    rows: list[dict[str, str]]
    document_date: Optional[_dt.date] = None

    def to_records(self) -> list[dict[str, str]]:
        """Rows as read, without numeric coercion.

        Coercion is deliberately left to the computation tool, which knows which
        column is a measure. Coercing here turned label columns into numbers - a
        cohort of "2024" became 2024.0 and stopped matching its own label.
        """
        return [dict(row) for row in self.rows]


@dataclass
class IngestionReport:
    """The user-auditable index of what was ingested and how it was classified."""

    engagement_id: str
    # Where this engagement documents came from. Recorded so a later ingestion can
    # tell that it would be adding a second, different data room.
    data_room: str = ""
    unstructured: list[dict[str, Any]] = field(default_factory=list)
    structured: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    undated: list[str] = field(default_factory=list)
    chunks_indexed: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.unstructured)} unstructured file(s) -> {self.chunks_indexed} chunks; "
            f"{len(self.structured)} structured table(s); "
            f"{len(self.skipped)} skipped; {len(self.undated)} undated"
        )


def classify_document(path: Path, text: str = "") -> tuple[str, Tier, Optional[_dt.date]]:
    """Assign doc_type, tier, and date from filename and leading content."""
    name = path.name.lower()
    head = text[:400].lower()
    doc_type = "document"
    for kind, markers in _TYPE_MARKERS:
        if any(m in name or m in head for m in markers):
            doc_type = kind
            break
    tier = Tier.ENRICHMENT
    for candidate, markers in _TIER_MARKERS:
        if any(m in name or m in head for m in markers):
            tier = candidate
            break
    return doc_type, tier, extract_date(path, text)


def extract_date(path: Path, text: str = "") -> Optional[_dt.date]:
    """Prefer a date in the filename, then one in the first lines of the document.

    An undated document is left undated on purpose: guessing a date would defeat the
    supersession filter, which is what stops a stale draft outranking its final version.
    """
    for source in (path.name, text[:400]):
        m = _DATE_IN_NAME.search(source)
        if m:
            try:
                return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        m = _YEAR_MONTH.search(source)
        if m:
            try:
                return _dt.date(int(m.group(1)), int(m.group(2)), 1)
            except ValueError:
                pass
    return None


def parse_structured(path: Path) -> Optional[StructuredTable]:
    suffix = path.suffix.lower()
    if suffix in STRUCTURED_SUFFIXES:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            rows = [{(k or ""): (v or "") for k, v in r.items()} for r in reader]
            cols = list(reader.fieldnames or [])
        return StructuredTable(
            source_file=path.name, name=path.stem, columns=cols, rows=rows,
            document_date=extract_date(path),
        )
    if suffix in JSON_SUFFIXES:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data and isinstance(data[0], dict):
            cols = list(data[0].keys())
            rows = [{k: str(r.get(k, "")) for k in cols} for r in data]
            return StructuredTable(
                source_file=path.name, name=path.stem, columns=cols, rows=rows,
                document_date=extract_date(path),
            )
    return None


class DataRoomConflict(RuntimeError):
    """Raised when a second, different data room would be added to an engagement."""


class DataRoomSharedAcrossTargets(DataRoomConflict):
    """Raised when one folder is already bound to a different company engagement."""


def default_data_room(engagement_id: str, *, create: bool = True) -> Path:
    """The folder this engagement owns, unless it is deliberately pointed elsewhere.

    Isolation should be what happens when nobody thinks about it. Leaving the path
    blank made the demo data room the nearest thing to hand, which is how another
    company documents got one click away from every engagement.
    """
    from cdd_agent.config import get_settings

    room = get_settings().engagements_dir / _safe_id(engagement_id) / "data_room"
    if create:
        room.mkdir(parents=True, exist_ok=True)
    return room


def _safe_id(engagement_id: str) -> str:
    """One path component, whatever the engagement id happens to be.

    Ids come from the user. Runs of dots are collapsed so the result can never read as
    a parent reference, and the caller asserts the folder lands under the engagements
    directory regardless.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", engagement_id)
    slug = re.sub(r"\.{2,}", ".", slug).strip(".-")
    return slug or "engagement"


def _target_of(engagement_id: str, store: Any) -> str:
    from cdd_agent.state.store import Collection

    profile = store.get(engagement_id, Collection.DEAL_PROFILE, "current") or {}
    return str((profile.get("target") or {}).get("legal_name") or "").strip()


def _bound_elsewhere(engagement_id: str, resolved: str, store: Any) -> list[tuple[str, str]]:
    """Engagements already using this folder, with the target each one is about."""
    from cdd_agent.state.store import Collection

    out: list[tuple[str, str]] = []
    for other in store.engagements():
        if other == engagement_id:
            continue
        record = store.get(other, Collection.METRICS, "ingestion") or {}
        if str(record.get("data_room") or "") == resolved:
            out.append((other, _target_of(other, store)))
    return out


def _recorded_data_room(engagement_id: str, store: Optional[Any] = None) -> str:
    from cdd_agent.state.store import Collection, StateStore

    store = store or StateStore()
    record = store.get(engagement_id, Collection.METRICS, "ingestion") or {}
    return str(record.get("data_room") or "")


def check_data_room(engagement_id: str, directory: Path | str, *,
                    store: Optional[Any] = None, force: bool = False) -> str:
    """Refuse a data room that would blend two companies.

    Two rules, and they are different failures.

    A second, *different* folder for one engagement is nearly always a misclick, and
    it is silent: the documents join the engagement index, the parsed tables
    overwrite, and the computed exhibits are rebuilt from another company numbers
    under this company name. It happened - a customer schedule from the demo data
    room was rendered as a real target concentration risk.

    One folder bound to two engagements is *not* automatically wrong. The same target
    is routinely diligenced under more than one structure, and those engagements
    should read one set of documents rather than drifting copies of it. What is wrong
    is sharing a folder across engagements about different companies, which is the
    same contamination arriving from the other direction.

    The check lives here rather than in the web layer because it was in the web layer
    first, and the command line walked straight past it.
    """
    from cdd_agent.state.store import StateStore

    store = store or StateStore()
    resolved = str(Path(directory).resolve())

    prior = _recorded_data_room(engagement_id, store)
    if prior and prior != resolved and not force:
        raise DataRoomConflict(
            f"{engagement_id} was already ingested from {prior}. Ingesting "
            f"{resolved} would add another company documents to the same index and "
            f"overwrite the parsed tables the computed exhibits are built from. Use a "
            f"new engagement, or pass force to replace the data room deliberately."
        )

    mine = _target_of(engagement_id, store)
    if mine and not force:
        for other, theirs in _bound_elsewhere(engagement_id, resolved, store):
            if theirs and theirs.casefold() != mine.casefold():
                raise DataRoomSharedAcrossTargets(
                    f"{resolved} is the data room for {other}, which is about "
                    f"{theirs}. This engagement is about {mine}. Sharing a folder "
                    f"between engagements on the same target is fine; sharing it "
                    f"across two companies puts one client documents into the other "
                    f"work product. Give this engagement its own data room."
                )
    return resolved


def ingest_directory(
    engagement_id: str,
    directory: Path | str,
    *,
    index: Optional["DataRoomIndex"] = None,
    force: bool = False,
    store: Optional[Any] = None,
) -> tuple[IngestionReport, list[StructuredTable]]:
    """Ingest a data-room folder into the engagement-scoped index."""
    from cdd_agent.retrieval.indexes import DataRoomIndex

    directory = Path(directory)
    # Every caller passes through here, so the boundary holds for the CLI, the API and
    # the Controller alike.
    resolved = check_data_room(engagement_id, directory, store=store, force=force)
    index = index or DataRoomIndex(engagement_id)
    report = IngestionReport(engagement_id=engagement_id)
    report.data_room = resolved
    tables: list[StructuredTable] = []

    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix in STRUCTURED_SUFFIXES or suffix in JSON_SUFFIXES:
            table = parse_structured(path)
            if table is None:
                report.skipped.append({"file": path.name, "reason": "unparsed structured file"})
                continue
            tables.append(table)
            report.structured.append(
                {"file": path.name, "rows": len(table.rows), "columns": table.columns,
                 "date": table.document_date.isoformat() if table.document_date else None}
            )
            if table.document_date is None:
                report.undated.append(path.name)
            continue

        if suffix not in UNSTRUCTURED_SUFFIXES:
            report.skipped.append(
                {"file": path.name,
                 "reason": f"unsupported extension {suffix or '(none)'}; convert to .txt/.md "
                           "or extend the extractor"}
            )
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_type, tier, date = classify_document(path, text)
        doc = SourceDocument(
            source_file=path.name,
            text=text,
            doc_tier=tier,
            document_date=date.isoformat() if date else None,
            doc_type=doc_type,
        )
        chunks = chunk_document(doc)
        added = index.add(chunks)
        report.chunks_indexed += added
        report.unstructured.append(
            {"file": path.name, "doc_type": doc_type, "tier": int(tier),
             "date": date.isoformat() if date else None, "chunks": added,
             "version_group": doc.version_group}
        )
        if date is None:
            report.undated.append(path.name)

    return report, tables


def ingest_knowledge_base(docs: Iterable[SourceDocument], *, topic: str = "reference") -> int:
    """Seed or extend the cross-engagement Knowledge-Base Index."""
    from cdd_agent.retrieval.indexes import KnowledgeBaseIndex

    kb = KnowledgeBaseIndex()
    total = 0
    for doc in docs:
        total += kb.add_reference(chunk_document(doc), topic=topic)
    return total


def parse_number(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    text = text.lstrip("$")
    try:
        number = float(text)
    except ValueError:
        return value
    if negative:
        number = -number
    if percent:
        number = number / 100.0
    return number
