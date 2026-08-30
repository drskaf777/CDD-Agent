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
from typing import Any, Iterable, Optional

from cdd_agent.retrieval.chunking import SourceDocument, chunk_document
from cdd_agent.retrieval.indexes import DataRoomIndex
from cdd_agent.schemas.common import Tier

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

    def to_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self.rows:
            rec: dict[str, Any] = {}
            for k, v in row.items():
                rec[k] = _maybe_number(v)
            out.append(rec)
        return out


@dataclass
class IngestionReport:
    """The user-auditable index of what was ingested and how it was classified."""

    engagement_id: str
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


def ingest_directory(
    engagement_id: str,
    directory: Path | str,
    *,
    index: Optional[DataRoomIndex] = None,
) -> tuple[IngestionReport, list[StructuredTable]]:
    """Ingest a data-room folder into the engagement-scoped index."""
    directory = Path(directory)
    index = index or DataRoomIndex(engagement_id)
    report = IngestionReport(engagement_id=engagement_id)
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


def _maybe_number(value: str) -> Any:
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
