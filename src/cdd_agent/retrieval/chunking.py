"""Boundary-aware chunking - Checkpoint 3.1 s 4.

Chunks are ~500-800 tokens with ~15% overlap, split on a *semantic* boundary rather
than a fixed length: a contract clause, a deck slide, or a transcript Q&A turn. The
overlap exists so a clause split across a boundary is still retrievable whole - the
specific failure this guards against is a renewal or step-down provision landing half
in one chunk and half in the next, which is exactly the passage the agent most needs
to find intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from cdd_agent.config import get_settings
from cdd_agent.schemas.common import SourceKind, Tier

# Boundary patterns, most specific first.
_CLAUSE = re.compile(
    r"^\s*(?:(?:Section|Article|Clause)\s+\d+[\.\d]*|\d+\.\d+(?:\.\d+)?|\(\w{1,3}\))[\.\)]?\s",
    re.MULTILINE,
)
_SLIDE = re.compile(r"^\s*(?:---+\s*)?(?:Slide|SLIDE)\s+\d+\s*[:\-]?", re.MULTILINE)
_QA_TURN = re.compile(r"^\s*(?:Q|A|Interviewer|Respondent|Expert|Analyst)\s*[:\-]", re.MULTILINE)
_PARAGRAPH = re.compile(r"\n\s*\n")


def estimate_tokens(text: str) -> int:
    """Whitespace-token approximation (~1.33 tokens/word).

    Deliberately dependency-free: the chunk band is a design parameter measured in
    tokens, and an approximation that is stable and inspectable is worth more here
    than an exact count that pulls in a tokenizer.
    """
    words = len(text.split())
    return int(words * 4 / 3)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    locator: str
    boundary_kind: str
    token_estimate: int
    metadata: dict[str, object] = field(default_factory=dict)


# Public-record documents can sit in the same folder as confidential material - a
# banker's data room routinely includes the last 10-K alongside the board pack. They
# are not the same kind of evidence: a filing is attested and creates no MNPI to
# read, while the board pack is neither. Classifying by filename keeps that
# distinction without asking the user to sort the folder.
_FILING_MARKERS = (
    "10-k", "10k", "10-q", "10q", "8-k", "8k", "20-f", "20f", "6-k",
    "def-14a", "def14a", "defa14a", "proxy", "annual-report", "annual_report",
    "annualreport", "interim-report", "earnings-call", "earnings_call",
    "earnings-transcript", "investor-day", "investor_day", "shareholder-letter",
    "prospectus", "s-1",
)
_RESEARCH_MARKERS = (
    "analyst", "broker", "research-note", "research_note", "sell-side",
    "sellside", "consensus", "equity-research", "initiation",
)


def classify_source_kind(source_file: str,
                         default: SourceKind = SourceKind.DATA_ROOM) -> SourceKind:
    """Read the document's provenance off its filename.

    Deliberately conservative: an unrecognised file stays whatever the index says it
    is. Mislabelling a confidential board pack as a public filing would understate
    both the MNPI exposure and the management-bias flag, so ambiguity resolves
    towards the more restrictive classification.
    """
    name = source_file.lower().replace(" ", "-").replace("_", "-")
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if any(m in stem for m in _RESEARCH_MARKERS):
        return SourceKind.SELL_SIDE_RESEARCH
    if any(m in stem for m in _FILING_MARKERS):
        return SourceKind.PUBLIC_FILING
    return default


@dataclass
class SourceDocument:
    """One unstructured data-room or knowledge-base file, already text-extracted."""

    source_file: str
    text: str
    doc_tier: Tier = Tier.DEPTH_BUILDING
    document_date: Optional[str] = None  # ISO date; drives supersession filtering
    doc_type: str = "document"           # contract | deck | transcript | document
    # Resolved from the filename in __post_init__ unless the caller states it. A
    # public filing sitting in the data room is public record wherever it is filed.
    source_kind: SourceKind = SourceKind.DATA_ROOM
    hypothesis_branch: Optional[str] = None
    version_group: Optional[str] = None  # derived in __post_init__ when not supplied

    def __post_init__(self) -> None:
        if self.version_group is None:
            self.version_group = _version_group(self.source_file)
        if self.source_kind is SourceKind.DATA_ROOM:
            self.source_kind = classify_source_kind(self.source_file)


def _file_slug(source_file: str) -> str:
    """A per-file identity, version markers and dates intact.

    Chunk ids must be unique per *document*, not per version group. Deriving them
    from the group made two versions of one document collide, so the upsert silently
    overwrote one with the other and the supersession filter had nothing left to
    filter - the exact grounded-but-wrong failure it exists to prevent, reintroduced
    at write time.
    """
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", source_file)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or stem.lower()


def _version_group(source_file: str) -> str:
    """Group name shared by drafts and finals of the same document.

    Supersession is decided on this group plus date, so "MSA_v2_DRAFT.pdf" and
    "MSA_FINAL.pdf" compete rather than both being retrievable as live sources.
    """
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", source_file)
    # A lookahead rather than \b: version markers are usually underscore-separated,
    # and \b never fires between "v2" and "_" because underscore is a word character.
    stem = re.sub(
        r"[ _-]*(v\d+(?:\.\d+)?|draft|final|clean|executed|rev\d*|copy|\(\d+\))"
        r"(?=[ _\-.]|$)",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"[ _-]*\d{4}[-_]?\d{2}[-_]?\d{2}\b", "", stem)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or stem.lower()


def detect_boundaries(text: str, doc_type: str) -> tuple[list[int], str]:
    """Return split offsets and the boundary kind that produced them."""
    for pattern, kind, applies in (
        (_QA_TURN, "qa_turn", doc_type in ("transcript", "interview")),
        (_SLIDE, "slide", doc_type in ("deck", "presentation")),
        (_CLAUSE, "clause", doc_type in ("contract", "agreement", "document")),
    ):
        if not applies:
            continue
        offsets = [m.start() for m in pattern.finditer(text)]
        if len(offsets) >= 2:
            return offsets, kind
    offsets = [0] + [m.end() for m in _PARAGRAPH.finditer(text)]
    return offsets, "paragraph"


def chunk_document(doc: SourceDocument) -> list[Chunk]:
    """Split one document into overlapping, boundary-aligned chunks."""
    settings = get_settings()
    text = doc.text.strip()
    if not text:
        return []

    offsets, kind = detect_boundaries(text, doc.doc_type)
    if not offsets or offsets[0] != 0:
        offsets = [0] + offsets
    units: list[str] = []
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else len(text)
        segment = text[start:end].strip()
        if segment:
            units.append(segment)
    if not units:
        units = [text]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    index = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens, index
        if not buffer:
            return
        body = "\n\n".join(buffer)
        index += 1
        chunks.append(
            Chunk(
                chunk_id=f"{_file_slug(doc.source_file)}::{index:04d}",
                text=body,
                source_file=doc.source_file,
                locator=_locator(kind, index, buffer[0]),
                boundary_kind=kind,
                token_estimate=estimate_tokens(body),
                metadata={
                    "source_file": doc.source_file,
                    "doc_tier": int(doc.doc_tier),
                    "doc_type": doc.doc_type,
                    "document_date": doc.document_date or "",
                    "version_group": doc.version_group or "",
                    "hypothesis_branch": doc.hypothesis_branch or "",
                    "boundary_kind": kind,
                    "source_kind": doc.source_kind.value,
                },
            )
        )
        # Carry ~15% of the flushed content forward so a boundary-split clause survives.
        overlap = _tail_for_overlap(buffer, settings.chunk_overlap_ratio)
        buffer = list(overlap)
        buffer_tokens = sum(estimate_tokens(u) for u in buffer)

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        # A single oversized unit (a very long clause) is emitted whole rather than
        # cut mid-sentence: an intact clause above the band beats a split one inside it.
        if unit_tokens >= settings.chunk_max_tokens:
            flush()
            index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{_file_slug(doc.source_file)}::{index:04d}",
                    text=unit,
                    source_file=doc.source_file,
                    locator=_locator(kind, index, unit),
                    boundary_kind=kind,
                    token_estimate=unit_tokens,
                    metadata={
                        "source_file": doc.source_file,
                        "doc_tier": int(doc.doc_tier),
                        "doc_type": doc.doc_type,
                        "document_date": doc.document_date or "",
                        "version_group": doc.version_group or "",
                        "hypothesis_branch": doc.hypothesis_branch or "",
                        "boundary_kind": kind,
                        "source_kind": doc.source_kind.value,
                        "oversized_unit": True,
                    },
                )
            )
            buffer, buffer_tokens = [], 0
            continue

        if buffer_tokens + unit_tokens > settings.chunk_max_tokens:
            flush()
        buffer.append(unit)
        buffer_tokens += unit_tokens
        if buffer_tokens >= settings.chunk_target_tokens:
            flush()

    if buffer and sum(estimate_tokens(u) for u in buffer) > 0:
        body = "\n\n".join(buffer)
        # Avoid emitting a trailing chunk that is pure overlap of the previous one.
        if not chunks or body not in chunks[-1].text:
            index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{_file_slug(doc.source_file)}::{index:04d}",
                    text=body,
                    source_file=doc.source_file,
                    locator=_locator(kind, index, buffer[0]),
                    boundary_kind=kind,
                    token_estimate=estimate_tokens(body),
                    metadata={
                        "source_file": doc.source_file,
                        "doc_tier": int(doc.doc_tier),
                        "doc_type": doc.doc_type,
                        "document_date": doc.document_date or "",
                        "version_group": doc.version_group or "",
                        "hypothesis_branch": doc.hypothesis_branch or "",
                        "boundary_kind": kind,
                        "source_kind": doc.source_kind.value,
                    },
                )
            )
    return chunks


def _tail_for_overlap(buffer: list[str], ratio: float) -> list[str]:
    total = sum(estimate_tokens(u) for u in buffer)
    budget = total * ratio
    tail: list[str] = []
    acc = 0
    for unit in reversed(buffer):
        t = estimate_tokens(unit)
        # Strictly under budget: carrying a unit larger than the overlap allowance
        # would push the next chunk past chunk_max_tokens, which is what the band is for.
        if acc + t > budget:
            break
        tail.insert(0, unit)
        acc += t
        if acc >= budget:
            break
    # Never carry the entire buffer forward - that would loop.
    return tail if len(tail) < len(buffer) else []


def _locator(kind: str, index: int, first_unit: str) -> str:
    head = " ".join(first_unit.split()[:8])
    label = {"clause": "clause", "slide": "slide", "qa_turn": "Q&A turn"}.get(kind, "para")
    return f"{label} {index}: {head}"


def chunk_all(docs: Iterable[SourceDocument]) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d)]
