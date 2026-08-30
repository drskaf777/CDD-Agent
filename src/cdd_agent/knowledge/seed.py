"""Seed the cross-engagement Knowledge-Base Index.

Checkpoint 3.1 s 2 lists what belongs in this index: sub-sector diagnostic frameworks,
the standing risk taxonomy, prior engagements' *redacted* findings, and externally
sourced market data. The first two are already structured data in this package, so they
are rendered to text and indexed here rather than maintained twice.

Nothing client-confidential is written to this index. That is not a convention - the
Data-Room Index is a separate Chroma collection per deal, and this module only reads
from `cdd_agent.knowledge`, which contains no engagement data.
"""

from __future__ import annotations

import datetime as _dt

from cdd_agent.knowledge.data_request_catalog import (
    ADDONS_BY_MODULE,
    UNIVERSAL_CATALOG,
)
from cdd_agent.knowledge.four_question_test import FOUR_QUESTIONS
from cdd_agent.knowledge.outline import PREBUILT_MODULES, UNIVERSAL_OUTLINE
from cdd_agent.knowledge.risk_taxonomy import TAXONOMY
from cdd_agent.retrieval.chunking import SourceDocument
from cdd_agent.retrieval.indexes import KnowledgeBaseIndex
from cdd_agent.schemas.common import Tier

_TODAY = _dt.date.today().isoformat()


def _outline_doc() -> SourceDocument:
    lines = ["Enhanced master outline for commercial due diligence.", ""]
    for section in UNIVERSAL_OUTLINE:
        tag = " (new)" if section.is_new else (" (enhanced)" if section.is_enhanced else "")
        lines.append(f"Section {section.number}: {section.title}{tag}")
        lines += [f"  - {e}" for e in section.key_elements]
        lines.append("")
    return SourceDocument(
        source_file="kb_enhanced_master_outline.txt",
        text="\n".join(lines),
        doc_tier=Tier.DEAL_CRITICAL,
        document_date=_TODAY,
        doc_type="document",
    )


def _taxonomy_doc() -> SourceDocument:
    lines = ["Standing risk taxonomy for commercial due diligence.", ""]
    for category, screens in TAXONOMY.items():
        lines.append(f"Risk category: {category.value}")
        for screen in screens:
            scope = " (strategic buyers only)" if screen.strategic_only else ""
            lines.append(f"  - Screen for: {screen.description}{scope}")
        lines.append("")
    return SourceDocument(
        source_file="kb_standing_risk_taxonomy.txt",
        text="\n".join(lines),
        doc_tier=Tier.DEAL_CRITICAL,
        document_date=_TODAY,
        doc_type="document",
    )


def _four_question_doc() -> SourceDocument:
    lines = [
        "The four-question test: the screen that separates plan-validation from "
        "company-judging.",
        "",
    ]
    for q in FOUR_QUESTIONS:
        lines.append(f"- {q.text}")
    lines += [
        "",
        "A diligence plan that cannot test one of these four is not a diligence plan; "
        "it is a company review. Every candidate hypothesis tree is screened against "
        "all four before it is scored on anything else.",
    ]
    return SourceDocument(
        source_file="kb_four_question_test.txt",
        text="\n".join(lines),
        doc_tier=Tier.DEAL_CRITICAL,
        document_date=_TODAY,
        doc_type="document",
    )


def _module_docs() -> list[tuple[str, SourceDocument]]:
    docs: list[tuple[str, SourceDocument]] = []
    for name, module in PREBUILT_MODULES.items():
        lines = [f"Sub-sector diagnostic framework: {name}.", ""]
        for section_no, elements in sorted(module.items()):
            lines.append(f"Applies at outline section {section_no}:")
            lines += [f"  - {e}" for e in elements]
            lines.append("")
        lines.append("Diagnostic data requests for this sub-sector:")
        for item in ADDONS_BY_MODULE.get(name, ()):
            reason = f" Rationale: {item.rationale}" if item.rationale else ""
            lines.append(f"  - [Tier {int(item.tier)}] {item.category}: {item.item}.{reason}")
        docs.append(
            (
                name,
                SourceDocument(
                    source_file=f"kb_subsector_{name}.txt",
                    text="\n".join(lines),
                    doc_tier=Tier.DEAL_CRITICAL,
                    document_date=_TODAY,
                    doc_type="document",
                ),
            )
        )
    return docs


def _universal_catalog_doc() -> SourceDocument:
    lines = ["Universal data-request catalogue, tiered by how load-bearing each item is.", ""]
    for item in UNIVERSAL_CATALOG:
        lines.append(f"[Tier {int(item.tier)}] {item.category}: {item.item}")
    lines += [
        "",
        "Tier 1 must be received before any hypothesis can be rated Confirmed or "
        "Contradicted. Tier 2 builds presentation-ready depth. Tier 3 enriches an "
        "exhibit and is never blocking.",
    ]
    return SourceDocument(
        source_file="kb_universal_data_request_catalog.txt",
        text="\n".join(lines),
        doc_tier=Tier.DEPTH_BUILDING,
        document_date=_TODAY,
        doc_type="document",
    )


def seed_knowledge_base(index: KnowledgeBaseIndex | None = None) -> dict[str, int]:
    """Index the built-in reference corpus. Idempotent - chunk ids are stable."""
    from cdd_agent.retrieval.chunking import chunk_document

    kb = index or KnowledgeBaseIndex()
    counts: dict[str, int] = {}

    for topic, doc in (
        ("outline", _outline_doc()),
        ("risk_taxonomy", _taxonomy_doc()),
        ("four_question_test", _four_question_doc()),
        ("data_request_catalog", _universal_catalog_doc()),
    ):
        counts[topic] = kb.add_reference(chunk_document(doc), topic=topic)

    for name, doc in _module_docs():
        counts[f"sub_sector::{name}"] = kb.add_reference(
            chunk_document(doc), topic="sub_sector_framework", sub_sector=name
        )
    return counts
