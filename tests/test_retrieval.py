"""Chunking and the two indexes.

The retrieval tests target the three mitigations for the "confident near-miss" from
Checkpoint 3.1 s 5 - supersession filtering, the similarity floor mapping to No Data,
and de-duplication - because those are what make a citation mean something.
"""

from __future__ import annotations

from cdd_agent.config import get_settings
from cdd_agent.retrieval.chunking import (
    SourceDocument,
    chunk_document,
    detect_boundaries,
    estimate_tokens,
)
from cdd_agent.retrieval.indexes import DataRoomIndex, KnowledgeBaseIndex
from cdd_agent.schemas.common import Tier

_CONTRACT = """Section 1. Scope
This agreement covers the twenty largest subscription contracts.

Section 2. Term and Renewal
2.1 Each agreement renews automatically for successive twelve month periods.
2.2 Eleven agreements carry a contractual step-down reducing the per-seat rate by
between eight and fifteen percent at first renewal.

Section 3. Cross-Sell
3.1 Additional modules are licensed under a separate order form.
"""


def test_contract_chunking_splits_on_clause_boundaries():
    offsets, kind = detect_boundaries(_CONTRACT, "contract")
    assert kind == "clause"
    assert len(offsets) >= 3


def test_transcript_chunking_splits_on_qa_turns():
    text = "Q: First question\nA: First answer\nQ: Second question\nA: Second answer\n"
    _, kind = detect_boundaries(text, "transcript")
    assert kind == "qa_turn"


def test_deck_chunking_splits_on_slides():
    text = "Slide 1: Overview\nbody\n\nSlide 2: Metrics\nbody\n"
    _, kind = detect_boundaries(text, "deck")
    assert kind == "slide"


def test_chunks_stay_within_the_configured_band():
    settings = get_settings()
    body = "\n\n".join(f"Section {i}. " + ("word " * 300) for i in range(1, 8))
    doc = SourceDocument(source_file="long.txt", text=body, doc_type="contract")
    chunks = chunk_document(doc)
    assert chunks
    for chunk in chunks:
        # An oversized single clause is emitted whole rather than cut mid-sentence.
        if chunk.metadata.get("oversized_unit"):
            continue
        assert chunk.token_estimate <= settings.chunk_max_tokens


def test_chunk_ids_are_unique_per_document_not_per_version_group():
    """Two versions of one document must not share chunk ids.

    They share a version group by design - that is what makes them compete for
    supersession. But if the chunk id also comes from the group, the second write
    upserts over the first and one version is destroyed at ingestion, leaving the
    supersession filter nothing to filter. The dates in these names are the realistic
    case: `_version_group` strips them, so both files reduce to "board-deck".
    """
    old = SourceDocument(source_file="Board_Deck_2025-11-02.txt",
                         text="Slide 1: retention was 124 percent", doc_type="deck",
                         document_date="2025-11-02")
    new = SourceDocument(source_file="Board_Deck_2026-05-10.txt",
                         text="Slide 1: retention was 118 percent", doc_type="deck",
                         document_date="2026-05-10")
    assert old.version_group == new.version_group == "board-deck"
    old_ids = {c.chunk_id for c in chunk_document(old)}
    new_ids = {c.chunk_id for c in chunk_document(new)}
    assert old_ids and new_ids
    assert not (old_ids & new_ids), "chunk ids collided; one version would overwrite the other"


def test_both_versions_survive_ingestion_so_the_filter_can_act():
    index = DataRoomIndex("collision-test")
    for name, date, body in (
        ("Board_Deck_2025-11-02.txt", "2025-11-02", "Slide 1: net revenue retention was 124 percent"),
        ("Board_Deck_2026-05-10.txt", "2026-05-10", "Slide 1: net revenue retention was 118 percent"),
    ):
        index.add(chunk_document(SourceDocument(
            source_file=name, text=body, doc_type="deck", document_date=date,
            doc_tier=Tier.DEAL_CRITICAL)))
    assert index.count() == 2, "both versions must be indexed for supersession to mean anything"
    result = index.query("net revenue retention", similarity_floor=0.0)
    assert {c.source_file for c in result.chunks} == {"Board_Deck_2026-05-10.txt"}
    assert result.filtered_superseded, "the stale version should be reported as filtered"


def test_version_group_collapses_drafts_and_finals():
    a = SourceDocument(source_file="MSA_v2_DRAFT.txt", text="x", doc_type="contract")
    b = SourceDocument(source_file="MSA_FINAL.txt", text="x", doc_type="contract")
    assert a.version_group == b.version_group


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five")


# ------------------------------------------------------------------- the index
def _doc(name: str, date: str, body: str, group: str) -> SourceDocument:
    return SourceDocument(
        source_file=name,
        text=body,
        doc_tier=Tier.DEAL_CRITICAL,
        document_date=date,
        doc_type="deck",
        version_group=group,
    )


def test_superseded_version_is_filtered_before_ranking():
    """A stale-but-cited figure is the grounded-but-wrong failure mode."""
    index = DataRoomIndex("supersession-test")
    old = _doc(
        "Board_Deck_2025.txt", "2025-11-02",
        "Slide 1: Net revenue retention was 124 percent for the trailing twelve months.",
        "board-deck",
    )
    new = _doc(
        "Board_Deck_2026.txt", "2026-05-10",
        "Slide 1: Net revenue retention was 118 percent for the trailing twelve months.",
        "board-deck",
    )
    index.add(chunk_document(old))
    index.add(chunk_document(new))

    result = index.query("net revenue retention trailing twelve months", similarity_floor=0.0)
    sources = {c.source_file for c in result.chunks}
    assert "Board_Deck_2025.txt" not in sources
    assert "Board_Deck_2026.txt" in sources
    assert any("superseded" in s for s in result.filtered_superseded)


def test_undated_documents_never_supersede_a_dated_one():
    index = DataRoomIndex("undated-test")
    index.add(chunk_document(_doc("A.txt", "", "Slide 1: retention commentary", "g")))
    index.add(chunk_document(_doc("B.txt", "2026-01-01", "Slide 1: retention commentary", "g")))
    result = index.query("retention commentary", similarity_floor=0.0)
    assert {c.source_file for c in result.chunks} == {"A.txt", "B.txt"}


def test_below_floor_match_is_reported_as_no_data_not_returned():
    index = DataRoomIndex("floor-test")
    index.add(
        chunk_document(_doc("A.txt", "2026-01-01", "Slide 1: catering invoices", "a"))
    )
    result = index.query("payer contract renegotiation calendar", similarity_floor=0.99)
    assert result.is_empty
    assert result.below_floor


def test_empty_index_returns_no_data():
    result = DataRoomIndex("empty-test").query("anything")
    assert result.is_empty and result.below_floor


def test_results_are_deduplicated_by_source_file():
    index = DataRoomIndex("dedupe-test")
    body = "\n\n".join(f"Slide {i}: retention and churn commentary" for i in range(1, 9))
    index.add(chunk_document(_doc("A.txt", "2026-01-01", body, "a")))
    result = index.query("retention and churn", similarity_floor=0.0)
    assert len({c.source_file for c in result.chunks}) == len(result.chunks)


def test_engagement_indexes_are_separate_collections():
    """A namespace boundary, not a metadata filter - a bad `where` cannot leak a deal."""
    a = DataRoomIndex("deal-a")
    b = DataRoomIndex("deal-b")
    a.add(chunk_document(_doc("A.txt", "2026-01-01", "Slide 1: confidential to deal A", "a")))
    assert a.collection_name != b.collection_name
    assert b.query("confidential to deal A").is_empty


def test_knowledge_base_seed_is_queryable():
    from cdd_agent.knowledge.seed import seed_knowledge_base

    counts = seed_knowledge_base()
    assert sum(counts.values()) > 0
    result = KnowledgeBaseIndex().query("standing risk taxonomy", similarity_floor=0.0)
    assert not result.is_empty


def test_periodic_filings_do_not_supersede_each_other():
    """Three consecutive annual reports are three periods, not three drafts.

    Under the ordinary rule they shared one version group and the two older ones were
    filtered as stale - removing exactly the history a growth trend or a
    guidance-against-delivery test is built from.
    """
    from cdd_agent.retrieval.chunking import _version_group

    groups = {_version_group(f) for f in (
        "FRSH_10-K_2026-02-26.txt", "FRSH_10-K_2025-02-20.txt",
        "FRSH_10-K_2024-02-16.txt")}
    assert len(groups) == 3, "each fiscal year must stand on its own"

    # An amendment is a real revision and still competes with its original.
    assert _version_group("MSA_TopAccounts_v2_DRAFT_2026-03-15.txt") == \
           _version_group("MSA_TopAccounts_FINAL_2026-04-01.txt")
    assert _version_group("Board_Deck_2025-11-02.txt") == \
           _version_group("Board_Deck_2026-05-10.txt")
