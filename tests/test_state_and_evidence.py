"""State store attribution, and the Evidence Matrix rating rules."""

from __future__ import annotations

import datetime as _dt

import pytest

from cdd_agent.schemas.common import Citation, ConfidenceTag, SourceKind
from cdd_agent.schemas.evidence import EvidenceItem, EvidenceMatrix
from cdd_agent.state.store import Collection


def _cite(kind: SourceKind = SourceKind.DATA_ROOM) -> Citation:
    return Citation(
        source_kind=kind,
        source_file="Board_Deck.txt",
        locator="slide 1",
        document_date=_dt.date(2026, 5, 10),
    )


def _item(hid: str, tag: ConfidenceTag, kind: SourceKind, n: int = 1) -> EvidenceItem:
    return EvidenceItem(
        id=f"EV-{hid}-{n}",
        engagement_id="e",
        created_by="Analyst",
        hypothesis_id=hid,
        claim="claim",
        tag=tag,
        citations=[] if tag is ConfidenceTag.NO_DATA else [_cite(kind)],
        source_kind=kind,
    )


# ------------------------------------------------------------------ the store
def test_writes_must_be_attributed(store):
    with pytest.raises(ValueError, match="attributed to an agent"):
        store.put("e", Collection.DEAL_PROFILE, "current", {"x": 1}, agent="")


def test_audit_log_records_every_write(store):
    store.put("e", Collection.DEAL_PROFILE, "current", {"v": 1}, agent="Intake Agent")
    store.put("e", Collection.DEAL_PROFILE, "current", {"v": 2}, agent="Analyst")
    trail = store.audit("e")
    assert [entry.action for entry in trail] == ["update", "create"]
    assert {entry.agent for entry in trail} == {"Intake Agent", "Analyst"}
    # An overwrite does not erase the prior payload from the trail.
    assert trail[0].payload_digest != trail[1].payload_digest


def test_purge_removes_documents_but_keeps_the_trail(store):
    store.put("e", Collection.DECK, "current", {"v": 1}, agent="Synthesizer")
    deleted = store.purge_engagement("e", agent="operator")
    assert deleted == 1
    assert store.get("e", Collection.DECK, "current") is None
    assert any(entry.action == "purge" for entry in store.audit("e"))


def test_full_teardown_clears_the_trail_too(store):
    store.put("e", Collection.DECK, "current", {"v": 1}, agent="Synthesizer")
    store.purge_engagement("e", agent="operator", keep_audit=False)
    assert [entry.action for entry in store.audit("e")] == ["purge"]


def test_corrections_are_recalled_across_engagements(store):
    """Checkpoint 2.1 recalibration, without carrying the deal it came from.

    A correction crosses engagements only when someone marks it shareable, and it
    crosses stripped of values, note and source. What survives is the signal that
    this field has been got wrong before in this sub-sector - which is what should
    change how the next deal is scoped. The note used to cross intact, and a note is
    free text a reviewer typed while looking at a client data room.
    """
    from cdd_agent.state.memory import Correction, LongTermMemory

    def record(shareable: bool) -> None:
        LongTermMemory(store, "deal-1").record_correction(
            Correction(
                engagement_id="deal-1",
                sub_sector="B2B cybersecurity SaaS",
                artifact="data_request",
                field_path="DR-012.tier",
                from_value="2",
                to_value="1",
                note="ARR waterfall is always Tier 1 in this sub-sector",
                at=_dt.datetime.now(_dt.timezone.utc),
                shareable=shareable,
            ),
            agent="operator",
        )

    record(shareable=False)
    assert LongTermMemory(store, "deal-2").corrections_for_sub_sector(
        "b2b cybersecurity saas") == [], "unmarked corrections stay in their engagement"

    record(shareable=True)
    recalled = LongTermMemory(store, "deal-2").corrections_for_sub_sector(
        "b2b cybersecurity saas"
    )
    assert len(recalled) == 1
    assert recalled[0].field_path == "DR-012.tier"
    assert recalled[0].note == "" and recalled[0].to_value == ""
    assert "Values withheld" in recalled[0].render_for_prompt()

    # The originating engagement keeps the whole thing.
    own = LongTermMemory(store, "deal-1").corrections_for_sub_sector(
        "b2b cybersecurity saas")
    assert "Tier 1" in own[0].note


# ---------------------------------------------------------- evidence matrix
def test_evidence_cannot_be_tagged_without_a_citation():
    with pytest.raises(ValueError, match="without a citation"):
        EvidenceItem(
            id="EV-1",
            engagement_id="e",
            created_by="Analyst",
            hypothesis_id="H1",
            claim="NRR is 118%",
            tag=ConfidenceTag.CONFIRMED,
            citations=[],
            source_kind=SourceKind.DATA_ROOM,
        )


def test_contradicted_dominates_supporting_evidence():
    """A contradicting finding must not be averaged away."""
    matrix = EvidenceMatrix(engagement_id="e", created_by="test")
    matrix.add(_item("H1", ConfidenceTag.CONFIRMED, SourceKind.PRIMARY_RESEARCH, 1))
    matrix.add(_item("H1", ConfidenceTag.CONTRADICTED, SourceKind.DATA_ROOM, 2))
    assert matrix.rating("H1") is ConfidenceTag.CONTRADICTED


def test_management_data_alone_cannot_reach_confirmed():
    matrix = EvidenceMatrix(engagement_id="e", created_by="test")
    matrix.add(_item("H1", ConfidenceTag.CONFIRMED, SourceKind.DATA_ROOM))
    assert matrix.rating("H1") is ConfidenceTag.PARTIALLY_CONFIRMED
    assert not matrix.triangulated("H1")


def test_independent_triangulation_permits_confirmed():
    matrix = EvidenceMatrix(engagement_id="e", created_by="test")
    matrix.add(_item("H1", ConfidenceTag.CONFIRMED, SourceKind.DATA_ROOM, 1))
    matrix.add(_item("H1", ConfidenceTag.CONFIRMED, SourceKind.PRIMARY_RESEARCH, 2))
    assert matrix.rating("H1") is ConfidenceTag.CONFIRMED
    assert matrix.triangulated("H1")


def test_no_evidence_is_no_data():
    assert EvidenceMatrix(engagement_id="e", created_by="t").rating("H9") is ConfidenceTag.NO_DATA
