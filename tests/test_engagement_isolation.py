"""No silent blend: one client data must never reach another client work product.

Mixing two engagements is not merely a correctness bug. The data room is supplied
under an NDA, and on a listed target it is material non-public information, so a
figure from one deal appearing in another deal prompt or deck is a breach whether or
not anyone notices it. These tests hold the boundary at every channel that crosses
it, including the one channel that is meant to.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from cdd_agent.state.memory import Correction, LongTermMemory
from cdd_agent.state.store import Collection, StateStore


def correction(engagement: str, *, shareable: bool = False,
               to_value: str = "118% NRR") -> Correction:
    return Correction(
        engagement_id=engagement, sub_sector="enterprise SaaS",
        artifact="risk_register", field_path="RISK-001.severity",
        from_value="124% NRR", to_value=to_value,
        note=f"Top-5 accounts at {engagement} step down at renewal.",
        at=_dt.datetime(2026, 9, 1, tzinfo=_dt.timezone.utc), shareable=shareable)


def _store_correction(store: StateStore, c: Correction) -> None:
    store.put(c.engagement_id, Collection.CORRECTION, f"c-{c.engagement_id}",
              c, agent="test")


def test_another_engagement_correction_does_not_cross_by_default(isolated_settings):
    """Off by default. Only the person who typed it can say it carries no client
    detail, so silence must mean no."""
    store = StateStore()
    _store_correction(store, correction("project-sentinel"))
    recalled = LongTermMemory(store, "project-kanpur").corrections_for_sub_sector(
        "enterprise SaaS")
    assert recalled == []


def test_a_shared_correction_crosses_without_its_values(isolated_settings):
    store = StateStore()
    _store_correction(store, correction("project-sentinel", shareable=True))
    recalled = LongTermMemory(store, "project-kanpur").corrections_for_sub_sector(
        "enterprise SaaS")
    assert len(recalled) == 1
    carried = recalled[0]
    assert carried.from_value == "" and carried.to_value == ""
    assert carried.note == "", "a free-text note can name a customer or a figure"
    assert carried.engagement_id == "", "the source deal is not disclosed either"
    # The structural signal survives, which is the whole point of the mechanism.
    assert carried.field_path == "RISK-001.severity"


def test_an_engagement_still_sees_its_own_corrections_in_full(isolated_settings):
    store = StateStore()
    _store_correction(store, correction("project-kanpur"))
    recalled = LongTermMemory(store, "project-kanpur").corrections_for_sub_sector(
        "enterprise SaaS")
    assert recalled[0].to_value == "118% NRR"
    assert "step down at renewal" in recalled[0].note


def test_a_redacted_correction_cannot_be_rendered_back_into_values(isolated_settings):
    """Rendering lives on the model so a call site cannot reassemble what was
    dropped - which is exactly how the values reached the prompt before."""
    carried = correction("project-sentinel", shareable=True).redacted()
    rendered = carried.render_for_prompt()
    assert "124%" not in rendered and "118%" not in rendered
    assert "project-sentinel" not in rendered
    assert "Values withheld" in rendered
    assert "RISK-001.severity" in rendered


def test_the_thesis_prompt_carries_no_other_client_figures(isolated_settings, context):
    """End to end: what actually reaches the model."""
    store = context.store
    other = correction("project-sentinel", shareable=True)
    # Match this engagement sub-sector, or the recall returns nothing and the test
    # passes without exercising anything.
    other = other.model_copy(update={"sub_sector": context.profile.sector.sub_sector})
    _store_correction(store, other)
    recalled = context.memory.corrections_for_sub_sector(
        context.profile.sector.sub_sector)
    assert recalled, "the correction must actually be recalled for this to mean anything"
    prompt_lines = " ".join(c.render_for_prompt() for c in recalled)
    for leaked in ("124%", "118% NRR", "project-sentinel", "step down at renewal"):
        assert leaked not in prompt_lines, f"{leaked!r} crossed the boundary"


def test_a_second_data_room_is_refused_from_any_caller(isolated_settings, tmp_path):
    """The guard lived in the web layer first, and the command line walked straight
    past it - so a purge-and-reingest from the CLI silently re-armed the contamination.
    It belongs where every caller passes through.
    """
    from cdd_agent.retrieval.ingestion import (
        DataRoomConflict,
        check_data_room,
        ingest_directory,
    )

    store = StateStore()
    first = tmp_path / "target-a"
    first.mkdir()
    (first / "Board_Deck_2026-01-01.txt").write_text("Revenue grew 20%. " * 40,
                                                    encoding="utf-8")
    report, _ = ingest_directory("deal-1", first, store=store)
    store.put("deal-1", Collection.METRICS, "ingestion",
              {"data_room": report.data_room}, agent="test")

    second = tmp_path / "target-b"
    second.mkdir()
    with pytest.raises(DataRoomConflict, match="already ingested"):
        check_data_room("deal-1", second, store=store)
    # Deliberate replacement stays possible; it just cannot happen by accident.
    assert check_data_room("deal-1", second, store=store, force=True)
    # And re-ingesting the same folder is not a conflict.
    assert check_data_room("deal-1", first, store=store)


def _profile_for(store: StateStore, engagement: str, target: str) -> None:
    from cdd_agent.schemas.deal_profile import (
        BuyerProfile, DealProfile, InvestmentThesis, SectorDefinition,
        TargetIdentification,
    )
    store.put(engagement, Collection.DEAL_PROFILE, "current", DealProfile(
        engagement_id=engagement, created_by="Intake Agent",
        target=TargetIdentification(legal_name=target),
        sector=SectorDefinition(sub_sector="enterprise SaaS"),
        thesis=InvestmentThesis(one_sentence_thesis="Buy it."),
        buyer=BuyerProfile(decision_criteria=["growth"])), agent="test")


def _bind(store: StateStore, engagement: str, room) -> None:
    store.put(engagement, Collection.METRICS, "ingestion",
              {"data_room": str(room.resolve())}, agent="test")


def test_one_target_may_be_diligenced_under_several_structures(isolated_settings, tmp_path):
    """Project Atlas runs one company under two structures against one set of
    documents. Forcing a copy per engagement would let the copies drift, so two decks
    would cite different versions of the same filing while both looked fine."""
    from cdd_agent.retrieval.ingestion import check_data_room

    store = StateStore()
    room = tmp_path / "meridian"
    room.mkdir()
    _profile_for(store, "atlas-tp", "Meridian Data Systems, Inc.")
    _profile_for(store, "atlas-min", "Meridian Data Systems, Inc.")
    _bind(store, "atlas-tp", room)
    # Same target, shared folder: allowed.
    assert check_data_room("atlas-min", room, store=store)


def test_two_companies_may_not_share_a_data_room(isolated_settings, tmp_path):
    """The same contamination arriving from the other direction."""
    from cdd_agent.retrieval.ingestion import DataRoomSharedAcrossTargets, check_data_room

    store = StateStore()
    room = tmp_path / "sentinel"
    room.mkdir()
    _profile_for(store, "project-sentinel", "Sentinel Secure Ltd")
    _profile_for(store, "project-kanpur", "Freshworks Inc.")
    _bind(store, "project-sentinel", room)
    with pytest.raises(DataRoomSharedAcrossTargets, match="Freshworks"):
        check_data_room("project-kanpur", room, store=store)
    # Deliberate override remains possible.
    assert check_data_room("project-kanpur", room, store=store, force=True)


def test_an_engagement_owns_a_folder_by_default(isolated_settings):
    """Isolation should be what happens when nobody thinks about it."""
    from cdd_agent.retrieval.ingestion import default_data_room

    a = default_data_room("project-kanpur")
    b = default_data_room("project-sentinel")
    assert a != b
    assert a.is_dir() and b.is_dir()
    assert "project-kanpur" in str(a)
    # Ids come from the user, so a hostile one must not escape the engagements
    # directory. Containment is the property that matters, not the spelling.
    from cdd_agent.config import get_settings

    root = get_settings().engagements_dir.resolve()
    escaped = default_data_room("../../etc", create=False).resolve()
    assert root in escaped.parents, f"{escaped} escaped {root}"
    assert escaped.name == "data_room"
