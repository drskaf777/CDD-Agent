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
