"""The web API and the report renderer.

The API's job is to expose the pipeline without weakening it, so these tests are
mostly about what it *refuses*: phases out of order, a gate cleared by nobody, a
framing pruned on the four-question hard check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DEMO = Path(__file__).resolve().parents[1] / "demo"


@pytest.fixture
def client(isolated_settings, monkeypatch):
    # The API module holds a module-level store bound at import time; rebind it to the
    # per-test database so tests cannot see each other's engagements.
    from cdd_agent.state.store import StateStore
    from cdd_agent.web import api

    monkeypatch.setattr(api, "_store", StateStore())
    monkeypatch.setattr(api, "_tables", {})
    with TestClient(api.app) as c:
        yield c


def _seed(client, engagement="web-test"):
    r = client.post(f"/api/engagements/{engagement}/intake", json={"use_demo_fixture": True})
    assert r.status_code == 200, r.text
    return engagement


def test_index_serves_the_app(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "CDD Agent" in r.text


def test_snapshot_of_an_empty_engagement_is_well_formed(client):
    body = client.get("/api/engagements/nothing-here").json()
    assert body["profile"] is None
    assert body["tree"] is None
    assert [p["state"] for p in body["phases"]][0] == "available"
    assert body["trace"] == [] and body["evidence"]["items"] == []


def test_thesis_refuses_to_run_before_intake(client):
    r = client.post("/api/engagements/no-profile/thesis")
    assert r.status_code == 409
    assert "Deal Profile" in r.json()["detail"]


def test_phases_advance_and_the_gate_holds(client):
    eng = _seed(client)
    snap = client.get(f"/api/engagements/{eng}").json()
    assert snap["profile_ready"] is True

    r = client.post(f"/api/engagements/{eng}/thesis").json()
    assert r["requires_human"] is True  # offline scores land inside the tie band

    snap = client.get(f"/api/engagements/{eng}").json()
    phases = {p["id"]: p["state"] for p in snap["phases"]}
    assert phases["thesis"] == "blocked"
    assert any(e["trigger"] == "phase1_tie_or_all_pruned" for e in snap["escalations"])
    assert len(snap["search"]["branches"]) == 3


def test_gate_requires_a_named_signatory(client):
    eng = _seed(client, "sig-test")
    client.post(f"/api/engagements/{eng}/thesis")
    r = client.post(f"/api/engagements/{eng}/select",
                    json={"branch_id": "risk", "approved_by": "   "})
    assert r.status_code == 400
    assert "name the person" in r.json()["detail"]


def test_selecting_a_framing_records_who_cleared_the_gate(client):
    eng = _seed(client, "select-test")
    client.post(f"/api/engagements/{eng}/thesis")
    r = client.post(f"/api/engagements/{eng}/select",
                    json={"branch_id": "risk", "approved_by": "M. A. Skaf"})
    assert r.status_code == 200
    snap = client.get(f"/api/engagements/{eng}").json()
    assert snap["tree"]["branch_id"] == "risk"
    assert snap["tree"]["human_approved"] is True
    assert "M. A. Skaf" in snap["tree"]["created_by"]


def test_full_run_through_the_api_populates_the_trace(client):
    eng = _seed(client, "trace-test")
    client.post(f"/api/engagements/{eng}/thesis")
    client.post(f"/api/engagements/{eng}/select",
                json={"branch_id": "risk", "approved_by": "reviewer"})
    client.post(f"/api/engagements/{eng}/request")
    ing = client.post(f"/api/engagements/{eng}/ingest",
                      json={"path": str(DEMO / "data_room")}).json()
    assert ing["unstructured"] and not ing["undated"]

    loop = client.post(f"/api/engagements/{eng}/analyze").json()
    assert loop["steps"] > 0

    snap = client.get(f"/api/engagements/{eng}").json()
    # The trace records the reasoning, not just that the matrix changed.
    assert len(snap["trace"]) == loop["steps"]
    first = snap["trace"][0]
    assert first["thought"] and first["action"] and first["hypothesis_id"]
    # Every state write is attributed; none is anonymous.
    assert snap["audit"] and all(a["agent"] for a in snap["audit"])

    audit = client.post(f"/api/engagements/{eng}/audit").json()
    assert 0.0 <= audit["coverage"] <= 1.0

    synth = client.post(f"/api/engagements/{eng}/synthesize").json()
    assert synth["sections"] == 9
    # Groundedness counts assertive claims only. Under the test embedder nothing clears
    # the similarity floor, so every claim is a logged gap and there is nothing
    # ungrounded to find - which is the correct reading, not a vacuous pass.
    assert synth["groundedness"] == 1.0


def test_ingest_rejects_a_path_that_is_not_a_directory(client):
    r = client.post("/api/engagements/x/ingest", json={"path": str(DEMO / "briefing.md")})
    assert r.status_code == 400


def test_report_renders_in_both_modes(client):
    eng = _seed(client, "report-test")
    client.post(f"/api/engagements/{eng}/thesis")
    client.post(f"/api/engagements/{eng}/select",
                json={"branch_id": "risk", "approved_by": "reviewer"})
    client.post(f"/api/engagements/{eng}/ingest", json={"path": str(DEMO / "data_room")})
    client.post(f"/api/engagements/{eng}/analyze")
    client.post(f"/api/engagements/{eng}/audit")
    client.post(f"/api/engagements/{eng}/synthesize")

    r = client.get(f"/api/engagements/{eng}/export")
    assert r.status_code == 200
    assert r.text.startswith("<!doctype html>")
    assert "Draft for partner/MD review" in r.text
    assert "Trace and audit history" in r.text

    from cdd_agent.agents.base import AgentContext
    from cdd_agent.web import api
    from cdd_agent.web.report import render_report

    ctx = AgentContext.create(eng, store=api._store)
    embedded = render_report(ctx, standalone=False)
    # Artifact mode omits the document shell; the host supplies it.
    assert not embedded.lstrip().startswith("<!doctype")
    assert embedded.lstrip().startswith("<title>")

    # Both themes are defined at token level, never only inside a media query.
    assert ':root[data-theme="dark"]' in embedded
    assert "prefers-color-scheme" in embedded


def test_the_demo_fixture_cannot_overwrite_a_real_deal_profile(client):
    """This happened on a live engagement, and was expensive.

    The Intake control read "Reload profile" once intake was done, which sounds like
    re-reading the current brief. It posted the demo fixture. A Deal Profile Brief
    built from a real briefing was replaced with a different company, and the next
    Phase-1 run decomposed the wrong target against the right evidence - slow, and
    quietly wrong, which is the worse half.
    """
    engagement = "overwrite-test"
    real = (
        "Target is Northbeam Analytics Ltd, privately held, majority buyout. "
        "Sub-sector is B2B analytics SaaS. The thesis in one sentence: acquire "
        "Northbeam and cross-sell into the installed base. Buyer is a financial "
        "sponsor and cares most about cash-flow stability."
    )
    r = client.post(f"/api/engagements/{engagement}/intake", json={"briefing": real})
    assert r.status_code == 200, r.text

    before = client.get(f"/api/engagements/{engagement}").json()["profile"]
    r = client.post(f"/api/engagements/{engagement}/intake",
                    json={"use_demo_fixture": True})
    assert r.status_code == 409, "the fixture must not silently replace a real brief"
    assert "already holds a Deal Profile Brief" in r.text

    after = client.get(f"/api/engagements/{engagement}").json()["profile"]
    assert after["target"]["legal_name"] == before["target"]["legal_name"]

    # Explicit intent still works, because sometimes you do mean to discard it.
    r = client.post(f"/api/engagements/{engagement}/intake",
                    json={"use_demo_fixture": True, "force": True})
    assert r.status_code == 200, r.text


def test_the_fixture_may_still_seed_an_empty_engagement(client):
    r = client.post("/api/engagements/fresh-one/intake", json={"use_demo_fixture": True})
    assert r.status_code == 200, r.text
    # And re-loading it over itself is a no-op, not a conflict.
    r = client.post("/api/engagements/fresh-one/intake", json={"use_demo_fixture": True})
    assert r.status_code == 200, r.text
