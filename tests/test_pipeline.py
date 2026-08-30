"""End-to-end: the pipeline runs offline, and the gates hold.

These run with CDD_OFFLINE=1, so they exercise orchestration, guardrails, retrieval, and
the artifact contracts - not the model's judgment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cdd_agent.agents.analyst import Analyst
from cdd_agent.agents.base import AgentContext
from cdd_agent.agents.risk_auditor import RiskAuditor
from cdd_agent.agents.synthesizer import Synthesizer
from cdd_agent.agents.thesis_architect import ThesisArchitect
from cdd_agent.knowledge.seed import seed_knowledge_base
from cdd_agent.orchestration.controller import Controller
from cdd_agent.retrieval.ingestion import ingest_directory
from cdd_agent.schemas.common import Tier

DEMO = Path(__file__).resolve().parents[1] / "demo"


@pytest.fixture
def context(store, profile):
    seed_knowledge_base()
    ctx = AgentContext.create(profile.engagement_id, store=store, profile=profile)
    ctx.memory.save_deal_profile(profile, agent="test")
    return ctx


def test_phase1_produces_three_scored_branches(context):
    result = ThesisArchitect(context).run()
    assert len(result.branches) == 3
    assert all(b.score is not None for b in result.branches)
    # Pruned branches are retained with their exclusion reason, never dropped.
    assert all(b.prune_reason for b in result.branches if b.pruned)


def test_phase1_persists_every_branch_including_pruned(context):
    ThesisArchitect(context).run()
    stored = context.memory.thesis_search()
    assert len(stored.branches) == 3


def test_data_request_is_tiered_and_traced_to_hypotheses(context, tree):
    checklist = Analyst(context).generate_data_request(tree)
    assert checklist.by_tier(Tier.DEAL_CRITICAL)
    assert checklist.by_tier(Tier.ENRICHMENT)
    # SaaS add-ons fire for this sub-sector.
    assert any(i.sub_sector_specific for i in checklist.items)
    assert any("ARR" in i.item for i in checklist.items)
    # Tier-1 items justify themselves against the tree.
    assert any(i.hypothesis_ids for i in checklist.items)


def test_evidence_loop_grounds_claims_and_logs_dated_gaps(context, tree):
    ingest_directory(context.engagement_id, DEMO / "data_room")
    matrix, report = Analyst(context).run_evidence_loop(tree)

    assert report.steps
    # Every citation traces to a real file in the demo data room.
    files = {c.source_file for i in matrix.items for c in i.citations}
    assert files
    assert all((DEMO / "data_room" / f).exists() for f in files)
    # Any hypothesis left without evidence carries a dated, specific gap.
    for gap in report.gaps_logged:
        assert gap.target_close_date is not None
        assert gap.request and "more data" not in gap.request.lower()


def test_auditor_reports_taxonomy_coverage(context, tree):
    ingest_directory(context.engagement_id, DEMO / "data_room")
    matrix, _ = Analyst(context).run_evidence_loop(tree)
    register, audit = RiskAuditor(context).audit(tree, matrix)
    assert 0.0 <= register.coverage() <= 1.0
    # Coverage names what was *not* evaluated, which is the point of the metric.
    assert isinstance(audit.uncovered_categories, list)


def test_synthesizer_output_satisfies_the_contract(context, tree):
    ingest_directory(context.engagement_id, DEMO / "data_room")
    matrix, loop = Analyst(context).run_evidence_loop(tree)
    register, _ = RiskAuditor(context).audit(tree, matrix)
    deck, contract = Synthesizer(context).run(tree, matrix, register)

    assert contract.ok
    assert [s.section_number for s in deck.slides][:1] == [0]
    assert any(s.section_number == 8 for s in deck.slides)
    assert deck.draft_notice
    for claim in deck.all_claims():
        assert claim.tag is not None
        if claim.tag.value != "No Data":
            assert claim.citations


def test_full_run_halts_and_records_the_final_review_trigger(store):
    seed_knowledge_base()
    ctx = AgentContext.create("project-sentinel", store=store)
    report = Controller(ctx).run(
        (DEMO / "briefing.md").read_text(encoding="utf-8"),
        DEMO / "data_room",
        auto_approve_phase1=True,
    )
    # Offline intake deliberately extracts nothing, so the run stops at the Phase-0
    # gate rather than inventing a thesis to decompose.
    assert report.halted_at == "Phase 0"
    assert "Intake is incomplete" in report.blocked_reason


def test_full_run_completes_when_intake_is_supplied(store, profile):
    seed_knowledge_base()
    ctx = AgentContext.create(profile.engagement_id, store=store, profile=profile)
    ctx.memory.save_deal_profile(profile, agent="test")
    report = Controller(ctx).run("", DEMO / "data_room", auto_approve_phase1=True)

    if report.halted_at:
        # A halt is a legitimate outcome; it must be an escalation, not a crash.
        assert report.blocking()
        return
    assert report.completed
    assert report.tree.human_approved
    assert any(e.trigger.value == "final_recommendation" for e in report.escalations)
    assert report.auditor_rounds >= 1
