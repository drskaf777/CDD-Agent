"""Local web API for driving an engagement.

Thin by design: every endpoint calls the same agents the CLI calls, so the interface
cannot do anything the pipeline would not do, and cannot skip a gate. In particular
there is no endpoint that selects a framing without recording who selected it, and
none that synthesises without the output contract running first.

The read model is one snapshot endpoint rather than a dozen fine-grained ones: the
whole point of a diligence UI is that the evidence, the risks, and the trace are read
against each other, and serving them from one consistent read avoids showing a matrix
from one moment beside a register from another.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from cdd_agent.agents.analyst import Analyst
from cdd_agent.agents.base import AgentContext, save_structured_tables
from cdd_agent.agents.intake import IntakeAgent
from cdd_agent.agents.risk_auditor import RiskAuditor
from cdd_agent.agents.synthesizer import Synthesizer
from cdd_agent.agents.thesis_architect import ThesisArchitect
from cdd_agent.config import get_settings
from cdd_agent.evaluation.metrics import evaluate
from cdd_agent.guardrails.authorization import AgentRole
from cdd_agent.guardrails.escalation import (
    check_phase1,
    check_tier1_evidence,
)
from cdd_agent.guardrails.escalation import (
    record as record_escalations,
)
from cdd_agent.knowledge.risk_taxonomy import applicable_categories
from cdd_agent.retrieval.indexes import IndexVersionMismatch
from cdd_agent.retrieval.ingestion import ingest_directory
from cdd_agent.schemas.common import ConfidenceTag, Tier
from cdd_agent.schemas.deal_profile import DealProfile
from cdd_agent.state.store import Collection, StateStore

STATIC = Path(__file__).parent / "static"
DEMO = Path(__file__).resolve().parents[3] / "demo"

app = FastAPI(title="CDD Agent", docs_url="/api/docs")


def _model_error_response(exc: Exception) -> Optional[JSONResponse]:
    """Translate a provider error into something an operator can act on.

    A rejected key, an exhausted balance or a rate limit are operational conditions,
    not bugs. Left unhandled they surface as a 500 and a stack trace in the server log,
    which tells the person running a demo nothing about the one line they need to
    change. Matched on class name and message so this does not hard-depend on any
    provider SDK's exception hierarchy.
    """
    name = type(exc).__name__
    message = str(exc)
    if "Authentication" in name or "authentication_error" in message:
        return JSONResponse(status_code=401, content={"detail":
            "The Anthropic API rejected the key. Update ANTHROPIC_API_KEY - in .env for "
            "the preview server, or in the shell for `cdd serve` - and restart the "
            "server so it is re-read."})
    if "anthropic-workspace-id" in message:
        return JSONResponse(status_code=401, content={"detail":
            "This key is identity-linked and must name a workspace. Set CDD_WORKSPACE_ID "
            "to the workspace id from the Console, then restart the server."})
    if "RateLimit" in name or "rate_limit" in message:
        return JSONResponse(status_code=429, content={"detail":
            "Rate limited by the Anthropic API. Wait a moment and re-run the phase."})
    if "credit balance" in message or "billing" in message.lower():
        return JSONResponse(status_code=402, content={"detail":
            "The key is valid but the account has no credit."})
    return None


@app.exception_handler(Exception)
def _unhandled(request: Any, exc: Exception) -> JSONResponse:
    """Last resort: never answer the interface with a bare Internal Server Error."""
    handled = _model_error_response(exc)
    if handled is not None:
        return handled
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)[:400]}"},
    )


@app.exception_handler(IndexVersionMismatch)
def _index_mismatch(request: Any, exc: IndexVersionMismatch) -> JSONResponse:
    """Surface an unreadable index as an actionable message, not a 500.

    The underlying failure is a Rust panic; leaving it as an Internal Server Error
    tells the operator nothing about the one thing that would fix it.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})
_store = StateStore()
_tables: dict[str, list] = {}


def _ctx(engagement: str) -> AgentContext:
    """Rebuild context per request. Parsed tables are cached per engagement."""
    return AgentContext.create(
        engagement, store=_store, tables=tuple(_tables.get(engagement, []))
    )


# --------------------------------------------------------------------- payloads
class IntakeBody(BaseModel):
    briefing: str = ""
    use_demo_fixture: bool = False
    # Explicit opt-in to discarding an existing Deal Profile Brief.
    force: bool = False


class SelectBody(BaseModel):
    branch_id: str
    approved_by: str


class IngestBody(BaseModel):
    path: str


# ------------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    return FileResponse(STATIC / "index.html")


# ------------------------------------------------------------------- read model
@app.get("/api/engagements")
def engagements() -> dict[str, Any]:
    return {
        "engagements": _store.engagements(),
        "offline": get_settings().offline,
        "model": get_settings().model,
        "demo_available": (DEMO / "deal_profile.json").exists(),
        "demo_data_room": str(DEMO / "data_room"),
    }


@app.get("/api/engagements/{engagement}")
def snapshot(engagement: str) -> dict[str, Any]:
    """One consistent read of everything the interface displays."""
    ctx = _ctx(engagement)
    mem = ctx.memory
    profile = mem.deal_profile()
    search = mem.thesis_search()
    tree = mem.hypothesis_tree()
    checklist = mem.data_request()
    matrix = mem.evidence_matrix()
    register = mem.risk_register()
    deck = mem.deck()

    escalations: list[dict[str, Any]] = []
    if search is not None and tree is None:
        # Once a tree is approved the Phase-1 gate is settled. The stored search may
        # still record the tie that was resolved; replaying it as a live escalation
        # would report the pipeline as blocked when a person has already unblocked it.
        escalations += [e.to_dict() for e in check_phase1(search)]
    if tree is not None:
        escalations += [e.to_dict() for e in check_tier1_evidence(tree, matrix, register)]
    stored = [doc for _, doc in _store.list(engagement, Collection.ESCALATION)]

    applicable = applicable_categories(ctx.deal_shape)
    return {
        "engagement_id": engagement,
        "phases": _phase_states(
            profile, search, tree, checklist, matrix, register, deck,
            ingested=_store.get(engagement, Collection.METRICS, "ingestion"),
        ),
        "profile": profile.model_dump(mode="json") if profile else None,
        "profile_ready": profile.is_ready_for_phase_1()[0] if profile else False,
        "profile_missing": profile.is_ready_for_phase_1()[1] if profile else [],
        "search": _search_view(search),
        "tree": _tree_view(tree, matrix),
        "checklist": _checklist_view(checklist),
        "evidence": _evidence_view(matrix),
        "register": _register_view(register, applicable),
        "deck": deck.model_dump(mode="json") if deck else None,
        "groundedness": deck.groundedness() if deck else None,
        "escalations": escalations,
        "escalation_history": stored,
        "trace": _trace_view(engagement),
        "audit": _audit_view(engagement),
        "tool_scope": _tool_scope(ctx),
    }


# ------------------------------------------------------------------- write model
@app.post("/api/engagements/{engagement}/intake")
def run_intake(engagement: str, body: IntakeBody) -> dict[str, Any]:
    import json

    ctx = _ctx(engagement)
    if body.use_demo_fixture:
        fixture = DEMO / "deal_profile.json"
        if not fixture.exists():
            raise HTTPException(404, "demo fixture not found")
        raw = json.loads(fixture.read_text(encoding="utf-8"))
        raw["engagement_id"] = engagement
        profile = DealProfile.model_validate(raw)
        # Loading the fixture over a real intake destroys it. That happened on a live
        # engagement: the demo profile replaced a Deal Profile Brief built from a real
        # briefing, and the next Phase-1 run decomposed the wrong company against the
        # right evidence - slow, and quietly wrong, which is worse. The fixture may
        # only seed an empty engagement or replace itself.
        existing = ctx.memory.deal_profile()
        if existing is not None and not body.force:
            same = existing.target.legal_name == profile.target.legal_name
            if not same:
                raise HTTPException(
                    409,
                    f"{engagement} already holds a Deal Profile Brief for "
                    f"{existing.target.legal_name}. Loading the demo fixture would "
                    f"replace it with {profile.target.legal_name} and orphan every "
                    f"artifact built from it. Re-run intake from the original "
                    f"briefing instead, or resend with force=true if you meant to "
                    f"discard it.",
                )
        ctx.memory.save_deal_profile(profile, agent="Intake Agent (demo fixture)")
    else:
        if not body.briefing.strip():
            raise HTTPException(400, "supply a briefing, or use the demo fixture")
        profile = IntakeAgent(ctx).run(body.briefing)
    ready, missing = profile.is_ready_for_phase_1()
    return {"ready": ready, "missing": missing}


@app.post("/api/engagements/{engagement}/thesis")
def run_thesis(engagement: str) -> dict[str, Any]:
    ctx = _ctx(engagement)
    if ctx.profile is None:
        raise HTTPException(409, "no Deal Profile Brief - run intake first")
    ready, missing = ctx.profile.is_ready_for_phase_1()
    if not ready:
        raise HTTPException(409, "intake incomplete: " + "; ".join(missing))
    result = ThesisArchitect(ctx).run()
    record_escalations(_store, engagement, check_phase1(result))
    return {"outcome": result.outcome, "requires_human": result.requires_human()}


@app.post("/api/engagements/{engagement}/select")
def select_branch(engagement: str, body: SelectBody) -> dict[str, Any]:
    """Resolve a tie, recover a soft-pruned framing, or clear the approval gate.

    `approved_by` is required and is written into the artifact's provenance: a gate
    cleared by nobody in particular is not a gate.
    """
    ctx = _ctx(engagement)
    architect = ThesisArchitect(ctx)
    result = ctx.memory.thesis_search()
    if result is None:
        raise HTTPException(409, "no Phase-1 search to select from")
    if not body.approved_by.strip():
        raise HTTPException(400, "name the person clearing this gate")
    try:
        if result.selected_branch_id != body.branch_id:
            result = architect.override(result, body.branch_id, approved_by=body.approved_by)
        tree = architect.approve(result, approved_by=body.approved_by)
    except ValueError as exc:
        # A four-question hard prune is not overridable; say so rather than 500.
        raise HTTPException(422, str(exc)) from exc
    return {"selected": tree.branch_id, "approved_by": body.approved_by}


@app.post("/api/engagements/{engagement}/request")
def run_request(engagement: str) -> dict[str, Any]:
    ctx = _ctx(engagement)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise HTTPException(409, "no approved hypothesis tree")
    checklist = Analyst(ctx).generate_data_request(tree)
    return {"items": len(checklist.items)}


@app.post("/api/engagements/{engagement}/ingest")
def run_ingest(engagement: str, body: IngestBody) -> dict[str, Any]:
    directory = Path(body.path)
    if not directory.is_dir():
        raise HTTPException(400, f"not a directory: {directory}")
    report, tables = ingest_directory(engagement, directory)
    _tables[engagement] = list(tables)
    save_structured_tables(_store, engagement, tables)
    _store.put(
        engagement, Collection.METRICS, "ingestion",
        {"summary": report.summary(), "unstructured": report.unstructured,
         "structured": report.structured, "skipped": report.skipped,
         "undated": report.undated},
        agent="Controller",
    )
    return {
        "summary": report.summary(),
        "unstructured": report.unstructured,
        "structured": report.structured,
        "skipped": report.skipped,
        "undated": report.undated,
    }


@app.post("/api/engagements/{engagement}/analyze")
def run_analyze(engagement: str) -> dict[str, Any]:
    ctx = _ctx(engagement)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise HTTPException(409, "no approved hypothesis tree")
    matrix, report = Analyst(ctx).run_evidence_loop(tree)
    record_escalations(
        _store, engagement,
        check_tier1_evidence(tree, matrix, ctx.memory.risk_register()),
    )
    return {
        "steps": len(report.steps),
        "gaps": len(report.gaps_logged),
        "stopped_because": report.stopped_because,
        "blocked": report.blocked_actions,
    }


@app.post("/api/engagements/{engagement}/audit")
def run_audit(engagement: str) -> dict[str, Any]:
    ctx = _ctx(engagement)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise HTTPException(409, "no approved hypothesis tree")
    register, report = RiskAuditor(ctx).audit(tree, ctx.memory.evidence_matrix())
    return {
        "risks": len(report.risks_raised),
        "gaps": len(report.gaps_raised),
        "conflicts": report.conflicts,
        "routed_back": sorted(set(report.routed_back)),
        "coverage": register.coverage(applicable_categories(ctx.deal_shape)),
    }


@app.post("/api/engagements/{engagement}/synthesize")
def run_synthesize(engagement: str) -> dict[str, Any]:
    ctx = _ctx(engagement)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise HTTPException(409, "no approved hypothesis tree")
    try:
        deck, contract = Synthesizer(ctx).run(
            tree, ctx.memory.evidence_matrix(), ctx.memory.risk_register()
        )
    except Exception as exc:
        # A contract violation is the guardrail working. Surface it as a 422 with the
        # violations, not a stack trace.
        raise HTTPException(422, str(exc)) from exc
    metrics = evaluate(
        deck=deck, matrix=ctx.memory.evidence_matrix(),
        register=ctx.memory.risk_register(), tree=tree, store=_store,
        engagement_id=engagement, strategic_buyer=ctx.deal_shape,
    )
    return {
        "sections": len(deck.slides),
        "groundedness": deck.groundedness(),
        "warnings": contract.warnings,
        "metrics": [
            {"name": m.name, "value": m.value, "detail": m.detail,
             "needs_human": m.needs_human, "is_ratio": m.is_ratio}
            for m in metrics.metrics
        ],
    }


@app.get("/api/engagements/{engagement}/export", response_class=HTMLResponse)
def export(engagement: str) -> HTMLResponse:
    from cdd_agent.web.report import render_report

    ctx = _ctx(engagement)
    html = render_report(ctx, standalone=True)
    return HTMLResponse(html)


# ------------------------------------------------------------------- projections
def _phase_states(
    profile, search, tree, checklist, matrix, register, deck, ingested=None
) -> list[dict]:
    def state(done: bool, blocked: bool = False, available: bool = True) -> str:
        if blocked:
            return "blocked"
        if done:
            return "done"
        return "available" if available else "pending"

    phase1_blocked = bool(search and search.requires_human() and tree is None)
    return [
        {"id": "intake", "label": "Intake", "phase": "Phase 0",
         "agent": "Intake Agent", "output": "Deal Profile Brief",
         "state": state(profile is not None)},
        {"id": "thesis", "label": "Thesis", "phase": "Phase 1",
         "agent": "Thesis Architect", "output": "Hypothesis Tree",
         "state": state(tree is not None, phase1_blocked, profile is not None)},
        {"id": "request", "label": "Data request", "phase": "Phase 2",
         "agent": "Analyst", "output": "Prioritized checklist",
         "state": state(checklist is not None, False, tree is not None)},
        # Ingestion is the first half of Phase 3 and the point where the supersession
        # filter runs. Leaving it out of the pipeline hid both, and let the evidence
        # loop be started against an empty index.
        {"id": "ingest", "label": "Ingest data room", "phase": "Phase 3",
         "agent": "Controller", "output": "Classified + indexed documents",
         "state": state(bool(ingested), False, tree is not None)},
        {"id": "analyze", "label": "Evidence loop", "phase": "Phase 3",
         "agent": "Analyst", "output": "Evidence Matrix",
         "state": state(bool(matrix.items), False, bool(ingested))},
        {"id": "audit", "label": "Risk audit", "phase": "Phase 3",
         "agent": "Risk Auditor", "output": "Risk Register",
         "state": state(bool(register.risks), False, bool(matrix.items))},
        {"id": "synthesize", "label": "Synthesis", "phase": "Phase 4",
         "agent": "Synthesizer", "output": "Draft presentation",
         "state": state(deck is not None, False, bool(matrix.items))},
    ]


def _search_view(search) -> Optional[dict[str, Any]]:
    if search is None:
        return None
    return {
        "outcome": search.outcome,
        "requires_human": search.requires_human(),
        "clarifying_question": search.clarifying_question,
        "tied_branch_ids": search.tied_branch_ids,
        "selected_branch_id": search.selected_branch_id,
        "branches": [
            {
                "branch_id": b.branch_id,
                "framing_label": b.framing_label,
                "selected": b.selected,
                "pruned": b.pruned,
                "prune_reason": b.prune_reason,
                "recoverable": b.prunable_pending_override,
                "human_approved": b.human_approved,
                "four_question_passed": bool(b.score and b.score.four_question.passed),
                "four_question_unmapped": b.score.four_question.unmapped() if b.score else [],
                "average": b.score.average if b.score else None,
                "scores": {
                    "buyer_criteria_coverage": b.score.buyer_criteria_coverage,
                    "four_question_alignment": b.score.four_question_alignment,
                    "sub_sector_fit": b.score.sub_sector_fit,
                    "testability": b.score.testability,
                } if b.score else {},
                "criterion_notes": b.score.criterion_notes if b.score else {},
                "tier_1": [
                    {"id": h.id, "statement": h.statement,
                     "required_evidence": h.required_evidence}
                    for h in b.tier_1()
                ],
            }
            for b in search.branches
        ],
    }


def _tree_view(tree, matrix) -> Optional[dict[str, Any]]:
    if tree is None:
        return None
    return {
        "branch_id": tree.branch_id,
        "framing_label": tree.framing_label,
        "root_thesis": tree.root_thesis,
        "human_approved": tree.human_approved,
        "created_by": tree.created_by,
        "tier_1": [
            {
                "id": h.id,
                "statement": h.statement,
                "rating": matrix.rating(h.id).value,
                "items": len(matrix.for_hypothesis(h.id)),
                "triangulated": matrix.triangulated(h.id),
                "required_evidence": h.required_evidence,
                "assumptions": [c.statement for c in tree.children_of(h.id)],
            }
            for h in tree.tier_1()
        ],
    }


def _checklist_view(checklist) -> Optional[dict[str, Any]]:
    if checklist is None:
        return None
    return {
        "items": [
            {"id": i.id, "category": i.category, "item": i.item, "tier": int(i.tier),
             "hypothesis_ids": i.hypothesis_ids, "rationale": i.rationale,
             "sub_sector_specific": i.sub_sector_specific, "received": i.received}
            for i in checklist.items
        ],
        "counts": {str(int(t)): len(checklist.by_tier(t)) for t in Tier},
    }


def _evidence_view(matrix) -> dict[str, Any]:
    return {
        "items": [
            {
                "id": i.id,
                "hypothesis_id": i.hypothesis_id,
                "claim": i.claim,
                "tag": i.tag.value,
                "source_kind": i.source_kind.value,
                "independent": i.is_independent,
                "query": i.query,
                "created_by": i.created_by,
                "created_at": i.created_at.isoformat(),
                "citations": [
                    {"source_file": c.source_file, "locator": c.locator,
                     "date": c.document_date.isoformat() if c.document_date else None,
                     "similarity": c.similarity, "kind": c.source_kind.value,
                     "quote": c.quoted_text[:400]}
                    for c in i.citations
                ],
            }
            for i in matrix.items
        ],
        "counts": {
            tag.value: sum(1 for i in matrix.items if i.tag is tag)
            for tag in ConfidenceTag
        },
    }


def _register_view(register, applicable) -> dict[str, Any]:
    return {
        "risks": [
            {"id": r.id, "category": r.category.value, "description": r.description,
             "severity": r.severity, "likelihood": r.likelihood, "score": r.score,
             "management_data_only": r.management_data_only,
             "hypothesis_ids": r.hypothesis_ids, "evidence_ids": r.evidence_ids}
            for r in register.ranked()
        ],
        "gaps": [
            {"id": g.id, "request": g.request, "owner": g.owner.value,
             "hypothesis_id": g.hypothesis_id,
             "target_close_date": g.target_close_date.isoformat()
             if g.target_close_date else None,
             "blocking": g.blocking, "confirmatory": g.carried_to_confirmatory,
             "resolved": g.resolved}
            for g in register.gaps
        ],
        "source_conflicts": register.source_conflicts,
        "coverage": register.coverage(applicable),
        "uncovered": [
            c.value for c in applicable if c not in register.categories_evaluated()
        ],
    }


def _trace_view(engagement: str) -> list[dict[str, Any]]:
    return [doc for _, doc in _store.list(engagement, Collection.TRACE)]


def _audit_view(engagement: str) -> list[dict[str, Any]]:
    return [
        {"seq": e.seq, "at": e.at.isoformat(), "agent": e.agent,
         "collection": e.collection, "key": e.key, "action": e.action,
         "digest": e.payload_digest}
        for e in _store.audit(engagement, limit=400)
    ]


def _tool_scope(ctx: AgentContext) -> list[dict[str, Any]]:
    """What each role may actually do on this engagement, and why not."""
    out = []
    for role in AgentRole:
        bundle = ctx.tools_for(role) if ctx.registry else None
        out.append({
            "role": role.value,
            "allowed": bundle.names() if bundle else [],
            "denied": bundle.denied if bundle else {},
        })
    return out


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
