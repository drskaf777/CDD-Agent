"""Command-line interface.

Phase-per-command by default, because the engagement is meant to be run with a human at
the gates: `cdd intake`, `cdd thesis`, `cdd approve`, `cdd request`, `cdd ingest`,
`cdd analyze`, `cdd audit`, `cdd synthesize`. `cdd run` chains them for demos and CI and
still stops at every escalation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cdd_agent.agents.analyst import Analyst
from cdd_agent.agents.base import AgentContext
from cdd_agent.agents.intake import IntakeAgent
from cdd_agent.agents.risk_auditor import RiskAuditor
from cdd_agent.agents.synthesizer import Synthesizer
from cdd_agent.agents.thesis_architect import ThesisArchitect
from cdd_agent.config import get_settings
from cdd_agent.evaluation.metrics import evaluate
from cdd_agent.guardrails.escalation import check_phase1, record as record_escalations
from cdd_agent.knowledge.intake_questions import INTAKE_PROTOCOL
from cdd_agent.knowledge.risk_taxonomy import applicable_categories
from cdd_agent.orchestration.controller import Controller
from cdd_agent.retrieval.ingestion import ingest_directory
from cdd_agent.schemas.common import Tier
from cdd_agent.state.store import Collection, StateStore
from cdd_agent.synthesis.render import write_markdown

app = typer.Typer(
    add_completion=False,
    help="AI-based commercial due diligence agent. Output is always a draft for review.",
)
console = Console()


def _context(engagement: str, data_room: Optional[Path] = None) -> AgentContext:
    tables: tuple = ()
    if data_room:
        _, parsed = ingest_directory(engagement, data_room)
        tables = tuple(parsed)
    return AgentContext.create(engagement, tables=tables)


# --------------------------------------------------------------------- Phase 0
@app.command()
def questions() -> None:
    """Print the diagnostic intake protocol (design spec s III)."""
    for category in INTAKE_PROTOCOL:
        flag = "  [required for Phase 1]" if category.required_for_phase_1 else ""
        console.print(f"\n[bold]{category.key}. {category.title}[/bold]{flag}")
        for q in category.questions:
            console.print(f"  - {q}")


@app.command()
def intake(
    engagement: str = typer.Argument(..., help="Engagement id, e.g. project-atlas"),
    briefing: Optional[Path] = typer.Option(None, help="File containing the deal briefing."),
    text: Optional[str] = typer.Option(None, help="Briefing text, inline."),
) -> None:
    """Phase 0 - produce the Deal Profile Brief."""
    body = text or (briefing.read_text(encoding="utf-8") if briefing else "")
    if not body.strip():
        raise typer.BadParameter("supply --text or --briefing")
    ctx = _context(engagement)
    profile = IntakeAgent(ctx).run(body)
    console.print(Panel(f"[bold]{profile.target.legal_name}[/bold]\n"
                        f"Sub-sector: {profile.sector.sub_sector or '(not stated)'}\n"
                        f"Thesis: {profile.thesis.one_sentence_thesis or '(not stated)'}",
                        title="Deal Profile Brief"))
    ready, missing = profile.is_ready_for_phase_1()
    if ready:
        console.print("[green]Ready for Phase 1.[/green]")
    else:
        console.print("[yellow]Phase 1 blocked until these are answered:[/yellow]")
        for m in missing:
            console.print(f"  - {m}")
    if profile.open_intake_questions:
        console.print(f"\n{len(profile.open_intake_questions)} open intake question(s).")


# --------------------------------------------------------------------- Phase 1
@app.command()
def thesis(engagement: str) -> None:
    """Phase 1 - Tree-of-Thought beam search over candidate hypothesis trees."""
    ctx = _context(engagement)
    result = ThesisArchitect(ctx).run()

    table = Table(title="Candidate framings (beam width 3)")
    for col in ("Branch", "Framing", "4Q", "Avg", "Status", "Reason"):
        table.add_column(col, overflow="fold")
    for branch in result.branches:
        score = branch.score
        table.add_row(
            branch.branch_id,
            branch.framing_label,
            "pass" if score and score.four_question.passed else "FAIL",
            f"{score.average:.2f}" if score else "-",
            "SELECTED" if branch.selected else ("pruned" if branch.pruned else "-"),
            branch.prune_reason or "",
        )
    console.print(table)

    escalations = check_phase1(result)
    record_escalations(ctx.store, engagement, escalations)
    for escalation in escalations:
        console.print(Panel(escalation.message, title="Human decision required",
                            border_style="yellow"))
    selected = result.selected()
    if selected:
        console.print(f"\n[bold]{selected.framing_label}[/bold] - Tier-1 hypotheses:")
        for h in selected.tier_1():
            console.print(f"  [{h.id}] {h.statement}")
        console.print(
            f"\nRun [bold]cdd approve {engagement}[/bold] to clear the Phase-1 gate."
        )


@app.command()
def approve(
    engagement: str,
    by: str = typer.Option(..., help="Who is approving."),
    branch: Optional[str] = typer.Option(None, help="Override: select this branch id."),
) -> None:
    """Clear the Phase-1 human-approval gate, or override the Controller's choice."""
    ctx = _context(engagement)
    architect = ThesisArchitect(ctx)
    result = ctx.memory.thesis_search()
    if result is None:
        raise typer.BadParameter("no Phase-1 search found; run `cdd thesis` first")
    if branch:
        result = architect.override(result, branch, approved_by=by)
    tree = architect.approve(result, approved_by=by)
    console.print(f"[green]Approved[/green] {tree.framing_label} ({tree.branch_id}) by {by}.")


# --------------------------------------------------------------------- Phase 2
@app.command()
def request(engagement: str) -> None:
    """Phase 2 - generate the prioritized data-request checklist."""
    ctx = _context(engagement)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise typer.BadParameter("no approved hypothesis tree; run `cdd thesis` and `cdd approve`")
    checklist = Analyst(ctx).generate_data_request(tree)
    for tier in (Tier.DEAL_CRITICAL, Tier.DEPTH_BUILDING, Tier.ENRICHMENT):
        items = checklist.by_tier(tier)
        console.print(f"\n[bold]Tier {int(tier)}[/bold] ({len(items)} items)")
        for item in items:
            mark = " *" if item.sub_sector_specific else ""
            links = f"  <- {', '.join(item.hypothesis_ids)}" if item.hypothesis_ids else ""
            console.print(f"  [{item.id}] {item.category}: {item.item}{mark}{links}")
    console.print("\n[dim]* sub-sector specific[/dim]")


# ------------------------------------------------------------------- Phase 3-4
@app.command()
def ingest(engagement: str, data_room: Path) -> None:
    """Phase 3 - classify, chunk, and index a data-room folder."""
    report, tables = ingest_directory(engagement, data_room)
    console.print(report.summary())
    for entry in report.unstructured:
        console.print(f"  {entry['file']}: {entry['doc_type']}, tier {entry['tier']}, "
                      f"{entry['chunks']} chunks, dated {entry['date'] or 'UNDATED'}")
    for entry in report.structured:
        console.print(f"  {entry['file']}: table, {entry['rows']} rows")
    if report.undated:
        console.print(f"[yellow]Undated ({len(report.undated)}): supersession cannot be "
                      f"resolved for these.[/yellow]")
    for skip in report.skipped:
        console.print(f"[dim]skipped {skip['file']}: {skip['reason']}[/dim]")


@app.command()
def analyze(engagement: str, data_room: Optional[Path] = typer.Option(None)) -> None:
    """Phase 3 - run the ReAct evidence loop."""
    ctx = _context(engagement, data_room)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise typer.BadParameter("no approved hypothesis tree")
    matrix, report = Analyst(ctx).run_evidence_loop(tree)
    console.print(report.summary())
    for h in tree.tier_1():
        console.print(f"  [{h.id}] {matrix.rating(h.id).value}"
                      f"  ({len(matrix.for_hypothesis(h.id))} item(s))")


@app.command()
def audit(engagement: str) -> None:
    """Risk Auditor - screen the Evidence Matrix and maintain the Risk Register."""
    ctx = _context(engagement)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise typer.BadParameter("no approved hypothesis tree")
    register, report = RiskAuditor(ctx).audit(tree, ctx.memory.evidence_matrix())
    console.print(report.summary())
    table = Table(title="Risk register (severity x likelihood)")
    for col in ("ID", "Category", "Finding", "Score"):
        table.add_column(col, overflow="fold")
    for risk in register.ranked()[:15]:
        table.add_row(risk.id, risk.category.value, risk.description, str(risk.score))
    console.print(table)
    if report.routed_back:
        console.print(f"[yellow]Routed back to the Analyst: "
                      f"{', '.join(sorted(set(report.routed_back)))}[/yellow]")
    applicable = applicable_categories(ctx.is_strategic_buyer)
    console.print(f"Taxonomy coverage: {register.coverage(applicable):.0%}")


@app.command()
def synthesize(
    engagement: str,
    out: Optional[Path] = typer.Option(None, help="Write the draft markdown here."),
) -> None:
    """Phase 4 - populate the enhanced master outline from the artifacts."""
    ctx = _context(engagement)
    tree = ctx.memory.hypothesis_tree()
    if tree is None:
        raise typer.BadParameter("no approved hypothesis tree")
    deck, contract = Synthesizer(ctx).run(
        tree, ctx.memory.evidence_matrix(), ctx.memory.risk_register()
    )
    console.print(f"{len(deck.slides)} sections, groundedness {deck.groundedness():.0%}")
    for warning in contract.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    target = out or Path("demo/output") / f"{engagement}-draft.md"
    write_markdown(deck, target)
    console.print(f"Draft written to {target}")


# ----------------------------------------------------------------- whole thing
@app.command()
def run(
    engagement: str,
    briefing: Path = typer.Option(..., help="File containing the deal briefing."),
    data_room: Optional[Path] = typer.Option(None),
    approve_phase1: bool = typer.Option(
        False, help="Clear the Phase-1 gate automatically (demo/CI only)."
    ),
    out: Optional[Path] = typer.Option(None),
) -> None:
    """Run Phase 0-4, halting at any gate that requires a human."""
    ctx = _context(engagement)
    controller = Controller(ctx)
    report = controller.run(
        briefing.read_text(encoding="utf-8"),
        data_room,
        auto_approve_phase1=approve_phase1,
        approver="cdd run --approve-phase1",
    )
    console.print(Panel(report.summary(), title="Run report"))
    if report.deck is not None:
        target = out or Path("demo/output") / f"{engagement}-draft.md"
        write_markdown(report.deck, target)
        console.print(f"Draft written to {target}")
        metrics = evaluate(
            deck=report.deck,
            matrix=report.matrix,
            register=report.register,
            tree=report.tree,
            store=ctx.store,
            engagement_id=engagement,
            strategic_buyer=ctx.is_strategic_buyer,
            timings=[(t.phase, t.seconds) for t in report.timings],
        )
        console.print(Panel(metrics.render(), title="Evaluation metrics"))


@app.command()
def demo(
    engagement: str = typer.Option("project-sentinel", help="Engagement id to use."),
    pick: Optional[str] = typer.Option(
        None, help="Resolve a Phase-1 tie by choosing this branch (growth|margin|risk)."
    ),
    out: Optional[Path] = typer.Option(None, help="Where to write the draft."),
    reset: bool = typer.Option(True, help="Start the engagement from scratch."),
) -> None:
    """Run the Project Sentinel demo end to end.

    Loads the Deal Profile Brief from demo/deal_profile.json rather than running
    Phase-0 intake: offline mode has no model to extract a thesis from prose, and
    inventing one would defeat the point of the intake gate. Everything after that
    is the real pipeline.
    """
    import json

    from cdd_agent.knowledge.seed import seed_knowledge_base
    from cdd_agent.schemas.deal_profile import DealProfile

    demo_dir = Path(__file__).resolve().parents[2] / "demo"
    profile_file = demo_dir / "deal_profile.json"
    data_room = demo_dir / "data_room"
    if not profile_file.exists():
        raise typer.BadParameter(f"missing demo fixture {profile_file}")

    store = StateStore()
    if reset:
        store.purge_engagement(engagement, agent="cdd demo", keep_audit=False)

    console.print("[bold]Seeding the cross-engagement knowledge base[/bold]")
    total = sum(seed_knowledge_base().values())
    console.print(f"  {total} reference chunk(s) indexed\n")

    raw = json.loads(profile_file.read_text(encoding="utf-8"))
    raw["engagement_id"] = engagement
    profile = DealProfile.model_validate(raw)
    ctx = AgentContext.create(engagement, store=store, profile=profile)
    ctx.memory.save_deal_profile(profile, agent="Intake Agent (demo fixture)")
    console.print(f"[bold]Phase 0[/bold] - Deal Profile Brief: {profile.target.legal_name}")
    console.print(f"  Thesis: {profile.thesis.one_sentence_thesis}")
    console.print(f"  Ready for Phase 1: {profile.is_ready_for_phase_1()[0]}\n")

    console.print("[bold]Phase 1[/bold] - Tree-of-Thought beam search")
    architect = ThesisArchitect(ctx)
    result = architect.run()
    table = Table(show_header=True)
    for col in ("Branch", "Framing", "4Q", "Avg", "Status"):
        table.add_column(col)
    for branch in result.branches:
        table.add_row(
            branch.branch_id,
            branch.framing_label,
            "pass" if branch.score and branch.score.four_question.passed else "FAIL",
            f"{branch.score.average:.2f}" if branch.score else "-",
            "SELECTED" if branch.selected else ("pruned" if branch.pruned else "-"),
        )
    console.print(table)

    escalations = check_phase1(result)
    record_escalations(store, engagement, escalations)
    for escalation in escalations:
        console.print(Panel(escalation.message, title="Human decision required",
                            border_style="yellow"))
    if result.requires_human():
        if not pick:
            console.print(
                "\n[yellow]The pipeline stops here by design.[/yellow] Phase-1 ties are "
                "not auto-resolved by reranking - a person chooses. Re-run with the "
                "branch you want:\n\n  cdd demo --pick risk --no-reset\n"
            )
            raise typer.Exit(code=0)
        result = architect.override(result, pick, approved_by="demo operator")
        console.print(f"[green]Operator selected[/green] branch {pick!r}\n")

    tree = architect.approve(result, approved_by="demo operator")

    console.print("[bold]Phase 2[/bold] - tailored data request")
    checklist = Analyst(ctx).generate_data_request(tree)
    console.print(
        f"  {len(checklist.items)} items; "
        f"{len(checklist.by_tier(Tier.DEAL_CRITICAL))} Tier-1, "
        f"{sum(1 for i in checklist.items if i.sub_sector_specific)} sub-sector specific\n"
    )

    console.print("[bold]Phase 3[/bold] - ingestion and the ReAct evidence loop")
    report, tables = ingest_directory(engagement, data_room)
    console.print(f"  {report.summary()}")
    if ctx.registry is not None:
        ctx.registry.tables = list(tables)
    matrix, loop = Analyst(ctx).run_evidence_loop(tree)
    console.print(f"  {loop.summary()}")
    for h in tree.tier_1():
        console.print(f"    [{h.id}] {matrix.rating(h.id).value}")

    console.print("\n[bold]Phase 3[/bold] - Risk Auditor")
    register, audit_report = RiskAuditor(ctx).audit(tree, matrix)
    console.print(f"  {audit_report.summary()}")
    console.print(
        "  taxonomy coverage: "
        f"{register.coverage(applicable_categories(ctx.is_strategic_buyer)):.0%}"
    )

    console.print("\n[bold]Phase 4[/bold] - synthesis")
    deck, contract = Synthesizer(ctx).run(tree, matrix, register)
    target = out or demo_dir / "output" / f"{engagement}-draft.md"
    write_markdown(deck, target)
    console.print(f"  {len(deck.slides)} sections, groundedness {deck.groundedness():.0%}")
    for warning in contract.warnings[:3]:
        console.print(f"  [yellow]warning:[/yellow] {warning}")

    metrics = evaluate(
        deck=deck, matrix=matrix, register=register, tree=tree, store=store,
        engagement_id=engagement, strategic_buyer=ctx.is_strategic_buyer,
    )
    console.print(Panel(metrics.render(), title="Evaluation metrics"))
    console.print(
        Panel(
            f"Draft written to {target}\n\n"
            "This is a working draft for partner/MD review, not an IC recommendation. "
            "Offline mode produced it, so it exercises the machinery and carries no "
            "judgment - see the README.",
            title="Done",
            border_style="green",
        )
    )


# ------------------------------------------------------------------ interface
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address. Localhost by default."),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on source changes."),
) -> None:
    """Start the local web interface.

    Binds to localhost by default: an engagement's data room is client-confidential,
    and this server has no authentication.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "uvicorn is missing; reinstall with `pip install -e .`"
        ) from exc
    console.print(f"[bold]CDD Agent[/bold] on http://{host}:{port}")
    uvicorn.run("cdd_agent.web.api:app", host=host, port=port, reload=reload)


@app.command()
def export(
    engagement: str,
    out: Optional[Path] = typer.Option(None, help="Where to write the HTML report."),
    no_trace: bool = typer.Option(False, help="Omit the trace and audit history."),
) -> None:
    """Render the engagement as a self-contained, shareable HTML report."""
    from cdd_agent.web.report import render_report

    ctx = _context(engagement)
    html = render_report(ctx, standalone=True, include_trace=not no_trace)
    target = out or Path("demo/output") / f"{engagement}-report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    console.print(f"Report written to {target} ({len(html):,} bytes)")


# ------------------------------------------------------------------- utilities
@app.command("seed-kb")
def seed_kb() -> None:
    """Seed the cross-engagement Knowledge-Base Index from the built-in corpus."""
    from cdd_agent.knowledge.seed import seed_knowledge_base

    counts = seed_knowledge_base()
    for topic, n in counts.items():
        console.print(f"  {topic}: {n} chunk(s)")
    console.print(f"[green]Knowledge base seeded[/green] ({sum(counts.values())} chunks).")


@app.command()
def status(engagement: str) -> None:
    """Show which artifacts exist and what the audit trail recorded."""
    store = StateStore()
    table = Table(title=f"State - {engagement}")
    table.add_column("Collection")
    table.add_column("Documents", justify="right")
    for collection in Collection:
        table.add_row(collection.value, str(len(store.list(engagement, collection))))
    console.print(table)

    trail = Table(title="Audit trail (most recent 15)")
    for col in ("seq", "when", "agent", "collection", "key", "action"):
        trail.add_column(col, overflow="fold")
    for entry in store.audit(engagement, limit=15):
        trail.add_row(str(entry.seq), f"{entry.at:%Y-%m-%d %H:%M:%S}", entry.agent,
                      entry.collection, entry.key, entry.action)
    console.print(trail)


@app.command()
def purge(
    engagement: str,
    confirm: bool = typer.Option(False, "--confirm", help="Required. Deletes deal data."),
    keep_audit: bool = typer.Option(True, help="Retain the audit trail for internal review."),
) -> None:
    """Confidentiality carry-through: tear down a deal's index and documents at close."""
    if not confirm:
        raise typer.BadParameter("pass --confirm; this deletes engagement data")
    from cdd_agent.retrieval.indexes import DataRoomIndex

    store = StateStore()
    deleted = store.purge_engagement(engagement, agent="operator", keep_audit=keep_audit)
    try:
        DataRoomIndex(engagement).purge()
        index_note = "data-room index dropped"
    except Exception as exc:  # index may not exist
        index_note = f"index not dropped: {exc}"
    console.print(f"{deleted} document(s) deleted; {index_note}.")


@app.command()
def config() -> None:
    """Show the resolved design parameters."""
    s = get_settings()
    table = Table(title="Design parameters")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_column("Source in the specs")
    for name, value, source in (
        ("model", s.model, "defaults to claude-opus-5"),
        ("beam_width", s.beam_width, "Checkpoint 4.1 s 2.3"),
        ("prune_threshold", s.prune_threshold, "Checkpoint 4.1 s 2.3"),
        ("tie_band", s.tie_band, "Checkpoint 4.1 s 2.3"),
        ("chunk band (tokens)", f"{s.chunk_min_tokens}-{s.chunk_max_tokens}", "Checkpoint 3.1 s 4"),
        ("chunk_overlap_ratio", s.chunk_overlap_ratio, "Checkpoint 3.1 s 4"),
        ("top_k", s.top_k, "Checkpoint 3.1 s 4"),
        ("similarity_floor", s.similarity_floor, "Checkpoint 3.1 s 5"),
        ("max_auditor_rounds", s.max_auditor_rounds, "Checkpoint 5.1, loop 2"),
        ("offline", s.offline, "local flag"),
    ):
        table.add_row(name, str(value), source)
    console.print(table)


if __name__ == "__main__":
    app()
