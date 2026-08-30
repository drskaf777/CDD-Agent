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
from cdd_agent.guardrails.escalation import check_phase1
from cdd_agent.knowledge.intake_questions import INTAKE_PROTOCOL
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

    for escalation in check_phase1(result):
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
    console.print(f"Taxonomy coverage: {register.coverage():.0%}")


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
