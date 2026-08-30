"""Controller - routing logic, not a persona.

Checkpoint 5.1: a lightweight non-persona Controller routes hand-offs, the same pattern
Checkpoint 4.1 used for the Decision Maker, because it decides rather than judges.

Coordination is hybrid. The backbone is sequential -

    Intake -> Thesis Architect -> Analyst -> Risk Auditor -> Synthesizer

- an assembly line, with two small project-network dependencies layered on top:

    Loop 1  generate -> score -> prune, inside the Thesis Architect
    Loop 2  Analyst <-> Risk Auditor, where a flagged gap routes back before Synthesis

The Controller gates each hand-off on the approval checkpoints that already existed. It
does not invent new ones, and it never auto-advances past an escalation - that is the
whole point of the escalation triggers being hard-coded rather than advisory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cdd_agent.agents.analyst import Analyst, LoopReport
from cdd_agent.agents.base import AgentContext
from cdd_agent.agents.intake import IntakeAgent
from cdd_agent.agents.risk_auditor import AuditReport, RiskAuditor
from cdd_agent.agents.synthesizer import Synthesizer
from cdd_agent.agents.thesis_architect import ThesisArchitect
from cdd_agent.config import get_settings
from cdd_agent.guardrails.escalation import (
    Escalation,
    check_phase1,
    check_source_conflicts,
    check_tier1_evidence,
    final_recommendation_review,
)
from cdd_agent.retrieval.ingestion import IngestionReport, ingest_directory
from cdd_agent.schemas.data_request import DataRequestChecklist
from cdd_agent.schemas.deal_profile import DealProfile
from cdd_agent.schemas.deck import Deck
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree, ThesisSearchResult
from cdd_agent.schemas.risk import RiskRegister
from cdd_agent.state.store import Collection, StateStore


class PipelineHalted(RuntimeError):
    """The pipeline stopped at a gate. Carries the escalations that stopped it."""

    def __init__(self, escalations: list[Escalation]) -> None:
        self.escalations = escalations
        super().__init__(
            "pipeline halted for human review:\n"
            + "\n".join(f"  {e.render()}" for e in escalations)
        )


@dataclass
class PhaseTiming:
    phase: str
    seconds: float


@dataclass
class RunReport:
    """Everything one run produced, including what it refused to do."""

    engagement_id: str
    profile: Optional[DealProfile] = None
    thesis_search: Optional[ThesisSearchResult] = None
    tree: Optional[HypothesisTree] = None
    checklist: Optional[DataRequestChecklist] = None
    ingestion: Optional[IngestionReport] = None
    matrix: Optional[EvidenceMatrix] = None
    loop: Optional[LoopReport] = None
    audit: Optional[AuditReport] = None
    register: Optional[RiskRegister] = None
    deck: Optional[Deck] = None
    escalations: list[Escalation] = field(default_factory=list)
    timings: list[PhaseTiming] = field(default_factory=list)
    auditor_rounds: int = 0
    halted_at: Optional[str] = None
    blocked_reason: Optional[str] = None

    @property
    def completed(self) -> bool:
        return self.deck is not None

    def blocking(self) -> list[Escalation]:
        return [e for e in self.escalations if e.blocking]

    def raise_if_halted(self) -> None:
        """For callers that want a halt to be an exception rather than a return value."""
        if self.halted_at and self.blocking():
            raise PipelineHalted(self.blocking())

    def summary(self) -> str:
        lines = [f"engagement: {self.engagement_id}"]
        for t in self.timings:
            lines.append(f"  {t.phase:<24} {t.seconds:6.2f}s")
        if self.halted_at:
            lines.append(f"  halted at: {self.halted_at}")
        if self.blocked_reason:
            lines.append(f"  {self.blocked_reason}")
        for e in self.escalations:
            lines.append(f"  {e.render()}")
        return "\n".join(lines)


class Controller:
    """Runs the backbone and gates each hand-off."""

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self.settings = get_settings()

    # ------------------------------------------------------------ whole pipeline
    def run(
        self,
        briefing: str,
        data_room: Optional[Path | str] = None,
        *,
        auto_approve_phase1: bool = False,
        approver: str = "unattended run",
    ) -> RunReport:
        """Run Phase 0 through Phase 4, stopping at any gate that requires a human.

        `auto_approve_phase1` exists for the demo and for CI. It clears the
        human-approval gate on an *uncontested* selection only - it can never clear a
        tie or an all-pruned outcome, because those are escalations, not approvals.
        """
        report = RunReport(engagement_id=self.context.engagement_id)

        report.profile = self._timed(report, "Phase 0 intake", lambda: self._intake(briefing))
        self.context.profile = report.profile

        ready, missing = report.profile.is_ready_for_phase_1()
        if not ready:
            # Not an escalation trigger - an ordinary phase gate. Phase 1 decomposes
            # the thesis, so it cannot start without one.
            report.halted_at = "Phase 0"
            report.blocked_reason = (
                "Intake is incomplete; Phase 1 needs these answers first: "
                + "; ".join(missing)
            )
            return report

        search = self._timed(report, "Phase 1 thesis (ToT)", self._thesis)
        report.thesis_search = search
        report.escalations.extend(check_phase1(search))
        if search.requires_human():
            report.halted_at = "Phase 1"
            self._record_escalations(report)
            return report

        tree = search.selected()
        assert tree is not None
        if auto_approve_phase1:
            tree = ThesisArchitect(self.context).approve(search, approved_by=approver)
        report.tree = tree

        report.checklist = self._timed(
            report, "Phase 2 data request", lambda: Analyst(self.context).generate_data_request(tree)
        )

        if data_room:
            report.ingestion = self._timed(
                report, "Phase 3 ingestion", lambda: self._ingest(Path(data_room))
            )

        matrix, loop, audit, register, rounds = self._timed(
            report, "Phase 3 evidence loop", lambda: self._evidence_and_audit(tree)
        )
        report.matrix, report.loop, report.audit = matrix, loop, audit
        report.register, report.auditor_rounds = register, rounds

        report.escalations.extend(check_source_conflicts(register))
        report.escalations.extend(check_tier1_evidence(tree, matrix, register))
        if any(e.blocking for e in report.escalations):
            report.halted_at = "Phase 4 gate"
            self._record_escalations(report)
            return report

        report.deck = self._timed(
            report,
            "Phase 4 synthesis",
            lambda: Synthesizer(self.context).run(tree, matrix, register)[0],
        )
        # Trigger 5 is unconditional: the draft is complete, and the recommendation
        # still requires human review before the IC.
        report.escalations.append(final_recommendation_review(self.context.engagement_id))
        self._record_escalations(report)
        return report

    # ---------------------------------------------------------------- the phases
    def _intake(self, briefing: str) -> DealProfile:
        """Run Phase 0, or resume from the Deal Profile already in the store.

        An empty briefing with a saved profile means "continue this engagement", not
        "re-scope it from nothing" - re-running intake would overwrite answers the deal
        team already gave.
        """
        if not briefing.strip():
            existing = self.context.profile or self.context.memory.deal_profile()
            if existing is not None:
                return existing
        return IntakeAgent(self.context).run(briefing)

    def _thesis(self) -> ThesisSearchResult:
        return ThesisArchitect(self.context).run()

    def _ingest(self, data_room: Path) -> IngestionReport:
        ingestion, tables = ingest_directory(self.context.engagement_id, data_room)
        # Parsed tables become the computation tool's inputs for the rest of the run.
        if self.context.registry is not None:
            self.context.registry.tables = list(tables)
        self.context.store.put(
            self.context.engagement_id,
            Collection.METRICS,
            "ingestion",
            {
                "summary": ingestion.summary(),
                "unstructured": ingestion.unstructured,
                "structured": ingestion.structured,
                "skipped": ingestion.skipped,
                "undated": ingestion.undated,
            },
            agent="Controller",
        )
        return ingestion

    def _evidence_and_audit(
        self, tree: HypothesisTree
    ) -> tuple[EvidenceMatrix, LoopReport, AuditReport, RiskRegister, int]:
        """Loop 2: Analyst <-> Risk Auditor.

        A flagged gap routes back to the Analyst before Synthesis can start. The loop
        is bounded: an auditor that can demand rework indefinitely would trade away the
        reliability it was added to buy.
        """
        analyst = Analyst(self.context)
        auditor = RiskAuditor(self.context)

        matrix, loop = analyst.run_evidence_loop(tree)
        register, audit = auditor.audit(tree, matrix)
        rounds = 1

        while audit.requires_analyst_rework and rounds < self.settings.max_auditor_rounds:
            rounds += 1
            matrix, loop = analyst.run_evidence_loop(tree, matrix=matrix)
            register, audit = auditor.audit(tree, matrix, register=register)

        return matrix, loop, audit, register, rounds

    # ------------------------------------------------------------------ plumbing
    def _timed(self, report: RunReport, phase: str, fn):  # type: ignore[no-untyped-def]
        """Record wall-clock per phase.

        Latency is an evaluation metric in its own right, particularly across the
        Analyst-Risk Auditor loop, since that loop is a deliberate trade.
        """
        start = time.perf_counter()
        try:
            return fn()
        finally:
            report.timings.append(PhaseTiming(phase, time.perf_counter() - start))

    def _record_escalations(self, report: RunReport) -> None:
        for escalation in report.escalations:
            self.context.store.append(
                self.context.engagement_id,
                Collection.ESCALATION,
                escalation.to_dict(),
                agent="Controller",
            )


def new_context(engagement_id: str, store: Optional[StateStore] = None) -> AgentContext:
    return AgentContext.create(engagement_id, store=store)
