"""Analyst - Phase 2 data requests plus the Phase 3 ReAct evidence loop.

Checkpoint 5.1 folds data-request generation into this agent rather than giving it its
own: deciding what evidence closes a gap is the same judgment the evidence loop makes,
not a separate skill.

The loop is ReAct, not ToT (Checkpoint 4.1 s 1): at any point the next action is
determined by which Tier-1 hypothesis has the weakest evidence - a priority-queue
problem with a well-defined next step. Scoring multiple candidate actions at every
Thought step would multiply the cost of every retrieval call without a matching
accuracy gain.

Stopping condition (Checkpoint 2.1): the loop exits for a deal only when every Tier-1
hypothesis has reached at least a partial rating or carries an explicitly dated
information gap. That is what stops the agent writing slides on an incomplete evidence
base.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

from cdd_agent.agents.base import Agent, AgentContext
from cdd_agent.guardrails.authorization import AgentRole, AuthorizationError
from cdd_agent.guardrails.coherence import raise_if_incoherent
from cdd_agent.knowledge.data_request_catalog import (
    ADDONS_BY_MODULE,
    UNIVERSAL_CATALOG,
    CatalogItem,
    catalog_for,
    public_record_note,
)
from cdd_agent.knowledge.outline import module_for_sub_sector
from cdd_agent.schemas.common import ConfidenceTag, Tier
from cdd_agent.schemas.data_request import DataRequestChecklist, DataRequestItem
from cdd_agent.schemas.deal_profile import PublicMarketContext
from cdd_agent.schemas.evidence import EvidenceItem, EvidenceMatrix
from cdd_agent.schemas.hypothesis import Hypothesis, HypothesisTree
from cdd_agent.schemas.risk import GapOwner, InformationGap
from cdd_agent.state.store import Collection
from cdd_agent.tools.retrieval_tools import RetrievalObservation

# Rating order for the priority queue: least-supported first.
_WEAKNESS: dict[ConfidenceTag, int] = {
    ConfidenceTag.NO_DATA: 0,
    ConfidenceTag.PARTIALLY_CONFIRMED: 1,
    ConfidenceTag.CONTRADICTED: 2,   # already decision-relevant; no more evidence needed
    ConfidenceTag.CONFIRMED: 3,
}


@dataclass
class LoopStep:
    """One Thought -> Action -> Observation cycle, kept for the audit trail."""

    step: int
    hypothesis_id: str
    thought: str
    action: str
    observation: str
    tag: ConfidenceTag
    # Versions the retrieval filter dropped before ranking. Recorded because a
    # superseded-but-cited figure is the failure mode the filter exists to prevent,
    # and a guard that works invisibly cannot be reviewed.
    superseded_filtered: list = field(default_factory=list)


@dataclass
class LoopReport:
    steps: list[LoopStep] = field(default_factory=list)
    gaps_logged: list[InformationGap] = field(default_factory=list)
    stopped_because: str = ""
    blocked_actions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.steps)} ReAct step(s), {len(self.gaps_logged)} gap(s) logged. "
            f"{self.stopped_because}"
        )


class Analyst(Agent):
    role = AgentRole.ANALYST

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._evidence_counter = 0
        self._gap_counter = 0

    # ------------------------------------------------------- Phase 2: requests
    def generate_data_request(
        self, tree: HypothesisTree, *, save: bool = True
    ) -> DataRequestChecklist:
        """Walk the outline-to-evidence mapping and emit one prioritized checklist.

        Two sources feed it: the universal catalogue (always), and the sub-sector
        add-ons where a pre-built module matches. Items are traced back to the
        hypotheses they are load-bearing for, so a Tier-1 item can be justified rather
        than just asserted.
        """
        # Everything below is derived from the tree, which was derived from the
        # brief. If those two no longer agree the whole checklist would be a request
        # for the wrong company data, so this is a stop rather than a warning.
        raise_if_incoherent(
            self.context.engagement_id, profile=self.context.profile, tree=tree
        )
        profile = self.context.profile
        sub_sector = profile.sector.sub_sector if profile else ""
        business_model = profile.sector.business_model.value if profile else ""
        module = module_for_sub_sector(sub_sector, business_model)

        catalog: list[CatalogItem] = list(UNIVERSAL_CATALOG)
        if module:
            catalog += list(ADDONS_BY_MODULE.get(module, ()))
        # Listed targets add the public record and the structure-specific asks.
        catalog += list(catalog_for(self.context.deal_shape))

        public = self.context.deal_shape.public_target
        items: list[DataRequestItem] = []
        for n, entry in enumerate(catalog, start=1):
            linked = _link_to_hypotheses(entry, tree)
            # Tier is a property of how load-bearing an item is *for this tree*: an
            # item nothing in the tree depends on is demoted rather than dropped, so
            # the checklist stays complete without inflating the blocking set.
            tier = entry.tier
            if tier is Tier.DEAL_CRITICAL and not linked:
                tier = Tier.DEPTH_BUILDING
            rationale = entry.rationale or (
                f"Supports {', '.join(linked)}" if linked else "Standard coverage"
            )
            # On a listed target, anything the filings already answer is retrieved
            # rather than requested. Sending management a list of things they
            # published last quarter burns the scarcest resource in a live process
            # and says plainly that nobody read them.
            where = public_record_note(entry.item) if public else ""
            if where:
                tier = Tier.ENRICHMENT
                rationale = f"Available from the public record: {where}. Retrieved, not requested."
            items.append(
                DataRequestItem(
                    id=f"DR-{n:03d}",
                    category=entry.category,
                    item=entry.item,
                    tier=tier,
                    hypothesis_ids=linked,
                    rationale=rationale,
                    sub_sector_specific=entry.sub_sector_specific,
                )
            )

        # Anything a hypothesis explicitly asked for that the catalogue does not cover.
        for h in tree.tier_1():
            for requirement in h.required_evidence:
                if _already_covered(requirement, items):
                    continue
                items.append(
                    DataRequestItem(
                        id=f"DR-{len(items) + 1:03d}",
                        category="Hypothesis-specific",
                        item=requirement,
                        tier=Tier.DEAL_CRITICAL,
                        hypothesis_ids=[h.id],
                        rationale=f"Named as required evidence by {h.id}",
                        sub_sector_specific=bool(module),
                    )
                )

        checklist = DataRequestChecklist(
            engagement_id=self.context.engagement_id,
            created_by=self.name,
            items=items,
        )
        if save:
            self.context.memory.save_data_request(checklist, agent=self.name)
        return checklist

    # -------------------------------------------------- Phase 3: evidence loop
    def run_evidence_loop(
        self,
        tree: HypothesisTree,
        *,
        matrix: Optional[EvidenceMatrix] = None,
        max_steps: Optional[int] = None,
        save: bool = True,
    ) -> tuple[EvidenceMatrix, LoopReport]:
        matrix = matrix or self.context.memory.evidence_matrix()
        report = LoopReport()
        budget = max_steps or self.context.settings.max_react_steps
        tools = self.tools()
        # A hypothesis whose planned queries are all spent leaves the queue. Without
        # this it stays the weakest forever and the loop burns its whole budget
        # re-logging the same gap.
        exhausted: set[str] = set()

        for step in range(1, budget + 1):
            target = self._weakest_tier1(tree, matrix, exhausted)
            if target is None:
                report.stopped_because = (
                    "every Tier-1 hypothesis reached at least a partial rating"
                    if not exhausted
                    else (
                        "every Tier-1 hypothesis is either rated or has exhausted its "
                        f"planned queries ({len(exhausted)} exhausted, "
                        f"{len(report.gaps_logged)} of those with no evidence at all)"
                    )
                )
                break

            # --- Thought ---
            rating = matrix.rating(target.id)
            thought = (
                f"{target.id} is the least-supported Tier-1 hypothesis ({rating.value}). "
                f"Needed: {'; '.join(target.required_evidence) or 'direct evidence'}."
            )
            queries = _queries_for(target)
            asked = {q for s in report.steps if s.hypothesis_id == target.id for q in [s.action]}
            query = next((q for q in queries if q not in asked), None)
            if query is None:
                # Every planned query for this hypothesis has been tried. It leaves the
                # queue either way; it only earns a gap if nothing was found, since a
                # gap against evidence the deal team already supplied is noise.
                exhausted.add(target.id)
                if rating is ConfidenceTag.NO_DATA:
                    gap = self._log_gap(target, matrix)
                    report.gaps_logged.append(gap)
                    report.steps.append(
                        LoopStep(step, target.id, thought, "log_information_gap",
                                 gap.request, ConfidenceTag.NO_DATA)
                    )
                continue

            # --- Action ---
            observation: Optional[RetrievalObservation] = None
            if tools.document_retrieval is not None:
                observation = tools.document_retrieval(query)
            elif tools.market_search is not None:
                observation = tools.market_search(query)
            else:
                report.blocked_actions.append(
                    "no retrieval tool authorized on this engagement"
                )
                report.stopped_because = (
                    "no retrieval tool is authorized - resolve intake Category F access "
                    "before Phase 3"
                )
                break

            # --- Observation ---
            item = self._observe(target, query, observation)
            if item is not None:
                matrix.add(item)
                for citation in item.citations:
                    self.context.memory.log_citation(citation, target.id, agent=self.name)
            report.steps.append(
                LoopStep(
                    step=step,
                    hypothesis_id=target.id,
                    thought=thought,
                    action=query,
                    observation=observation.render()[:600],
                    tag=item.tag if item else ConfidenceTag.NO_DATA,
                    superseded_filtered=list(observation.superseded_filtered),
                )
            )
        else:
            report.stopped_because = (
                f"step budget of {budget} exhausted - remaining gaps are logged"
            )

        # Any Tier-1 hypothesis still at No Data must leave with a dated gap.
        for h in tree.tier_1():
            if matrix.rating(h.id) is ConfidenceTag.NO_DATA and not any(
                g.hypothesis_id == h.id for g in report.gaps_logged
            ):
                report.gaps_logged.append(self._log_gap(h, matrix))

        if save:
            self.context.memory.save_evidence_matrix(matrix, agent=self.name)
            self._save_gaps(report.gaps_logged)
            self._save_trace(report)
            # Design spec s VI step 4: quantitative analysis - cohort builds, revenue
            # bridges, concentration - is the Analyst's work, done before slide
            # generation at step 6. Computing it here keeps the Synthesizer free of a
            # computation tool it has no business holding.
            self.compute_exhibits(tree, matrix)
        return matrix, report

    def compute_exhibits(self, tree: HypothesisTree, matrix: EvidenceMatrix) -> list:
        """Build and store the quantitative exhibits from the parsed data-room tables."""
        from cdd_agent.synthesis.exhibits import ExhibitContext, build_computed

        exhibits = build_computed(ExhibitContext(
            tree=tree, matrix=matrix,
            register=self.context.memory.risk_register(),
            computation=self.tools().computation,
            shape=self.context.deal_shape,
            public=(self.context.profile.public_market
                    if self.context.profile else PublicMarketContext()),
            access=self.context.profile.access if self.context.profile else None,
        ))
        self.context.store.put(
            self.context.engagement_id, Collection.EXHIBIT, "computed",
            {"exhibits": [e.model_dump(mode="json") for e in exhibits]},
            agent=self.name,
        )
        return exhibits

    # --------------------------------------------------------------- internals
    def _weakest_tier1(
        self,
        tree: HypothesisTree,
        matrix: EvidenceMatrix,
        exhausted: Optional[set[str]] = None,
    ) -> Optional[Hypothesis]:
        """The priority queue that makes this a ReAct loop rather than a search."""
        skip = exhausted or set()
        candidates = [
            h for h in tree.tier_1()
            if h.id not in skip
            and _WEAKNESS[matrix.rating(h.id)] < _WEAKNESS[ConfidenceTag.CONTRADICTED]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda h: (_WEAKNESS[matrix.rating(h.id)], h.id))

    def _observe(
        self, hypothesis: Hypothesis, query: str, observation: RetrievalObservation
    ) -> Optional[EvidenceItem]:
        """Tag the retrieved result back to the hypothesis.

        The tag, not the raw output, is what carries forward. A below-floor retrieval
        produces no evidence item at all - the gap is logged instead, so a near-miss
        becomes an explicit hole rather than a weak citation.
        """
        if not observation.citations:
            return None
        self._evidence_counter += 1
        source_kind = observation.citations[0].source_kind
        claim = (
            f"Retrieved evidence bearing on {hypothesis.id}: "
            f"{' '.join(observation.passages[0].split())[:240]}"
        )
        return EvidenceItem(
            id=f"EV-{self._evidence_counter:04d}",
            engagement_id=self.context.engagement_id,
            created_by=self.name,
            hypothesis_id=hypothesis.id,
            claim=claim,
            tag=observation.provisional_tag,
            citations=observation.citations,
            source_kind=source_kind,
            query=query,
        )

    def _log_gap(self, hypothesis: Hypothesis, matrix: EvidenceMatrix) -> InformationGap:
        """A specific, addressed follow-up - never a generic "more data needed"."""
        self._gap_counter += 1
        profile = self.context.profile
        wanted = hypothesis.required_evidence[0] if hypothesis.required_evidence else (
            f"evidence that would confirm or contradict: {hypothesis.statement}"
        )
        owner = (
            GapOwner.THIRD_PARTY_EXPERT
            if any(w in wanted.lower() for w in ("market", "third-party", "independent"))
            else GapOwner.MANAGEMENT
        )
        target_date = _target_date(profile.target.ic_date if profile else None)
        carried = bool(
            profile
            and not profile.access.customer_contact_permitted
            and "customer" in wanted.lower()
        )
        return InformationGap(
            id=f"GAP-{self._gap_counter:03d}",
            engagement_id=self.context.engagement_id,
            created_by=self.name,
            hypothesis_id=hypothesis.id,
            request=wanted,
            owner=owner,
            target_close_date=target_date,
            blocking=hypothesis.is_tier_1
            and matrix.rating(hypothesis.id) is ConfidenceTag.NO_DATA,
            carried_to_confirmatory=carried,
        )

    def _save_trace(self, report: LoopReport) -> None:
        """Persist the reasoning, not just its result.

        The audit log records that the Evidence Matrix changed and who changed it.
        This records *why* the Analyst went where it went - which hypothesis was
        weakest, what it asked, and what came back. Without it the run is auditable
        only at the artifact level, which is not enough to review a judgment call.
        """
        for step in report.steps:
            self.context.store.append(
                self.context.engagement_id,
                Collection.TRACE,
                {
                    "step": step.step,
                    "at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "phase": "Phase 3 - evidence loop",
                    "hypothesis_id": step.hypothesis_id,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "tag": step.tag.value,
                    "superseded_filtered": step.superseded_filtered,
                },
                agent=self.name,
            )

    def _save_gaps(self, gaps: list[InformationGap]) -> None:
        if not gaps:
            return
        register = self.context.memory.risk_register()
        known = {g.id for g in register.gaps}
        register.gaps.extend(g for g in gaps if g.id not in known)
        self.context.memory.save_risk_register(register, agent=self.name)

    # ------------------------------------------------- independent research
    def commission_research(
        self, hypothesis: Hypothesis, *, contact_type: str = "customer", sample_size: int = 8
    ) -> Optional[str]:
        """Commission independent interviews where intake authorizes it.

        A blocked call is recorded as a confirmatory-diligence item rather than
        swallowed, so the deck cannot look outside-in when it is not.
        """
        tools = self.tools()
        if tools.primary_research is None:
            return None
        try:
            request = tools.primary_research.commission(
                hypothesis_id=hypothesis.id,
                contact_type=contact_type,
                objective=f"Test: {hypothesis.statement}",
                target_sample_size=sample_size,
            )
        except AuthorizationError as exc:
            self._gap_counter += 1
            register = self.context.memory.risk_register()
            register.gaps.append(
                InformationGap(
                    id=f"GAP-{self._gap_counter:03d}",
                    engagement_id=self.context.engagement_id,
                    created_by=self.name,
                    hypothesis_id=hypothesis.id,
                    request=(
                        f"Independent {contact_type} interviews on {hypothesis.id} - "
                        f"blocked pre-signing: {exc}"
                    ),
                    owner=GapOwner.DEAL_TEAM,
                    carried_to_confirmatory=True,
                    blocking=False,
                )
            )
            self.context.memory.save_risk_register(register, agent=self.name)
            return None
        return request.methodology_disclosure()


# --------------------------------------------------------------------- helpers
def _link_to_hypotheses(entry: CatalogItem, tree: HypothesisTree) -> list[str]:
    """Match a catalogue item to the hypotheses whose required evidence it satisfies."""
    terms = _keywords(entry.item)
    linked: list[str] = []
    for h in tree.tier_1():
        haystack = _keywords(" ".join(h.required_evidence) + " " + h.statement)
        if len(terms & haystack) >= 2:
            linked.append(h.id)
    return linked


def _already_covered(requirement: str, items: list[DataRequestItem]) -> bool:
    terms = _keywords(requirement)
    if not terms:
        return True
    for item in items:
        overlap = terms & _keywords(item.item)
        if len(overlap) >= max(2, len(terms) // 2):
            return True
    return False


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "per",
    "data", "detail", "level", "report", "schedule", "history", "summary", "case",
}


def _keywords(text: str) -> set[str]:
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def _queries_for(hypothesis: Hypothesis) -> list[str]:
    """Several narrow, targeted queries per Thought step - not one broad query.

    Checkpoint 3.1 s 4: a targeted query against a smaller candidate set is what lets
    the confidence tag in the Observation step mean something.
    """
    queries = [e for e in hypothesis.required_evidence if e.strip()]
    queries.append(hypothesis.statement)
    return queries


def _target_date(ic_date: Optional[_dt.date]) -> _dt.date:
    """Target-resolution date tied to the IC deadline captured at intake."""
    today = _dt.date.today()
    if ic_date and ic_date > today:
        # Leave a week of slack before the committee meets.
        return max(today + _dt.timedelta(days=3), ic_date - _dt.timedelta(days=7))
    return today + _dt.timedelta(days=14)
