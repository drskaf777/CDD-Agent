"""Thesis Architect - Phase 1, the one place Tree of Thought applies.

Checkpoint 4.1 scopes ToT deliberately: the Phase 3-5 evidence loop stays ReAct, because
there is rarely real ambiguity about which gap to close next (a priority-queue problem).
Phase 1 is different - a thesis admits several defensible decompositions, whichever is
generated first anchors weeks of data requests, and a good decomposition has to satisfy
the buyer's criteria, the four-question test, and sub-sector patterns at once.

The agent composes three collaborators across two frameworks:

    Generator (LangChain LCEL)  ->  Critic (CrewAI)  ->  Controller (LangChain routing)

with the branch state living in the shared store, reached through the state-access
protocol that MCP implements.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from cdd_agent.agents.base import Agent, AgentContext
from cdd_agent.agents.thesis_architect.beam_search import (
    apply_override,
    build_result,
    prune_and_route,
)
from cdd_agent.agents.thesis_architect.critic import Critic
from cdd_agent.agents.thesis_architect.generator import ThoughtGenerator
from cdd_agent.guardrails.authorization import AgentRole
from cdd_agent.schemas.hypothesis import HypothesisTree, ThesisSearchResult
from cdd_agent.state.access import StateAccess
from cdd_agent.state.store import Collection


class ThesisArchitect(Agent):
    """Runs the width-3 beam search and hands over the selected Hypothesis Tree."""

    role = AgentRole.THESIS_ARCHITECT

    def __init__(self, context: AgentContext, *, state_access: Optional[StateAccess] = None) -> None:
        super().__init__(context)
        self.state_access = state_access

    def run(self, *, save: bool = True) -> ThesisSearchResult:
        profile = self.context.profile
        if profile is None:
            raise ValueError(
                "Phase 1 requires a Deal Profile Brief, and this engagement has none. "
                "Run `cdd intake <engagement> --briefing <file>` to produce one from a "
                "deal briefing, or `cdd demo` to load the worked example. If you expected "
                "one to exist, check CDD_DATA_DIR - each data directory holds its own "
                "engagements."
            )
        ready, missing = profile.is_ready_for_phase_1()
        if not ready:
            raise ValueError(
                "Phase 1 blocked - intake is incomplete: " + "; ".join(missing)
            )

        # Rendering belongs to the Correction, which knows whether it was redacted.
        # Formatting the fields here is how another client figures reach this prompt
        # without anyone deciding that they should.
        corrections = [
            c.render_for_prompt()
            for c in self.context.memory.corrections_for_sub_sector(profile.sector.sub_sector)
        ]

        # 1. Generate the beam.
        branches = ThoughtGenerator(profile).generate(corrections)
        self._persist_branches(branches, stage="generated")

        # 2. Score each branch independently. The Critic never sees another branch
        #    score, so ordering cannot influence judgment - which is also precisely
        #    what makes it safe to score them at the same time. Scoring was the larger
        #    half of Phase 1 latency and it was entirely serial.
        critic = Critic(profile, market_search=self.tools().market_search)
        if self.offline or len(branches) < 2:
            for branch in branches:
                branch.score = critic.score(branch)
        else:
            with ThreadPoolExecutor(max_workers=len(branches),
                                    thread_name_prefix="tot-score") as pool:
                for branch, score in zip(branches,
                                         pool.map(critic.score, branches),
                                         strict=True):
                    branch.score = score
        self._persist_branches(branches, stage="scored")

        # 3. Route: select, escalate a tie, or ask a clarifying question.
        routing = prune_and_route(branches)
        result = build_result(self.context.engagement_id, branches, routing)

        if save:
            self.context.memory.save_thesis_search(result, agent=self.name)
            selected = result.selected()
            if selected is not None:
                self.context.memory.save_hypothesis_tree(selected, agent=self.name)
        return result

    def approve(self, result: ThesisSearchResult, *, approved_by: str) -> HypothesisTree:
        """The human-approval gate before Phase 2 data requests are generated.

        ToT changed how the candidate tree is produced, not the checkpoint that
        approves it (Checkpoint 4.1 s 2.2).
        """
        tree = result.selected()
        if tree is None:
            raise ValueError(
                "no branch selected - resolve the escalation before approving"
            )
        tree.human_approved = True
        # Override then approve is one decision by one person, not two. Appending both
        # produces a provenance string that reads as a committee.
        stamp = f"(approved by {approved_by})"
        if stamp not in tree.created_by:
            tree.created_by = (
                tree.created_by.replace(f"(override by {approved_by})", "").strip()
                + f" {stamp}"
            ).replace("  ", " ")
        self.context.memory.save_hypothesis_tree(tree, agent=self.name)
        self.context.memory.save_thesis_search(result, agent=self.name)
        return tree

    def override(
        self, result: ThesisSearchResult, branch_id: str, *, approved_by: str
    ) -> ThesisSearchResult:
        """Recover a soft-pruned framing, or resolve a tie, by human choice."""
        updated = apply_override(result, branch_id, approved_by=approved_by)
        self.context.memory.save_thesis_search(updated, agent=self.name)
        selected = updated.selected()
        if selected is not None:
            self.context.memory.save_hypothesis_tree(selected, agent=self.name)
        return updated

    def _persist_branches(self, branches: list[HypothesisTree], *, stage: str) -> None:
        """Write branch state through the state-access layer during the search.

        Uses the protocol rather than the store directly, so the same code path works
        when the Critic runs out-of-process behind MCP.
        """
        access = self.state_access
        for branch in branches:
            payload = branch.model_dump(mode="json") | {"stage": stage}
            if access is not None:
                access.write(
                    self.context.engagement_id,
                    Collection.THESIS_SEARCH.value,
                    f"branch::{branch.branch_id}",
                    payload,
                    agent=self.name,
                )
            else:
                self.context.store.put(
                    self.context.engagement_id,
                    Collection.THESIS_SEARCH,
                    f"branch::{branch.branch_id}",
                    payload,
                    agent=self.name,
                )


__all__ = [
    "Critic",
    "ThesisArchitect",
    "ThoughtGenerator",
    "apply_override",
    "build_result",
    "prune_and_route",
]
