"""Critic / Evaluator - CrewAI.

Checkpoint 4.1 s 2.4 assigns this role to CrewAI because it needs its own persona
("skeptical reviewer, not the author") kept separate from the Generator's context, so
it cannot grade its own output. That separation is the whole reason this is a distinct
module and a distinct model object: the Critic is constructed without any reference to
the Generator's chain, prompt, or messages.

Scoring is deliberately split:

* The four-question check is **rule-based and hard**. A framing that leaves one of the
  four unmapped is pruned outright, and that prune is not recoverable by user override.
* The three soft criteria are scored 1-5 by the persona, with sub-sector fit checked
  against the Knowledge-Base Index rather than unaided judgment. A prune on these is
  marked recoverable, and the driving criterion is logged (Checkpoint 4.1 s 2.5).
"""

from __future__ import annotations

import os
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from cdd_agent.config import get_settings
from cdd_agent.knowledge.four_question_test import FOUR_QUESTIONS, classify
from cdd_agent.schemas.deal_profile import DealProfile
from cdd_agent.schemas.hypothesis import CriticScore, FourQuestionCheck, HypothesisTree
from cdd_agent.tools.retrieval_tools import MarketSearchTool

_BACKSTORY = """You are a skeptical reviewer of due diligence work plans. You did not
write the hypothesis trees you are given and you have no stake in any of them. Your job
is to find the framing that will hold up eight weeks from now, when the data room has
been read and the easy answers are gone.

You are hard to impress. A framing that reads well but cannot be evidenced from a real
data room scores low on testability no matter how elegant it is. A framing that ignores
what this specific buyer said it cares about scores low on buyer-criteria coverage no
matter how thorough it looks."""

_TASK = """Score this candidate hypothesis tree on three criteria, 1 to 5.

Framing: {label}
Root thesis: {thesis}
Buyer: {buyer_type}. Stated decision criteria: {criteria}
Sub-sector: {sub_sector}

Tier-1 hypotheses:
{hypotheses}

Sub-sector reference material retrieved from the knowledge base:
{kb_context}

Criteria:
1. buyer_criteria_coverage - does the tree test what this buyer said drives its go/no-go?
2. four_question_alignment - how well does it *answer* the four screening questions
   (whether each is merely addressed is checked separately and is not your call).
3. sub_sector_fit - does it test the metrics practitioners in this sub-sector treat as
   diagnostic? Judge against the retrieved reference material above, not from memory.
4. testability - could a real data room confirm or contradict each hypothesis?

For each criterion give a one-line reason. Be specific about what is missing."""


class CriticVerdict(BaseModel):
    buyer_criteria_coverage: float = Field(ge=1, le=5)
    four_question_alignment: float = Field(ge=1, le=5)
    sub_sector_fit: float = Field(ge=1, le=5)
    testability: float = Field(ge=1, le=5)
    buyer_criteria_reason: str = ""
    four_question_reason: str = ""
    sub_sector_reason: str = ""
    testability_reason: str = ""
    notes: str = ""

    @field_validator("buyer_criteria_coverage", "four_question_alignment",
                     "sub_sector_fit", "testability", mode="before")
    @classmethod
    def _score_from_text(cls, value: object) -> object:
        """Recover a score the model wrote as prose in the numeric field.

        Each score is paired with a reason field, and models periodically put the
        reason in both - returning "4 - the tree covers three of the four questions"
        where a float was asked for. Rejecting that outright loses a verdict that was
        substantively fine, and killed a whole Phase-1 search in practice.

        Only a leading in-range score is recovered. Text with no number, or whose
        first number is not a valid score, still fails: the prune threshold is 3.0
        and the beam is three branches wide, so inventing a score here would quietly
        decide which decomposition the engagement runs on.
        """
        if not isinstance(value, str):
            return value
        match = re.match(r"\s*([1-5](?:\.\d+)?)\b", value)
        if match:
            return float(match.group(1))
        return value


def _configure_crewai_tracing(enabled: bool) -> None:
    """State the tracing preference explicitly, whichever way it is set.

    CrewAI asks "Would you like to view your execution traces? [y/N]" with a
    20-second timeout only while the preference is *unset* - and it asks on every
    kickoff, which is three times per Phase-1 search since each branch gets its own
    Crew. That is an interactive block in a non-interactive pipeline.

    Answering it in the environment removes the prompt without giving up the traces:
    tracing stays on unless CDD_CREWAI_TRACING says otherwise. This is set before any
    crew is constructed, because CrewAI reads it when its telemetry module imports.
    """
    os.environ["CREWAI_TRACING_ENABLED"] = "true" if enabled else "false"


class Critic:
    """Scores one branch at a time. Never sees the other branches' scores."""

    def __init__(
        self, profile: DealProfile, market_search: Optional[MarketSearchTool] = None
    ) -> None:
        self.profile = profile
        self.settings = get_settings()
        self.market_search = market_search

    # ---------------------------------------------------------- hard check
    @staticmethod
    def four_question_check(tree: HypothesisTree) -> FourQuestionCheck:
        """Deterministic pass/fail: is each of the four questions mapped by some hypothesis?

        Rule-based on purpose. This is the one gate that must behave identically on
        every run, so it does not go through the model at all.
        """
        covered: set[str] = set()
        for h in tree.hypotheses:
            if h.depth != 1:
                continue
            covered |= classify(f"{h.statement} {h.rationale}")
            for evidence in h.required_evidence:
                covered |= classify(evidence)
        return FourQuestionCheck(**{q.key: (q.key in covered) for q in FOUR_QUESTIONS})

    # ------------------------------------------------------------- scoring
    def score(self, tree: HypothesisTree) -> CriticScore:
        check = self.four_question_check(tree)
        if self.settings.offline:
            verdict = self._offline_verdict(tree, check)
        else:
            verdict = self._crew_verdict(tree)

        return CriticScore(
            four_question=check,
            buyer_criteria_coverage=verdict.buyer_criteria_coverage,
            four_question_alignment=verdict.four_question_alignment,
            sub_sector_fit=verdict.sub_sector_fit,
            testability=verdict.testability,
            notes=verdict.notes,
            criterion_notes={
                "buyer_criteria_coverage": verdict.buyer_criteria_reason,
                "four_question_alignment": verdict.four_question_reason,
                "sub_sector_fit": verdict.sub_sector_reason,
                "testability": verdict.testability_reason,
            },
        )

    def _kb_context(self, tree: HypothesisTree) -> str:
        """Sub-sector fit is checked against the Knowledge-Base Index, per the spec."""
        if self.market_search is None:
            return "(knowledge base not available to the Critic on this run)"
        query = (
            f"diagnostic metrics and risk factors for {self.profile.sector.sub_sector} "
            f"commercial due diligence"
        )
        obs = self.market_search(query, sub_sector=self.profile.sector.sub_sector or None)
        return obs.render()

    def _crew_verdict(self, tree: HypothesisTree) -> CriticVerdict:
        _configure_crewai_tracing(self.settings.crewai_tracing)

        from crewai import Agent as CrewAgent
        from crewai import Crew, Task

        from cdd_agent.llm.models import get_crew_llm

        critic = CrewAgent(
            role="Skeptical diligence reviewer",
            goal=(
                "Judge whether a candidate hypothesis tree will survive contact with a "
                "real data room and answer what this buyer actually needs to decide."
            ),
            backstory=_BACKSTORY,
            llm=get_crew_llm(),
            allow_delegation=False,
            verbose=False,
        )
        task = Task(
            description=_TASK.format(
                label=tree.framing_label,
                thesis=tree.root_thesis,
                buyer_type=self.profile.buyer.buyer_type.value,
                criteria="; ".join(self.profile.buyer.decision_criteria) or "not stated",
                sub_sector=self.profile.sector.sub_sector or "not stated",
                hypotheses="\n".join(
                    f"- [{h.id}] {h.statement}" for h in tree.tier_1()
                ),
                kb_context=self._kb_context(tree),
            ),
            expected_output="A CriticVerdict with four scores and a reason for each.",
            agent=critic,
            output_pydantic=CriticVerdict,
        )
        try:
            result = Crew(
                agents=[critic], tasks=[task], verbose=False,
                tracing=self.settings.crewai_tracing,
            ).kickoff()
        except Exception as exc:
            # CrewAI validates the structured output inside kickoff and *raises* when
            # it will not parse, so the fallback below was unreachable on the path it
            # was written for: a single unparseable verdict ended the whole Phase-1
            # search. One branch failing to score is not a reason to lose the other
            # two, so it degrades to the deterministic rubric and says so in the notes
            # that reach the deck.
            return self._offline_verdict(tree, self.four_question_check(tree)).model_copy(
                update={"notes": f"critic output could not be scored ({type(exc).__name__}); "
                                 "deterministic rubric used for this branch"}
            )
        verdict = getattr(result, "pydantic", None)
        if isinstance(verdict, CriticVerdict):
            return verdict
        # CrewAI returns the raw string when structured parsing fails. Rather than
        # guess at scores, fall back to the deterministic rubric and say so.
        return self._offline_verdict(tree, self.four_question_check(tree)).model_copy(
            update={"notes": "structured critic output unavailable; deterministic rubric used"}
        )
    def _offline_verdict(
        self, tree: HypothesisTree, check: FourQuestionCheck
    ) -> CriticVerdict:
        """A transparent rubric, not a simulated judgment.

        Scores what can be measured without a model: how much of the buyer's stated
        criteria vocabulary the tree touches, how many of the four questions it maps,
        whether hypotheses name sub-sector diagnostics, and whether each carries a
        specific required-evidence item.
        """
        tier1 = tree.tier_1()
        text = " ".join(f"{h.statement} {h.rationale}" for h in tier1).lower()

        criteria = [c.lower() for c in self.profile.buyer.decision_criteria]
        hits = sum(1 for c in criteria if any(w in text for w in c.split() if len(w) > 4))
        coverage = 1.0 + 4.0 * (hits / len(criteria)) if criteria else 3.0

        mapped = sum(1 for v in check.model_dump().values() if v)
        alignment = 1.0 + 4.0 * (mapped / 4)

        diagnostics = ("nrr", "grr", "cac", "ltv", "payback", "churn", "concentration",
                       "payer", "referral", "utilization", "rule of 40", "arr")
        fit = 1.0 + 4.0 * min(1.0, sum(1 for d in diagnostics if d in text) / 3)

        with_evidence = sum(1 for h in tier1 if h.required_evidence)
        testability = 1.0 + 4.0 * (with_evidence / len(tier1)) if tier1 else 1.0

        return CriticVerdict(
            buyer_criteria_coverage=round(min(5.0, coverage), 2),
            four_question_alignment=round(min(5.0, alignment), 2),
            sub_sector_fit=round(min(5.0, fit), 2),
            testability=round(min(5.0, testability), 2),
            buyer_criteria_reason=f"{hits}/{len(criteria) or 0} stated criteria echoed",
            four_question_reason=f"{mapped}/4 screening questions mapped",
            sub_sector_reason="scored on presence of sub-sector diagnostic metrics",
            testability_reason=f"{with_evidence}/{len(tier1)} hypotheses name specific evidence",
            notes="offline deterministic rubric - not a substitute for the Critic persona",
        )
