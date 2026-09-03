"""Intake Agent - Phase 0 scoping, produces the Deal Profile Brief.

Checkpoint 5.1 gives this agent no data-room access at all: it runs a conversation, and
widening its scope would put the file-reading and the scoping judgment in the same
place. The authorization table enforces that; this module simply never asks.

The agent's job is not to fill every field. It is to reach the three Phase-1
prerequisites (Categories B, C, D) and to record honestly what remains open - because
an unanswered Category F question is the difference between an authorized interview
programme and a hard block later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from cdd_agent.agents.base import Agent
from cdd_agent.guardrails.authorization import AgentRole
from cdd_agent.knowledge.intake_questions import INTAKE_PROTOCOL
from cdd_agent.schemas.deal_profile import (
    AccessConstraints,
    BuyerProfile,
    DealProfile,
    DeliverableParameters,
    InvestmentThesis,
    ProcessContext,
    PublicMarketContext,
    SectorDefinition,
    TargetIdentification,
)

_SYSTEM = """You are the Intake Agent on a commercial due diligence engagement.

Run the diagnostic intake protocol: extract everything the deal team has told you into
the Deal Profile Brief, and list what is still missing. Two rules:

1. Never infer a fact the user did not state. An unanswered question belongs in
   open_intake_questions, not filled with a plausible default. The downstream hypothesis
   tree is built from these answers, so a guessed thesis produces weeks of misdirected
   data requests.
2. Category F (access constraints) governs what tools may run at all. If the user has
   not said whether customer contact is permitted, leave the field at its default and
   record the question as open - do not assume permission.
3. If the target is listed, three things are load-bearing and must be captured exactly
   as stated, never inferred. First, which structure is contemplated: a significant
   minority stake, a controlling stake with the listing retained, or a take-private.
   These ask different questions of the same company and the decomposition depends on
   knowing which. Second, the public-market block - ticker, exchange, the unaffected
   price and the date it refers to, share count, free float, insider holdings and
   voting structure. Do not populate a price or a float you were not given; an
   unstated figure is an open question, because every premium in the deck is measured
   from it. Third, whether the data room is expected to carry material non-public
   information and whether compliance has acknowledged the resulting trading
   restriction. Until that acknowledgement is recorded the data-room tools do not run
   at all, so a guess here does real damage in both directions.

Return the structured Deal Profile Brief."""


class IntakeExtraction(BaseModel):
    """What the model is asked to return. Narrower than DealProfile on purpose."""

    target: TargetIdentification
    public_market: PublicMarketContext = Field(
        default_factory=PublicMarketContext,
        description="Populate only for a listed target, only from stated facts.",
    )
    sector: SectorDefinition
    thesis: InvestmentThesis
    buyer: BuyerProfile
    process: ProcessContext = Field(default_factory=ProcessContext)
    access: AccessConstraints = Field(default_factory=AccessConstraints)
    deliverable: DeliverableParameters = Field(default_factory=DeliverableParameters)
    open_intake_questions: list[str] = Field(default_factory=list)


class IntakeAgent(Agent):
    role = AgentRole.INTAKE

    def opening_questions(self) -> list[tuple[str, str]]:
        """The protocol, ordered so the Phase-1 prerequisites come first."""
        ordered = sorted(
            INTAKE_PROTOCOL, key=lambda c: (not c.required_for_phase_1, c.key)
        )
        return [(c.title, q) for c in ordered for q in c.questions]

    def run(self, briefing: str, *, save: bool = True) -> DealProfile:
        """Turn a free-text briefing into the Deal Profile Brief."""
        if self.offline:
            extraction = self._offline_extract(briefing)
        else:
            from langchain_core.prompts import ChatPromptTemplate

            from cdd_agent.llm.models import get_chat_model

            prompt = ChatPromptTemplate.from_messages(
                [("system", _SYSTEM), ("human", "Deal team briefing:\n\n{briefing}")]
            )
            chain = prompt | get_chat_model().with_structured_output(IntakeExtraction)
            extraction = chain.invoke({"briefing": briefing})

        profile = DealProfile(
            engagement_id=self.context.engagement_id,
            created_by=self.name,
            target=extraction.target,
            public_market=extraction.public_market,
            sector=extraction.sector,
            thesis=extraction.thesis,
            buyer=extraction.buyer,
            process=extraction.process,
            access=extraction.access,
            deliverable=extraction.deliverable,
            open_intake_questions=extraction.open_intake_questions,
        )
        profile = self._add_unanswered(profile)
        if save:
            self.context.memory.save_deal_profile(profile, agent=self.name)
            self.context.profile = profile
        return profile

    def _add_unanswered(self, profile: DealProfile) -> DealProfile:
        """Append the Phase-1 prerequisites that are still missing.

        Deliberately additive rather than a rewrite: what the model reported as open
        stays open, and the deterministic check only adds what it can prove is missing.
        """
        ready, missing = profile.is_ready_for_phase_1()
        if ready:
            return profile
        existing = set(profile.open_intake_questions)
        for item in missing:
            if item not in existing:
                profile.open_intake_questions.append(item)
        return profile

    # ----------------------------------------------------------------- offline
    def _offline_extract(self, briefing: str) -> IntakeExtraction:
        """Deterministic path. Records the briefing verbatim and flags everything else.

        This does not attempt extraction: guessing structure from prose without a model
        would produce a Deal Profile that looks complete and is not, which is the exact
        failure the intake protocol exists to prevent.
        """
        first_line = next((ln.strip() for ln in briefing.splitlines() if ln.strip()), "")
        return IntakeExtraction(
            target=TargetIdentification(legal_name=first_line[:120] or "UNSPECIFIED"),
            sector=SectorDefinition(sub_sector=""),
            thesis=InvestmentThesis(one_sentence_thesis=""),
            buyer=BuyerProfile(),
            open_intake_questions=[
                f"[{c.key}] {q}" for c in INTAKE_PROTOCOL for q in c.questions
            ],
        )


def unanswered_by_category(profile: DealProfile) -> dict[str, list[str]]:
    """Group open questions by intake category for display."""
    out: dict[str, list[str]] = {}
    for question in profile.open_intake_questions:
        key = question[1] if question.startswith("[") and len(question) > 2 else "?"
        out.setdefault(key, []).append(question)
    return out
