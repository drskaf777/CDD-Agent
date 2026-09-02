"""Thought Generator - LangChain (LCEL).

Checkpoint 4.1 s 2.4: a structured one-shot transform (thesis + intake -> 3 candidate
trees). It needs templated prompting and structured-output parsing, not a persistent
persona, which is why it is an LCEL chain rather than a CrewAI agent.

The three framings are fixed by design, not chosen by the model: growth-led, margin-led,
and risk-led readings of the same thesis. Letting the model pick its own three would
reintroduce the anchoring this step exists to break - it would generate three variations
on whichever framing it thought of first.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from cdd_agent.config import get_settings
from cdd_agent.schemas.common import Tier
from cdd_agent.schemas.deal_profile import DealProfile, TransactionStructure
from cdd_agent.schemas.hypothesis import Hypothesis, HypothesisTree


@dataclass(frozen=True)
class Framing:
    key: str
    label: str
    instruction: str


FRAMINGS: tuple[Framing, ...] = (
    Framing(
        "growth",
        "growth-led",
        "Centre the decomposition on the market and the expansion thesis: is the "
        "demand there, and can the target capture it.",
    ),
    Framing(
        "margin",
        "margin-led",
        "Centre the decomposition on unit economics and cost structure: does the "
        "money the plan assumes actually drop through.",
    ),
    Framing(
        "risk",
        "risk-led",
        "Centre the decomposition on concentration and downside: what dependency or "
        "cliff would break the plan, and how load-bearing is it.",
    ),
)

# What each structure makes the decomposition responsible for. Kept here rather than
# in the prompt template because the same company under two structures is genuinely
# two different diligence questions, and the difference should be legible in code.
_PUBLIC_PREAMBLE = """
The target is listed. Its filings are public and its price already reflects a plan
that thousands of people have read. A hypothesis that restates published guidance or
consensus is not diligence - it is a summary. Every Tier-1 hypothesis must be
falsifiable against something the market has not already settled, and where the
thesis agrees with consensus, say so explicitly rather than presenting agreement as
a finding."""

_STRUCTURE_INSTRUCTIONS: dict[str, str] = {
    TransactionStructure.PUBLIC_MINORITY_STAKE.value: """
The buyer will hold a significant minority and cannot compel any decision. The plan
being underwritten is the incumbent management team's, and they cannot be replaced.
Decompose accordingly: hypotheses about what the buyer would do differently are
untestable here. Test whether the existing plan works, whether the buyer obtains
influence that is real rather than nominal, and whether value can be realised without
control - including whether a stake this size can be exited at all.""",
    TransactionStructure.PUBLIC_CONTROL_STAKE.value: """
The buyer takes control but the company remains listed, so minority shareholders
continue alongside. Value capture that requires related-party dealing, transfer
pricing, or the disclosure freedom of a private company is not available, and the
costs of remaining listed stay in the base case. Test the value-creation plan against
what a controlling shareholder of a listed company may actually do.""",
    TransactionStructure.TAKE_PRIVATE.value: """
The buyer must win a shareholder vote at a premium to the unaffected price. Two
consequences for the decomposition: the base case has to clear that premium, not
merely show a good business; and completion is a commercial question, not only a
legal one - customer and partner change-of-control consents that can be withheld are
revenue at risk on close. Delisting removes the public-company cost base, which is a
real and sizeable lever that must be evidenced rather than assumed.""",
}


def structure_brief(profile: DealProfile) -> str:
    """The structure-specific instruction block, empty for a private target."""
    if not profile.is_public_target:
        return ""
    structure = profile.target.transaction_structure
    return _PUBLIC_PREAMBLE + _STRUCTURE_INSTRUCTIONS.get(structure.value, "")


_SYSTEM = """You are decomposing an investment thesis into a testable hypothesis tree
for a commercial due diligence engagement.

{instruction}

Rules that are not negotiable:
- Produce {tier1_min} to {tier1_max} Tier-1 hypotheses at depth 1. Each must be a
  falsifiable claim that a real data room could confirm or contradict - not a topic
  heading, not a question.
- Between them, the Tier-1 hypotheses must collectively address all four screening
  questions: is the market genuinely growing; can the target keep winning share; do the
  unit economics hold; what specific findings would break the deal. A framing that
  leaves one unaddressed is discarded outright.
- Under each Tier-1 hypothesis, give 2 to 4 depth-2 supporting assumptions.
- For each hypothesis, state in `required_evidence` what specific artifact would move it
  to Confirmed or Contradicted. This seeds the Phase-2 data request, so "financial data"
  is useless; "contract-level ARR waterfall by month, reconciled to recognized revenue"
  is what is wanted.
- Reflect the buyer's stated decision criteria. This decomposition is scored against them.

Stay inside the commercial workstream. Financial, legal, tax, and technical diligence
are separate workstreams whose findings this engagement references but does not replicate."""

_HUMAN = """Target: {target}
Transaction structure: {structure}
Sub-sector: {sub_sector} ({business_model})
Investment thesis: {thesis}
Critical model assumptions: {assumptions}
Buyer: {buyer_type}; decision criteria: {criteria}
Base case: {growth} growth, {margin} margin expansion, {hold} year hold
{structure_brief}
{corrections}
Produce the {label} decomposition."""


class GeneratedHypothesis(BaseModel):
    statement: str = Field(description="A falsifiable claim.")
    rationale: str = ""
    required_evidence: list[str] = Field(default_factory=list)
    supporting_assumptions: list[str] = Field(default_factory=list)


class GeneratedFraming(BaseModel):
    tier_1: list[GeneratedHypothesis]


class ThoughtGenerator:
    """Generates the beam: one candidate tree per framing."""

    def __init__(self, profile: DealProfile) -> None:
        self.profile = profile
        self.settings = get_settings()

    def generate(self, prior_corrections: list[str] | None = None) -> list[HypothesisTree]:
        return [
            self._one(framing, prior_corrections or [])
            for framing in FRAMINGS[: self.settings.beam_width]
        ]

    def _one(self, framing: Framing, corrections: list[str]) -> HypothesisTree:
        if self.settings.offline:
            generated = _offline_framing(self.profile, framing)
        else:
            from langchain_core.prompts import ChatPromptTemplate

            from cdd_agent.llm.models import get_chat_model

            prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
            chain = prompt | get_chat_model().with_structured_output(GeneratedFraming)
            generated = chain.invoke(
                {
                    "instruction": framing.instruction,
                    "tier1_min": self.settings.tier1_min,
                    "tier1_max": self.settings.tier1_max,
                    "label": framing.label,
                    "target": self.profile.target.legal_name,
                    "structure": self.profile.target.transaction_structure.value,
                    "structure_brief": structure_brief(self.profile),
                    "sub_sector": self.profile.sector.sub_sector,
                    "business_model": self.profile.sector.business_model.value,
                    "thesis": self.profile.thesis.one_sentence_thesis,
                    "assumptions": "; ".join(self.profile.thesis.critical_model_assumptions)
                    or "not stated",
                    "buyer_type": self.profile.buyer.buyer_type.value,
                    "criteria": "; ".join(self.profile.buyer.decision_criteria) or "not stated",
                    "growth": self.profile.thesis.base_case_organic_growth or "unstated",
                    "margin": self.profile.thesis.base_case_margin_expansion or "unstated",
                    "hold": self.profile.thesis.hold_period_years or "unstated",
                    "corrections": _corrections_block(corrections),
                }
            )
        return _to_tree(self.profile, framing, generated)


def _corrections_block(corrections: list[str]) -> str:
    """Replay prior user corrections in the same sub-sector as context.

    Checkpoint 2.1: a correction should recalibrate how the agent scopes the next deal
    in the same sub-sector, rather than sitting inert in a log.
    """
    if not corrections:
        return ""
    body = "\n".join(f"- {c}" for c in corrections)
    return (
        "\nCorrections the deal team made on prior engagements in this sub-sector "
        f"(treat as guidance, not fact about this target):\n{body}\n"
    )


def _to_tree(
    profile: DealProfile, framing: Framing, generated: GeneratedFraming
) -> HypothesisTree:
    hypotheses: list[Hypothesis] = []
    for i, h in enumerate(generated.tier_1, start=1):
        hid = f"{framing.key.upper()}-H{i}"
        hypotheses.append(
            Hypothesis(
                id=hid,
                statement=h.statement,
                rationale=h.rationale,
                tier=Tier.DEAL_CRITICAL,
                depth=1,
                required_evidence=h.required_evidence,
            )
        )
        for j, assumption in enumerate(h.supporting_assumptions, start=1):
            hypotheses.append(
                Hypothesis(
                    id=f"{hid}.{j}",
                    statement=assumption,
                    tier=Tier.DEPTH_BUILDING,
                    depth=2,
                    parent_id=hid,
                )
            )
    return HypothesisTree(
        engagement_id=profile.engagement_id,
        created_by="Thesis Architect / Generator",
        branch_id=framing.key,
        framing_label=framing.label,
        root_thesis=profile.thesis.one_sentence_thesis,
        hypotheses=hypotheses,
    )


# --------------------------------------------------------------------- offline
_OFFLINE_TEMPLATES: dict[str, list[tuple[str, list[str], list[str]]]] = {
    "growth": [
        (
            "The addressable market for {sub_sector} grows at or above the rate embedded "
            "in the base case over the hold period",
            ["Bottom-up customer count reconciles to the stated TAM",
             "Segment growth is not concentrated in a sub-segment the target does not serve"],
            ["Third-party market sizing with a stated methodology",
             "Bottom-up build: addressable accounts x realistic penetration"],
        ),
        (
            "The target keeps winning share against incumbents and new entrants rather "
            "than growing only with the market",
            ["Win rates are stable or improving in competitive deals",
             "Retention holds in the cohorts most exposed to substitutes"],
            ["Win/loss log with reasons", "Cohort retention by customer size band"],
        ),
        (
            "The unit economics of incremental growth hold as the mix shifts, so growth "
            "does not dilute margin",
            ["CAC payback does not lengthen in newer cohorts",
             "Gross margin by segment does not fall as mix shifts"],
            ["CAC and payback by cohort", "Gross margin bridge by segment"],
        ),
        (
            "No single customer, channel, or regulatory dependency would break the growth "
            "plan if it moved against the target",
            ["Top-5 customer concentration is within tolerance",
             "No contract cliff or step-down falls inside the hold period"],
            ["Customer concentration schedule",
             "Top-20 contracts with renewal dates and pricing terms"],
        ),
    ],
    "margin": [
        (
            "Reported gross margin is sustainable and not flattered by mix, one-time "
            "items, or under-invested cost lines",
            ["Cost of delivery is fully loaded in gross margin",
             "Non-recurring items are excluded from the run-rate"],
            ["Cost structure by function", "Audited financials with the revenue bridge"],
        ),
        (
            "The margin expansion in the base case is driven by identified levers, not by "
            "an unexplained trend line",
            ["Each lever has a named owner and a quantified size",
             "Pricing power is evidenced by realized rate, not list price"],
            ["Operating model with the margin bridge",
             "Pricing schedules and realized discount history"],
        ),
        (
            "Demand growth in the served market is sufficient to carry volume at the "
            "assumed price points",
            ["Market growth does not depend on a segment the target cannot serve"],
            ["Third-party market growth data by segment"],
        ),
        (
            "The target can keep winning profitable share rather than buying it with "
            "discount",
            ["Discount depth is not trending up in won deals",
             "Churn is not concentrated in the highest-margin cohort"],
            ["Win/loss log with discount depth", "Cohort churn by margin band"],
        ),
        (
            "No cost, wage, or reimbursement step-change inside the hold period would "
            "break the margin plan",
            ["Key input costs are contracted or hedged through the hold period"],
            ["Supplier concentration and contract expiry schedule",
             "Wage inflation and turnover by revenue-generating function"],
        ),
    ],
    "risk": [
        (
            "Revenue quality withstands scrutiny: concentration, recurring share, and "
            "recognition are what the model assumes",
            ["Top-5 customer share is stable, not rising",
             "Recurring revenue is contractual, not habitual repeat purchase"],
            ["Customer-level billing detail", "Revenue recognition policy and audit notes"],
        ),
        (
            "The market is genuinely growing rather than being redistributed among "
            "incumbents",
            ["Independent market data supports the stated growth rate"],
            ["Third-party market study", "Bottom-up demand build"],
        ),
        (
            "Competitive position is defensible: the target keeps winning share against "
            "substitutes and low-cost entrants",
            ["Switching costs are real, not assumed from contract length",
             "No substitute technology is displacing the core use case"],
            ["Win/loss log", "Product usage telemetry by account"],
        ),
        (
            "Unit economics hold under the downside case, not only the base case",
            ["Contribution margin stays positive under the downside volume case"],
            ["Sensitivity model with driver ranges", "Cohort LTV and CAC payback"],
        ),
        (
            "Key-person, regulatory, and contractual exposures are identified, bounded, "
            "and priced into the plan",
            ["Revenue-carrying individuals are under retention agreements",
             "No licence, permit, or reimbursement rate expires unaddressed in the hold period"],
            ["Key-person and non-compete agreements",
             "Licence and regulatory filing schedule"],
        ),
    ],
}


def _offline_framing(profile: DealProfile, framing: Framing) -> GeneratedFraming:
    """Deterministic framings used when CDD_OFFLINE=1.

    Structurally faithful - each covers all four questions and carries specific
    required-evidence items - but generic to the framing rather than to this target.
    An offline run exercises the machinery; it is not diligence.
    """
    sub_sector = profile.sector.sub_sector or "the sub-sector"
    return GeneratedFraming(
        tier_1=[
            GeneratedHypothesis(
                statement=statement.format(sub_sector=sub_sector),
                rationale=f"{framing.label} reading of: {profile.thesis.one_sentence_thesis}",
                supporting_assumptions=assumptions,
                required_evidence=evidence,
            )
            for statement, assumptions, evidence in _OFFLINE_TEMPLATES[framing.key]
        ]
    )
