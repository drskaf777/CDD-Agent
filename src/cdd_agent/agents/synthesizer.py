"""Synthesizer - turns the three artifacts into the draft presentation.

Reads the Hypothesis Tree, Evidence Matrix, and Risk Register and populates the enhanced
master outline section by section. It has no retrieval or outreach tools: everything it
can say must already exist as tagged, cited evidence in the store. That is not a
limitation to work around - it is what makes the output contract enforceable, because
there is no path by which an uncited assertion can enter the deck.

Two sections are generated rather than written: Section 0 renders the evidence-status
dashboard from the Evidence Matrix, and Section 8 renders the Risk Register. Both are
consequences of the data, so writing them by hand would let them drift from it.
"""

from __future__ import annotations

from typing import Optional

from cdd_agent.agents.base import Agent, AgentContext
from cdd_agent.guardrails.authorization import AgentRole
from cdd_agent.guardrails.output_contract import ContractReport, check_deck
from cdd_agent.knowledge.four_question_test import FOUR_QUESTIONS
from cdd_agent.knowledge.outline import tailored_outline
from cdd_agent.knowledge.risk_taxonomy import applicable_categories
from cdd_agent.schemas.common import ConfidenceTag, OutlineSection
from cdd_agent.schemas.deck import Claim, Deck, Exhibit, Slide
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree
from cdd_agent.schemas.risk import RiskRegister

_SYSTEM = """You are writing a commercial due diligence deck for an investment committee.

You are given a hypothesis tree, an evidence matrix, and a risk register. You may write
only what those support. Rules:

- Lead every page with a so-what headline: an evidence-backed assertion, not a topic
  label. "Retention is strong" is a label. "NRR of 118% is carried by 25% of ARR sitting
  in contracts that step down at renewal" is a headline.
- Every claim must map to evidence in the matrix and carry that evidence's confidence
  tag. Do not soften a Contradicted finding into a caveat, and do not present a
  Partially Confirmed finding as settled.
- Where the evidence for a claim is management-supplied only, say so in the claim.
- Never fill a gap with a plausible assertion. A logged gap is the correct output.
- This is a draft for partner/MD review, not an IC recommendation."""


class Synthesizer(Agent):
    role = AgentRole.SYNTHESIZER

    def run(
        self,
        tree: HypothesisTree,
        matrix: EvidenceMatrix,
        register: RiskRegister,
        *,
        save: bool = True,
    ) -> tuple[Deck, ContractReport]:
        profile = self.context.profile
        sub_sector = profile.sector.sub_sector if profile else ""
        business_model = profile.sector.business_model.value if profile else ""
        outline = tailored_outline(sub_sector, business_model)

        slides: list[Slide] = []
        for section in outline:
            if section.number == 0:
                slides.append(self._section_0(section, tree, matrix))
            elif section.number == 8:
                slides.append(self._section_8(section, register))
            else:
                slides.append(self._evidence_section(section, tree, matrix))

        deck = Deck(
            engagement_id=self.context.engagement_id,
            created_by=self.name,
            title=(
                f"Commercial Due Diligence - "
                f"{profile.target.legal_name if profile else self.context.engagement_id}"
            ),
            slides=slides,
        )

        report = check_deck(deck, tree=tree, matrix=matrix, register=register)
        # The contract is checked before the deck is saved, so a violating deck never
        # reaches the store and cannot be mistaken for a reviewed draft.
        report.raise_if_violated()
        if save:
            self.context.memory.save_deck(deck, agent=self.name)
        return deck, report

    # ------------------------------------------------------------- Section 0
    def _section_0(
        self, section: OutlineSection, tree: HypothesisTree, matrix: EvidenceMatrix
    ) -> Slide:
        """Deal Thesis & Hypothesis Tree, with the evidence-status dashboard.

        The deck shows the tree and each branch's evidence status, not just the
        conclusion (design spec s IV, enhancement 1).
        """
        rows: list[list[str]] = []
        for h in tree.tier_1():
            rating = matrix.rating(h.id)
            rows.append(
                [
                    h.id,
                    h.statement,
                    rating.value,
                    str(len(matrix.for_hypothesis(h.id))),
                    "yes" if matrix.triangulated(h.id) else "no",
                ]
            )

        all_citations = sorted(
            _all_citations(tree, matrix),
            key=lambda c: not c.source_kind.is_independent,
        )
        if all_citations:
            claims = [
                Claim(
                    text=f"Lead hypothesis: {tree.root_thesis}",
                    tag=_overall_tag(tree, matrix),
                    citations=all_citations[:3],
                    management_data_only=all(
                        c.source_kind.is_management_supplied for c in all_citations
                    ),
                )
            ]
        else:
            claims = [
                Claim(
                    text=f"Lead hypothesis: {tree.root_thesis} - no evidence gathered yet",
                    tag=ConfidenceTag.NO_DATA,
                )
            ]

        return Slide(
            section_number=section.number,
            section_title=section.title,
            so_what_headline=_dashboard_headline(tree, matrix),
            claims=claims,
            exhibits=[
                Exhibit(
                    title="Evidence-status dashboard",
                    kind="matrix",
                    columns=["ID", "Tier-1 hypothesis", "Status", "Items", "Triangulated"],
                    rows=rows,
                    note=f"Framing selected: {tree.framing_label}",
                ),
                Exhibit(
                    title="Four-question screening summary",
                    kind="table",
                    columns=["Question", "Mapped hypotheses"],
                    rows=[
                        [q.text, ", ".join(_hypotheses_for_question(tree, q.key)) or "-"]
                        for q in FOUR_QUESTIONS
                    ],
                ),
            ],
        )

    # ------------------------------------------------------------- Section 8
    def _section_8(self, section: OutlineSection, register: RiskRegister) -> Slide:
        """Risk Register & Outstanding Information Gaps, ranked by severity x likelihood."""
        risk_rows = [
            [
                r.id,
                r.category.value,
                r.description,
                str(r.severity),
                str(r.likelihood),
                str(r.score),
                "management data only" if r.management_data_only else "",
            ]
            for r in register.ranked()
        ]
        gap_rows = [
            [
                g.id,
                g.request,
                g.owner.value,
                g.target_close_date.isoformat() if g.target_close_date else "undated",
                "blocking" if g.blocking else "",
                "confirmatory" if g.carried_to_confirmatory else "",
            ]
            for g in register.gaps
            if not g.resolved
        ]
        top = register.ranked()[0] if register.risks else None
        headline = (
            f"Highest-ranked risk is {top.category.value.lower()} (severity x likelihood "
            f"= {top.score}); {len(register.open_blocking_gaps())} blocking gap(s) remain open"
            if top
            else "No risks raised - taxonomy coverage is incomplete, treat with suspicion"
        )
        return Slide(
            section_number=section.number,
            section_title=section.title,
            so_what_headline=headline,
            claims=[],
            exhibits=[
                Exhibit(
                    title="Consolidated risk register",
                    kind="table",
                    columns=["ID", "Category", "Finding", "Sev", "Lik", "Score", "Flags"],
                    rows=risk_rows,
                    note=(
                        "Taxonomy coverage: "
                        f"{register.coverage(applicable_categories(self.context.is_strategic_buyer)):.0%}"
                    ),
                ),
                Exhibit(
                    title="Outstanding information gaps",
                    kind="table",
                    columns=["ID", "Request", "Owner", "Target close", "Blocking", "Stage"],
                    rows=gap_rows,
                ),
            ],
        )

    # ------------------------------------------------- evidence-backed sections
    def _evidence_section(
        self, section: OutlineSection, tree: HypothesisTree, matrix: EvidenceMatrix
    ) -> Slide:
        """Populate one outline section from the evidence that bears on it."""
        relevant = _relevant_hypotheses(section, tree)
        claims: list[Claim] = []
        for h in relevant:
            items = matrix.for_hypothesis(h.id)
            rating = matrix.rating(h.id)
            if not items:
                claims.append(
                    Claim(
                        text=(
                            f"{h.statement} - no evidence retrieved; carried as a logged "
                            f"information gap"
                        ),
                        tag=ConfidenceTag.NO_DATA,
                        hypothesis_id=h.id,
                    )
                )
                continue
            management_only = all(i.source_kind.is_management_supplied for i in items)
            # Independent citations lead. A Confirmed rating is only reachable with
            # independent triangulation, so the citations shown have to include it -
            # otherwise the page reads as management-sourced when it is not.
            citations = sorted(
                (c for i in items for c in i.citations),
                key=lambda c: not c.source_kind.is_independent,
            )[:4]
            claims.append(
                Claim(
                    text=f"{h.statement} - {_evidence_gloss(items)}",
                    tag=rating,
                    citations=citations,
                    hypothesis_id=h.id,
                    management_data_only=management_only,
                )
            )

        return Slide(
            section_number=section.number,
            section_title=section.title,
            so_what_headline=self._headline(section, claims),
            claims=claims,
            exhibits=[
                Exhibit(
                    title=f"Section {section.number} coverage",
                    kind="table",
                    columns=["Key element", "Status"],
                    rows=[
                        [element, _element_status(element, claims)]
                        for element in section.key_elements
                    ],
                    note=(
                        "Status is the confidence tag of the claim covering that "
                        "element. No Data means the current evidence base says nothing "
                        "about it - which is the finding, not a formatting gap."
                    ),
                )
            ],
        )

    def _headline(self, section: OutlineSection, claims: list[Claim]) -> str:
        """Write the so-what headline.

        The model may only rewrite what the claims already say, and its output is
        rejected unless it stays within them - a headline is the one place where fluent
        prose could smuggle in an assertion the evidence does not carry.
        """
        deterministic = _section_headline(section, claims)
        if self.offline or not claims:
            return deterministic

        from langchain_core.prompts import ChatPromptTemplate

        from cdd_agent.llm.models import get_chat_model

        body = "\n".join(f"- {c.render()}" for c in claims)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
                (
                    "human",
                    "Section {number}: {title}\n\nEvidence-backed claims:\n{body}\n\n"
                    "Write one so-what headline of at most 30 words. Use only what the "
                    "claims above state. Do not introduce a number, name, or conclusion "
                    "that is not in them. Return the headline text only.",
                ),
            ]
        )
        try:
            response = (prompt | get_chat_model()).invoke(
                {"number": section.number, "title": section.title, "body": body}
            )
            headline = _response_text(response).strip().strip('"')
        except Exception:
            # A failed headline call must not take the deck down; the deterministic
            # headline is always available and is never ungrounded.
            return deterministic
        # A headline is one sentence. Anything wildly long is a malformed response,
        # not a headline - prefer the deterministic one over putting it on a slide.
        if not headline or len(headline) > 400:
            return deterministic
        return headline


# --------------------------------------------------------------------- helpers
def _response_text(response: object) -> str:
    """Pull the assistant text out of a chat response.

    On a thinking-enabled model the content is a *list* of blocks - thinking first,
    then text - not a string. Stringifying the list put raw thinking blocks, including
    their signatures, on the slides. Only text blocks are headline material.
    """
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _hypotheses_for_question(tree: HypothesisTree, key: str) -> list[str]:
    from cdd_agent.knowledge.four_question_test import classify

    return [
        h.id
        for h in tree.tier_1()
        if key in classify(f"{h.statement} {' '.join(h.required_evidence)}")
    ]


def _relevant_hypotheses(section: OutlineSection, tree: HypothesisTree) -> list:
    """Map hypotheses to outline sections by keyword overlap with the key elements."""
    from cdd_agent.agents.analyst import _keywords

    section_terms = _keywords(" ".join(section.key_elements) + " " + section.title)
    scored = []
    for h in tree.tier_1():
        overlap = len(section_terms & _keywords(h.statement + " " + " ".join(h.required_evidence)))
        if overlap:
            scored.append((overlap, h))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    # Sections 1 and 6 are summary sections: they carry every Tier-1 hypothesis.
    if section.number in (1, 6) or not scored:
        return list(tree.tier_1())
    return [h for _, h in scored]


def _evidence_gloss(items: list) -> str:
    tags = {i.tag.value for i in items}
    independent = sum(1 for i in items if i.is_independent)
    return (
        f"{len(items)} evidence item(s) ({', '.join(sorted(tags))}), "
        f"{independent} independently sourced"
    )


def _dashboard_headline(tree: HypothesisTree, matrix: EvidenceMatrix) -> str:
    tier1 = tree.tier_1()
    if not tier1:
        return "No Tier-1 hypotheses in the selected framing"
    counts: dict[str, int] = {}
    for h in tier1:
        counts[matrix.rating(h.id).value] = counts.get(matrix.rating(h.id).value, 0) + 1
    parts = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    return (
        f"{len(tier1)} Tier-1 hypotheses under the {tree.framing_label} framing: {parts}"
    )


def _section_headline(section: OutlineSection, claims: list[Claim]) -> str:
    if not claims:
        return f"{section.title}: no hypotheses map to this section"
    contradicted = [c for c in claims if c.tag is ConfidenceTag.CONTRADICTED]
    if contradicted:
        return f"Contradicted: {contradicted[0].text[:150]}"
    no_data = [c for c in claims if c.tag is ConfidenceTag.NO_DATA]
    if len(no_data) == len(claims):
        return (
            f"{section.title}: every hypothesis in this section is unevidenced - "
            "see the outstanding information gaps"
        )
    strongest = max(claims, key=lambda c: len(c.citations))
    return strongest.text[:180]


def _element_status(element: str, claims: list[Claim]) -> str:
    from cdd_agent.agents.analyst import _keywords

    terms = _keywords(element)
    for c in claims:
        if terms & _keywords(c.text):
            return c.tag.value
    return ConfidenceTag.NO_DATA.value


def _all_citations(tree: HypothesisTree, matrix: EvidenceMatrix) -> list:
    return [c for h in tree.tier_1() for i in matrix.for_hypothesis(h.id) for c in i.citations]


def _overall_tag(tree: HypothesisTree, matrix: EvidenceMatrix) -> ConfidenceTag:
    """The lead hypothesis is no stronger than its weakest Tier-1 branch."""
    ratings = [matrix.rating(h.id) for h in tree.tier_1()]
    if not ratings:
        return ConfidenceTag.NO_DATA
    if ConfidenceTag.CONTRADICTED in ratings:
        return ConfidenceTag.CONTRADICTED
    if ConfidenceTag.NO_DATA in ratings:
        return ConfidenceTag.PARTIALLY_CONFIRMED
    if all(r is ConfidenceTag.CONFIRMED for r in ratings):
        return ConfidenceTag.CONFIRMED
    return ConfidenceTag.PARTIALLY_CONFIRMED
