"""The Executive Dashboard: the diligence read in one view.

A view, deliberately, and not a second synthesis path. Every exhibit it shows has
already been through the same rule the draft applies - data plus a citation, or it
does not appear - and it is read back out of the saved deck rather than rebuilt. A
dashboard that could assemble its own charts would be a way around the discipline the
rest of the system exists to enforce: the most senior reader would be looking at the
least verified page.

What it adds over the draft is arrangement. The draft follows the master outline
because that is what an IC memo has to do. The dashboard follows the order a partner
reads in - the recommendation first, then what would break it, then the evidence
behind each - and says plainly, section by section, what is missing and what would
close it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cdd_agent.schemas.deck import Deck, Exhibit
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree
from cdd_agent.schemas.risk import RiskRegister

# Dashboard section -> the exhibits that belong to it, in reading order. Keys are
# catalogue keys, so an exhibit can appear here only if it exists in the catalogue
# and survived the presentability rule on its way into the deck.
LAYOUT: tuple[tuple[int, str, str, tuple[str, ...]], ...] = (
    (1, "Executive summary and investment thesis",
     "The recommendation, the pillars under it, and what would break it.",
     ("rule_of_40",)),
    (2, "Market dynamics and attractiveness",
     "Is the market genuinely growing, and is that growth structural?",
     ("tam_sam_som", "segmented_tam", "market_growth", "pestel", "cyclicality",
      "legacy_migration")),
    (3, "Competitive landscape and positioning",
     "Can the target keep winning share within it?",
     ("market_share", "landscape", "feature_bench", "moat", "switching_cost",
      "rd_efficiency", "ai_displacement", "win_loss")),
    (4, "Customer and voice-of-market analysis",
     "Do the customers behave the way the plan assumes?",
     ("nps", "concentration", "cohort_retention", "nrr_grr", "purchasing_criteria")),
    (5, "Financial and operational assessment",
     "Do the underlying unit economics hold up?",
     ("unit_economics", "magic_number", "arr_bridge", "pricing_elasticity", "ps_drag",
      "seats_consumption", "pipeline_health", "white_space", "guidance_delivery")),
    (6, "Valuation, sensitivities and growth levers",
     "What is it worth, and what has to be true for that to hold?",
     ("market_context", "consensus_vs_plan", "forecast_vs_base", "growth_levers",
      "exit_hypotheses")),
)


@dataclass
class Pillar:
    """One Tier-1 hypothesis, with how well it actually stood up."""
    id: str
    statement: str
    rating: str
    evidence_count: int
    independent: int
    management_only: bool


@dataclass
class Section:
    number: int
    title: str
    question: str
    exhibits: list[Exhibit] = field(default_factory=list)
    withheld: list[str] = field(default_factory=list)

    @property
    def shown(self) -> int:
        return len(self.exhibits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "title": self.title, "question": self.question,
            "exhibits": [e.model_dump(mode="json") for e in self.exhibits],
            "withheld": self.withheld,
        }


@dataclass
class Dashboard:
    engagement_id: str
    headline: str
    pillars: list[Pillar] = field(default_factory=list)
    red_flags: list[dict[str, Any]] = field(default_factory=list)
    upside: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    shown: int = 0
    withheld: int = 0
    groundedness: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engagement_id": self.engagement_id,
            "headline": self.headline,
            "pillars": [vars(p) for p in self.pillars],
            "red_flags": self.red_flags,
            "upside": self.upside,
            "sections": [s.to_dict() for s in self.sections],
            "shown": self.shown,
            "withheld": self.withheld,
            "groundedness": self.groundedness,
        }


def build(engagement_id: str, deck: Optional[Deck], tree: Optional[HypothesisTree],
          matrix: Optional[EvidenceMatrix], register: Optional[RiskRegister],
          gap_requests: Optional[dict[str, str]] = None) -> Dashboard:
    """Assemble the dashboard from artifacts that already exist.

    Nothing here builds an exhibit. Exhibits are lifted out of the saved deck by key,
    which means anything the draft withheld is withheld here too, for the same reason
    and with the same request recorded against it.
    """
    dash = Dashboard(engagement_id=engagement_id, headline="")
    if deck is None:
        return dash

    dash.groundedness = deck.groundedness()
    by_key: dict[str, Exhibit] = {}
    for slide in deck.slides:
        for exhibit in slide.exhibits:
            if exhibit.key:
                by_key.setdefault(exhibit.key, exhibit)

    # The so-what headline the Synthesizer wrote for Section 1, verbatim. The
    # dashboard does not get to write its own - that would be an uncited conclusion
    # on the most-read page in the pack.
    for slide in deck.slides:
        if slide.section_number == 1 and slide.so_what_headline:
            dash.headline = slide.so_what_headline
            break
    if not dash.headline and deck.slides:
        dash.headline = deck.slides[0].so_what_headline or ""

    if tree is not None and matrix is not None:
        for h in tree.tier_1():
            items = matrix.for_hypothesis(h.id)
            independent = sum(1 for i in items if i.is_independent)
            dash.pillars.append(Pillar(
                id=h.id,
                statement=" ".join(h.statement.split()),
                rating=matrix.rating(h.id).value,
                evidence_count=len(items),
                independent=independent,
                management_only=bool(items) and independent == 0,
            ))

    if register is not None:
        for risk in register.ranked()[:5]:
            dash.red_flags.append({
                "id": risk.id,
                "category": risk.category.value,
                "description": " ".join(risk.description.split()),
                "score": risk.score,
                "severity": risk.severity,
                "likelihood": risk.likelihood,
                "management_data_only": bool(risk.management_data_only),
            })

    # Upside is read off the growth-levers exhibit rather than invented, so it is
    # cited like anything else. Absent that exhibit the section stays empty and the
    # request shows up in the withheld list, which is the honest outcome.
    levers = by_key.get("growth_levers")
    if levers is not None and levers.rows:
        dash.upside = [" ".join(str(r[1]).split()) for r in levers.rows[:4] if len(r) > 1]

    requests = gap_requests or {}
    for number, title, question, keys in LAYOUT:
        section = Section(number=number, title=title, question=question)
        for key in keys:
            exhibit = by_key.get(key)
            if exhibit is not None:
                section.exhibits.append(exhibit)
            else:
                ask = requests.get(key)
                if ask:
                    section.withheld.append(ask)
        dash.sections.append(section)

    dash.shown = sum(s.shown for s in dash.sections)
    dash.withheld = sum(len(s.withheld) for s in dash.sections)
    return dash


def requests_by_key() -> dict[str, str]:
    """The data request behind each catalogue exhibit, keyed for the withheld list."""
    from cdd_agent.synthesis.exhibits import CATALOGUE

    return {spec.key: spec.requires for spec in CATALOGUE}
