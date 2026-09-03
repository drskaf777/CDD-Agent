"""The standard commercial due diligence exhibit set.

A CDD report has a known repertoire: market sizing, competitive position, customer
health, plan validation, risk. This module declares that repertoire once, maps each
exhibit to the outline section it belongs in, and builds it from whatever the
engagement actually holds.

The discipline is the same one the rest of the system applies to claims. An exhibit is
built from arithmetic over parsed data-room tables (`COMPUTED`), or assembled from cited
evidence (`EVIDENCED`), or it is not built at all (`GAP`) - in which case it renders as
the specific data request that would close it, and that request is logged in the Risk
Register with a target date. A market-sizing waterfall with invented numbers would be
the most convincing thing in the deck and the most dangerous; a labelled gap is worth
more to an investment committee than a plausible chart.

Nothing here calls a model. Every number is computed or cited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from cdd_agent.knowledge.four_question_test import classify
from cdd_agent.schemas.common import Citation, SourceKind
from cdd_agent.schemas.deal_profile import (
    AccessConstraints,
    DealShape,
    PublicMarketContext,
    TransactionStructure,
)
from cdd_agent.schemas.deck import Exhibit, ExhibitStatus, Series
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree
from cdd_agent.schemas.risk import RiskCategory, RiskRegister
from cdd_agent.tools.structured_computation import (
    ComputationError,
    StructuredComputationTool,
)


@dataclass
class ExhibitSpec:
    """One standard exhibit: where it belongs, and what it needs to exist."""

    key: str
    title: str
    section: int
    kind: str
    # The data request that would let this exhibit be built for real.
    requires: str
    builder: str
    # Listed targets only, optionally narrowed to particular structures. A minority
    # holder and a bidder taking the company private are looking at the same company
    # and asking different questions of it, so they should not get the same exhibits.
    public_only: bool = False
    structures: tuple[str, ...] = ()

    def applies_to(self, shape: "DealShape | None") -> bool:
        if not self.public_only:
            return True
        if shape is None or not shape.public_target:
            return False
        return not self.structures or shape.structure.value in self.structures


# Ordered by outline section. Titles follow standard CDD practice so a reader
# recognises what is missing as readily as what is present.
CATALOGUE: tuple[ExhibitSpec, ...] = (
    # ---------------------------------------------------- Market dynamics (2)
    ExhibitSpec("tam_sam_som", "Market sizing: TAM / SAM / SOM", 2, "waterfall",
                "Third-party market sizing with stated methodology, plus a bottom-up "
                "build of addressable accounts x realistic penetration",
                "market_sizing"),
    ExhibitSpec("market_growth", "Historical and projected market growth (CAGR)", 2, "bar",
                "Market size by year for the last 3-5 years and a 5-year forecast, "
                "with the CAGR basis stated", "market_growth"),
    ExhibitSpec("pestel", "Macro factor analysis (PESTEL)", 2, "heatmap",
                "Evidence on political, economic, social, technological, environmental "
                "and legal factors bearing on the served segment", "pestel"),
    # ----------------------------------------------- Competitive landscape (3)
    ExhibitSpec("landscape", "Competitive landscape matrix", 3, "scatter",
                "Competitor list with product-completeness and market-share estimates",
                "landscape"),
    ExhibitSpec("feature_bench", "Competitor feature benchmarking", 3, "table",
                "Feature-by-feature comparison against named peers, with pricing model "
                "and service capability", "feature_bench"),
    ExhibitSpec("market_share", "Market share and concentration (HHI)", 3, "bar",
                "Revenue or share by competitor for the served segment",
                "market_share"),
    # ------------------------------------------------- Customer analysis (4)
    ExhibitSpec("cohort_retention", "Customer cohort retention curves", 4, "line",
                "Cohort-level revenue or usage by period, trailing 12 quarters minimum",
                "cohort_retention"),
    ExhibitSpec("nps", "NPS benchmarking against peers", 4, "bar",
                "NPS or CSAT survey history, with the industry benchmark and sample "
                "sizes disclosed", "nps"),
    ExhibitSpec("concentration", "Customer concentration risk (top 5 / 10 / 20)", 4, "bar",
                "Customer-level revenue detail", "concentration"),
    ExhibitSpec("win_loss", "Win/loss analysis by reason", 4, "bar",
                "Win/loss log with a categorised reason per deal", "win_loss"),
    # --------------------------------------- Plan validation and upside (6, 7)
    ExhibitSpec("arr_bridge", "ARR waterfall: new, expansion, contraction, churn", 6,
                "waterfall", "Contract-level ARR movement by month or quarter, "
                "reconciled to recognized revenue", "arr_bridge"),
    ExhibitSpec("forecast_vs_base", "Management forecast vs. diligence base case", 7, "line",
                "Management's operating model, with the assumptions behind each "
                "revenue line", "forecast_vs_base"),
    ExhibitSpec("growth_levers", "Growth levers: ease of execution vs. revenue impact",
                7, "scatter",
                "Named growth initiatives with sizing and an execution owner",
                "growth_levers"),
    ExhibitSpec("pricing_elasticity", "Pricing elasticity and headroom", 7, "table",
                "Realized price by cohort with churn at each price move",
                "pricing_elasticity"),
    # ------------------------------------------------------------- Risk (8)
    ExhibitSpec("margin_erosion", "Margin erosion risk", 8, "table",
                "Cost structure by function, supplier concentration, and wage inflation "
                "by revenue-generating role", "margin_erosion"),
    ExhibitSpec("substitution", "Technology and substitution risk", 8, "table",
                "Competitive intelligence on substitutes, open-source alternatives and "
                "platform bundling", "substitution"),
    ExhibitSpec("regulatory_heatmap", "Regulatory and compliance heatmap", 8, "heatmap",
                "Licence and permit schedule, plus pending regulatory change with "
                "expected effective dates", "regulatory_heatmap"),
    # --- Listed targets ---
    ExhibitSpec("ownership_control", "Ownership, control and register", 3, "table",
                "Shareholder register, free float, and any dual-class or founder "
                "holdings, with the disclosure threshold under the governing law",
                "ownership_control", public_only=True),
    ExhibitSpec("guidance_delivery", "Guidance to the market against delivery", 6,
                "table",
                "Guided figure and reported outcome for each of the last eight "
                "quarters, from the filings and transcripts",
                "guidance_delivery", public_only=True),
    ExhibitSpec("market_context", "Unaffected price and reference date", 7, "table",
                "Unaffected share price, the date the market was last uninformed of "
                "the approach, shares outstanding, and the 52-week range",
                "market_context", public_only=True),
    ExhibitSpec("consensus_vs_plan",
                "Published consensus against the management plan", 7, "table",
                "Consensus estimates with dispersion, and the management operating "
                "model on the same line items",
                "consensus_vs_plan", public_only=True),
    ExhibitSpec("mnpi_status", "MNPI and wall-crossing status", 8, "table",
                "Compliance confirmation of the trading restriction and the parties "
                "wall-crossed",
                "mnpi_status", public_only=True),
    ExhibitSpec("influence_rights", "Influence rights secured", 8, "table",
                "Draft shareholder agreement or term sheet: board seats, consent "
                "rights, information rights and standstill",
                "influence_rights", public_only=True,
                structures=(TransactionStructure.PUBLIC_MINORITY_STAKE.value,)),
    ExhibitSpec("completion_conditions", "Completion conditions and approvals", 8,
                "table",
                "Constitutional approval thresholds, regulatory clearance list, and "
                "any change-of-control consents in the top customer contracts",
                "completion_conditions", public_only=True,
                structures=(TransactionStructure.PUBLIC_CONTROL_STAKE.value,
                            TransactionStructure.TAKE_PRIVATE.value)),
)

def specs_for_section(section: int,
                      shape: "DealShape | None" = None) -> list[ExhibitSpec]:
    return [s for s in CATALOGUE if s.section == section and s.applies_to(shape)]


# --------------------------------------------------------------------- context
# The Analyst wraps a retrieved passage in "Retrieved evidence bearing on H1: ...".
# That preamble is our own bookkeeping, not something a source said, so it is stripped
# before any sentence is quoted - otherwise the scaffolding appears in the deck as
# though the board deck had asserted it.
_CLAIM_PREAMBLE = re.compile(r"^Retrieved evidence bearing on [\w.-]+:\s*")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?<=\.)\s(?=[A-Z])")

# A figure of the shape the exhibit's title promises. An exhibit headed "TAM / SAM /
# SOM" that shows a sentence with no magnitude in it is asserting, by placement, that
# a size was supplied.
MONEY = re.compile(
    r"[$\u00a3\u20ac]\s?\d[\d,.]*\s*(?:bn|b\b|billion|m\b|mm|million|k\b)?"
    r"|\b\d[\d,.]*\s*(?:bn|billion|million|trillion)\b",
    re.IGNORECASE,
)
PERCENT = re.compile(r"\d[\d,.]*\s?%|\bcagr\b", re.IGNORECASE)
FIGURE = re.compile(r"\d")

@dataclass
class ExhibitContext:
    tree: HypothesisTree
    matrix: EvidenceMatrix
    register: RiskRegister
    computation: Optional[StructuredComputationTool]
    shape: DealShape = field(default_factory=DealShape)
    # Public-market facts the intake supplied. Never inferred, so an exhibit that
    # needs a share price and has none is a gap like any other.
    public: PublicMarketContext = field(default_factory=PublicMarketContext)
    access: Optional[AccessConstraints] = None

    @property
    def strategic_buyer(self) -> bool:
        return self.shape.strategic_buyer

    def evidence_about(self, *terms: str) -> list:
        """Evidence whose claim or quoted source mentions any of these terms.

        Topical only. Use it to decide whether a subject was *discussed*; use
        `statements` when the exhibit asserts that a figure was actually given.
        """
        needles = [t.lower() for t in terms]
        found = []
        for item in self.matrix.items:
            haystack = (
                item.claim + " " + " ".join(c.quoted_text for c in item.citations)
            ).lower()
            if any(n in haystack for n in needles):
                found.append(item)
        return found

    def statements(self, *terms: str, figure=None, limit: int = 5) -> list[tuple]:
        """The individual sentences that state the thing, not the chunks around them.

        Matching a whole retrieved chunk on a keyword is how a sizing exhibit ends up
        full of evidence that never states a size: the chunk mentions "market", the
        cell is duly cited, and the reader infers a number that nobody supplied.
        So a sentence qualifies only if it both mentions the subject and carries the
        shape of figure the exhibit claims to display.

        Each hit comes back with the evidence item re-ordered so that the citation the
        sentence was actually taken from is first. An evidence item routinely carries
        several citations, and naming the first one regardless put a quote from the
        earnings call under the board deck - a sourcing error, and the kind this
        system exists to prevent rather than commit.
        """
        needles = [t.lower() for t in terms]
        out: list[tuple] = []
        seen: set[str] = set()
        for item in self.matrix.items:
            # (text, the citation it came from) - the claim is the Analyst own
            # wording, so it has no single source document of its own.
            sources: list[tuple[str, object]] = []
            if not all(c.source_kind is SourceKind.KNOWLEDGE_BASE for c in item.citations):
                sources.append((_CLAIM_PREAMBLE.sub("", item.claim), None))
            # The Knowledge Base holds cross-engagement *method* references - the
            # outline, the risk taxonomy, the data-request catalogue. They describe how
            # diligence is done, not anything about this target, so a sentence lifted
            # from one and placed in an exhibit reads as a finding about the company
            # when it is nothing of the kind. Retrieval may still use them for context;
            # they simply cannot be quoted as evidence.
            sources += [(c.quoted_text, c) for c in item.citations
                        if c.quoted_text and c.source_kind is not SourceKind.KNOWLEDGE_BASE]
            for text, citation in sources:
                for sentence in _SENTENCE_SPLIT.split(text):
                    sentence = " ".join(sentence.split())
                    if not (12 <= len(sentence) <= 320):
                        continue
                    # A quotable finding is a whole declarative sentence. Retrieval
                    # hands back chunk-boundary fragments ("Now the platform vendors
                    # bundle") and interview prompts ("Q: How do you evaluate
                    # vendors?"); quoting either as a finding attributes to the source
                    # something it did not assert.
                    if not sentence.endswith((".", "!")) or len(sentence.split()) < 5:
                        continue
                    if sentence.startswith(("Q:", "Q.")) or "?" in sentence:
                        continue
                    low = sentence.lower()
                    if not any(n in low for n in needles):
                        continue
                    if figure is not None and not figure.search(sentence):
                        continue
                    key = low[:80]
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((_attributed(item, citation), sentence))
                    if len(out) >= limit:
                        return out
        return out



def _attributed(item, citation):
    """The evidence item with the citation this sentence came from placed first.

    A copy, not a mutation: the matrix is shared across every exhibit built in this
    pass, and re-ordering it in place would make one exhibit change what the next one
    cites.
    """
    if citation is None or not item.citations:
        return item
    rest = [c for c in item.citations if c is not citation]
    return item.model_copy(update={"citations": [citation] + rest})

# Exhibits whose content is arithmetic over parsed data-room tables. Design spec s VI
# puts quantitative analysis at step 4, under the Analyst, before slide generation at
# step 6 - so these are computed in Phase 3 and merely rendered by the Synthesizer,
# which keeps the Synthesizer free of tools it should not have.
COMPUTED_KEYS: frozenset[str] = frozenset(
    {"concentration", "cohort_retention", "arr_bridge", "market_share"}
)


def _gap(spec: ExhibitSpec, why: str = "") -> Exhibit:
    """An exhibit that cannot be built, rendered as the request that would build it."""
    return Exhibit(
        key=spec.key,
        title=spec.title,
        kind=spec.kind,
        status=ExhibitStatus.GAP,
        gap_request=spec.requires,
        note=why or "Not evidenced in the material supplied.",
    )


# -------------------------------------------------------------------- builders
def _sourcing(item) -> str:
    return "independent" if item.is_independent else "management"


def _cite(item) -> str:
    return item.citations[0].short() if item.citations else "-"


def _citations_of(pairs, limit: int = 4) -> list:
    return [c for item, _ in pairs for c in item.citations][:limit]

def market_sizing(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """TAM / SAM / SOM.

    Deliberately never inferred. A sizing waterfall is the most quoted exhibit in a
    CDD deck; producing one from a model's prior would be indistinguishable from
    research and completely unfounded. It requires a stated magnitude, not merely a
    document that discusses the market - an exhibit under this title implies a number
    was supplied, and a cited sentence containing no number still implies it.
    """
    pairs = ctx.statements("tam", "addressable market", "market siz", "sam",
                           figure=MONEY, limit=4)
    if not pairs:
        return _gap(spec)
    layers = (("tam", "TAM"), ("addressable", "TAM"), ("sam", "SAM"), ("som", "SOM"))
    rows = []
    for item, sentence in pairs:
        low = sentence.lower()
        layer = next((label for k, label in layers if k in low), "As supplied")
        rows.append([layer, sentence, _cite(item)])
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Layer", "Stated in the data room", "Source"],
        rows=rows, citations=_citations_of(pairs),
        note="Management-supplied sizing, reproduced as stated. It has not been "
             "rebuilt bottom-up, so it is not independent corroboration.",
    )


def market_growth(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    pairs = ctx.statements("cagr", "growth", "grew", "growing", "expand",
                           figure=PERCENT, limit=5)
    if not pairs:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Growth statement", "Sourcing", "Source"],
        rows=[[s, _sourcing(i), _cite(i)] for i, s in pairs],
        citations=_citations_of(pairs),
        note="Growth rates as stated in the material. A CAGR that cannot be "
             "reconciled to a bottom-up build is logged under Growth sustainability.",
    )


_PESTEL = (
    ("Political", ("policy", "government", "political", "tariff")),
    ("Economic", ("inflation", "interest rate", "recession", "wage", "pricing pressure")),
    ("Social", ("adoption", "workforce", "talent", "demographic")),
    ("Technological", ("ai ", "open-source", "open source", "platform", "substitut",
                       "displacement", "cloud")),
    # "sustainab" matched "a sustainable retention figure" and filed a financial
    # statement under Environmental. A marker has to be unambiguous in context or it
    # produces a confidently mislabelled row.
    ("Environmental", ("environmental", "carbon", "emission", "esg",
                       "energy consumption", "data centre power", "data center power")),
    ("Legal", ("regulat", "compliance", "licence", "license", "gdpr", "hipaa",
               "privacy", "litigation")),
)


def pestel(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    rows, citations = [], []
    for factor, markers in _PESTEL:
        # The sentence that mentions the factor, not the chunk that contains it. A
        # factor marked "evidenced" on the strength of the word "cloud" appearing
        # somewhere nearby, illustrated by an unrelated claim, is worse than a blank.
        hits = ctx.statements(*markers, limit=1)
        if hits:
            item, sentence = hits[0]
            rows.append([factor, "evidenced", sentence, _cite(item)])
            citations += item.citations[:1]
        else:
            rows.append([factor, "no data", "Not addressed by the material supplied", "-"])
    if all(r[1] == "no data" for r in rows):
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="heatmap", status=ExhibitStatus.EVIDENCED,
        columns=["Factor", "Status", "What the evidence says", "Source"],
        rows=rows, citations=citations[:6],
        note="Factors with no data are unexamined, not benign.",
    )


def landscape(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    return _gap(spec, "No competitor-level positioning data in the material supplied.")


def feature_bench(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    # No figure required: this one is honestly qualitative, and its note says so.
    pairs = ctx.statements("competitor", "competitive", "vendor", "bundl",
                           "displacement", "substitut", "rival", limit=6)
    if not pairs:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Competitive observation", "Sourcing", "Source"],
        rows=[[s, _sourcing(i), _cite(i)] for i, s in pairs],
        citations=_citations_of(pairs),
        note="Qualitative until a feature-by-feature comparison against named peers "
             "is supplied.",
    )


def market_share(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """Share and HHI. Computed only where per-competitor revenue exists."""
    if ctx.computation is None:
        return _gap(spec)
    for table in ctx.computation.available():
        if "competitor" not in table and "share" not in table:
            continue
        try:
            result = ctx.computation.herfindahl(table, "competitor", "revenue")
        except ComputationError:
            continue
        return Exhibit(
            title=spec.title, kind="bar", status=ExhibitStatus.COMPUTED,
            columns=["Competitor", "Share"],
            rows=[[r["name"], f"{r['share']:.1%}"] for r in result.table],
            series=[Series(name="share", unit="%",
                           labels=[r["name"] for r in result.table],
                           values=[r["share"] for r in result.table])],
            citations=[result.citation] if result.citation else [],
            note=f"HHI {result.value:,.0f} - "
                 f"{'concentrated' if result.value and result.value > 2500 else 'unconcentrated'} "
                 "on the standard thresholds.",
        )
    return _gap(spec)


def cohort_retention(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    if ctx.computation is None:
        return _gap(spec)
    for table in ctx.computation.available():
        if "cohort" not in table:
            continue
        try:
            result = ctx.computation.retention_cohorts(
                table, cohort_column="cohort", period_column="period",
                value_column="net_arr")
        except ComputationError:
            continue
        cohorts: dict[str, list[tuple[str, float]]] = {}
        for row in result.table:
            cohorts.setdefault(str(row["cohort"]), []).append(
                (str(row["period"]), float(row["retention"])))
        return Exhibit(
            title=spec.title, kind="line", status=ExhibitStatus.COMPUTED,
            columns=["Cohort", "Period", "Retention"],
            rows=[[str(r["cohort"]), str(r["period"]), f"{r['retention']:.0%}"]
                  for r in result.table],
            series=[Series(name=name, unit="%",
                           labels=[p for p, _ in points],
                           values=[v for _, v in points])
                    for name, points in sorted(cohorts.items())],
            citations=[result.citation] if result.citation else [],
            note="Retention indexed to each cohort's first observed period.",
        )
    return _gap(spec)


def nps(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    pairs = ctx.statements("nps", "net promoter", "csat", "satisfaction",
                           figure=FIGURE, limit=4)
    if not pairs:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Observation", "Sourcing", "Source"],
        rows=[[s, _sourcing(i), _cite(i)] for i, s in pairs],
        citations=_citations_of(pairs, 3),
        note="A peer benchmark requires the industry comparison set and disclosed "
             "sample sizes; without them this is not a benchmark.",
    )


def concentration(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    if ctx.computation is None:
        return _gap(spec)
    for table in ctx.computation.available():
        if "customer" not in table and "revenue" not in table:
            continue
        for customer_col, revenue_col in (("customer", "arr"), ("customer", "revenue")):
            try:
                result = ctx.computation.customer_concentration(
                    table, customer_col, revenue_col)
            except ComputationError:
                continue
            return Exhibit(
                title=spec.title, kind="bar", status=ExhibitStatus.COMPUTED,
                columns=["Bucket", "Revenue", "Share of total"],
                rows=[[r["bucket"], f"{r['revenue']:,.0f}", f"{r['share_of_total']:.1%}"]
                      for r in result.table],
                series=[Series(name="share of revenue", unit="%",
                               labels=[r["bucket"] for r in result.table],
                               values=[r["share_of_total"] for r in result.table])],
                citations=[result.citation] if result.citation else [],
                note=result.note,
            )
    return _gap(spec)


def win_loss(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    pairs = ctx.statements("win rate", "win/loss", "won", "lost", "renewal price",
                           "displaced", figure=FIGURE, limit=5)
    if not pairs:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Observation", "Sourcing", "Source"],
        rows=[[s, _sourcing(i), _cite(i)] for i, s in pairs],
        citations=_citations_of(pairs, 3),
        note="Categorised win/loss reasons require the deal-level log.",
    )


def arr_bridge(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    if ctx.computation is None:
        return _gap(spec)
    for table in ctx.computation.available():
        if "arr" not in table and "waterfall" not in table:
            continue
        try:
            result = ctx.computation.arr_bridge(
                table, period_column="period", new_column="new",
                expansion_column="expansion", contraction_column="contraction",
                churn_column="churn", opening_column="opening_arr")
        except ComputationError:
            continue
        return Exhibit(
            title=spec.title, kind="waterfall", status=ExhibitStatus.COMPUTED,
            columns=result.columns,
            rows=[[str(r[c]) for c in result.columns] for r in result.table],
            series=[
                Series(name=movement, unit="currency",
                       labels=[str(r["period"]) for r in result.table],
                       values=[float(r[movement]) for r in result.table])
                for movement in ("new", "expansion", "contraction", "churn")
            ],
            citations=[result.citation] if result.citation else [],
            note=result.note + ". Closing balance is computed from the movements, "
                               "not read from a summary row.",
        )
    return _gap(spec)


def forecast_vs_base(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """Management case against a risk-adjusted base case.

    Built only from a stated closing balance and the downside implied by evidence.
    Without the operating model there is no forecast to test, and inventing one would
    be inventing the very thing diligence exists to challenge.
    """
    return _gap(spec, "No operating model supplied, so there is no management forecast "
                      "to test a base case against.")


def growth_levers(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    levers = []
    for h in ctx.tree.tier_1():
        keys = classify(h.statement)
        if "target_keeps_winning" in keys or "market_growing" in keys:
            rating = ctx.matrix.rating(h.id)
            levers.append([h.id, " ".join(h.statement.split())[:150], rating.value,
                           str(len(ctx.matrix.for_hypothesis(h.id)))])
    if not levers:
        return _gap(spec)
    citations = _target_citations(
        [item for row in levers for item in ctx.matrix.for_hypothesis(row[0])]
    )
    if not citations:
        return _gap(spec, "No evidence yet supports the levers the thesis depends on.")
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Hypothesis", "Lever under test", "Evidence status", "Items"],
        rows=levers, citations=citations,
        note="Ease-of-execution scoring requires named initiatives with owners and "
             "sizing; these are the growth levers the thesis actually depends on.",
    )


def pricing_elasticity(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    pairs = ctx.statements("pricing", "price", "discount", "escalator", "uplift",
                           "step-down", "step down", figure=FIGURE, limit=6)
    if not pairs:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Pricing observation", "Sourcing", "Source"],
        rows=[[s, _sourcing(i), _cite(i)] for i, s in pairs],
        citations=_citations_of(pairs),
        note="Elasticity proper requires realized price by cohort with churn at each "
             "price move.",
    )


def _target_citations(items, limit: int = 6) -> list:
    """Citations that actually say something about the target.

    Knowledge-Base chunks are cross-engagement method references - the outline, the
    risk taxonomy, the data-request catalogue. A risk table sourced to our own
    taxonomy document is circular: it cites the list the risk was screened from, not
    evidence that the risk is real here. Dropping them can leave an exhibit unsourced,
    which is the correct outcome - it is then omitted and logged as a data request.
    """
    return [
        c for item in items for c in item.citations
        if c.source_kind is not SourceKind.KNOWLEDGE_BASE
    ][:limit]


def _risk_table(ctx: ExhibitContext, spec: ExhibitSpec,
                categories: tuple[RiskCategory, ...]) -> Exhibit:
    risks = [r for r in ctx.register.ranked() if r.category in categories]
    if not risks:
        return _gap(spec)
    # Carry through the citations of the evidence each risk was raised from. A risk
    # table with no traceable source is an assertion, and an exhibit that cannot be
    # sourced does not belong in the report at all.
    wanted = {eid for r in risks for eid in r.evidence_ids}
    citations = _target_citations(
        [item for item in ctx.matrix.items if item.id in wanted]
    )
    if not citations:
        return _gap(spec, "The findings in this category carry no traceable source.")
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["ID", "Finding", "Sev", "Lik", "Score", "Flags"],
        rows=[[r.id, r.description, str(r.severity), str(r.likelihood), str(r.score),
               "management data only" if r.management_data_only else ""] for r in risks],
        citations=citations,
        note="Drawn from the Risk Register, so it cannot disagree with Section 8.",
    )


def margin_erosion(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    return _risk_table(ctx, spec,
                       (RiskCategory.UNIT_ECONOMICS, RiskCategory.RETENTION))


def substitution(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    return _risk_table(ctx, spec, (RiskCategory.COMPETITIVE_DISRUPTION,))


def regulatory_heatmap(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    return _risk_table(ctx, spec, (RiskCategory.REGULATORY,))



# ------------------------------------------------- listed-target exhibits
def _intake_citation(locator: str) -> Citation:
    """Attribute a buyer-stated fact to the artifact that recorded it.

    These figures are not evidence in the diligence sense - nobody has verified them
    - but they are traceable to a stored artifact with an author and a timestamp,
    which is the standard every other cell in the deck is held to.
    """
    return Citation(source_kind=SourceKind.INTAKE,
                    source_file="Deal Profile Brief (intake)", locator=locator)


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        text = f"{value:,.2f}".rstrip("0").rstrip(".")
        return f"{text}{suffix}"
    return f"{value}{suffix}"


def market_context(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """The price the deal is measured against.

    A premium is only meaningful against an unaffected price on a stated date. Where
    intake gave neither, this is a gap: an exhibit headed "unaffected price" that
    quietly used the latest close would misstate every premium computed from it.
    """
    p = ctx.public
    rows = []
    if p.unaffected_share_price is not None:
        rows.append(["Unaffected share price",
                     _fmt(p.unaffected_share_price) + (f" {p.currency}" if p.currency else ""),
                     _fmt(p.unaffected_price_date) or "date not stated"])
    if p.unaffected_share_price is not None and p.shares_outstanding_m is not None:
        equity = p.unaffected_share_price * p.shares_outstanding_m
        rows.append(["Equity value at the unaffected price",
                     f"{equity:,.0f}m" + (f" {p.currency}" if p.currency else ""),
                     "computed: price x shares outstanding"])
    if p.shares_outstanding_m is not None:
        rows.append(["Shares outstanding", _fmt(p.shares_outstanding_m, "m"), "as stated at intake"])
    if not rows:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Measure", "Value", "Basis"], rows=rows,
        citations=[_intake_citation("Category A: public-market context")],
        note="As stated at intake and not verified against market data. The premium "
             "the base case must clear is measured from this price, so an incorrect "
             "reference date misstates every figure derived from it.",
    )


def ownership_control(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """Who actually decides the outcome, which is rarely the largest bidder."""
    p = ctx.public
    fields = (
        ("Free float", _fmt(p.free_float_pct, "%")),
        ("Insider / founder holding", _fmt(p.insider_or_founder_stake_pct, "%")),
        ("Dual-class share structure", _fmt(p.dual_class_shares)),
        ("Activist holder on the register", _fmt(p.activist_holder_present)),
        ("Index membership", ", ".join(p.index_memberships)),
        ("Disclosure threshold", _fmt(p.disclosure_threshold_pct, "%")),
        ("Approval threshold", _fmt(p.shareholder_approval_threshold_pct, "%")),
        ("Analyst coverage", _fmt(p.analyst_coverage_count, " brokers")),
    )
    rows = [[label, value] for label, value in fields if value]
    if not rows:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Register / governance fact", "Position"], rows=rows,
        citations=[_intake_citation("Category A: ownership and governance")],
        note="Blank lines are unknowns, not zeroes. A dual-class structure or a "
             "concentrated insider holding can decide the outcome regardless of the "
             "stake acquired.",
    )


def mnpi_status(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """What the compliance position permits, stated in the deck rather than assumed."""
    if not ctx.shape.public_target:
        return _gap(spec)
    a = ctx.access
    if a is None:
        return _gap(spec)
    rows = [
        ["Data room expected to carry MNPI", _fmt(a.mnpi_expected)],
        ["Trading restriction acknowledged", _fmt(a.trading_restriction_acknowledged)],
        ["Issuer / insider contact permitted", _fmt(a.issuer_contact_permitted)],
    ]
    if a.wall_crossed_parties:
        rows.append(["Wall-crossed parties", ", ".join(a.wall_crossed_parties)])
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Compliance position", "Status"], rows=rows,
        citations=[_intake_citation("Category F: access and MNPI constraints")],
        note="Restrictions in force until announcement. Where issuer contact is not "
             "permitted, questions that would need it are carried to confirmatory "
             "diligence rather than answered from inference.",
    )


def consensus_vs_plan(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """What the market already expects, set against what management is showing us.

    The point of the exhibit is the gap. If the plan in the data room is the plan the
    street has already published, the buyer is paying a premium for growth that is
    priced, and that conclusion is worth stating plainly.
    """
    # Deliberately not "estimate": every filing says "assumptions and estimates used
    # in preparing our consolidated financial statements", and matching that put a
    # stock-compensation note into the deck under Published consensus.
    consensus = [
        (i, s) for i, s in ctx.statements(
            "consensus", "analyst", "sell-side", "sell side", "price target",
            "covering analysts", figure=FIGURE, limit=4)
    ]
    guidance = [
        (i, s) for i, s in ctx.statements(
            "guidance", "outlook", "management plan", "budget", "forecast",
            figure=FIGURE, limit=4)
    ]
    if not consensus and not guidance:
        return _gap(spec)
    rows = [["Published consensus", s, _cite(i)] for i, s in consensus]
    rows += [["Management plan", s, _cite(i)] for i, s in guidance]
    citations = _citations_of(consensus + guidance)
    if not citations:
        return _gap(spec)
    note = ("Consensus is not corroboration of the plan - analysts are guided by the "
            "company, so agreement between the two is one source counted twice.")
    if consensus and not guidance:
        note += " No management plan located, so the comparison is one-sided."
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["View", "As stated", "Source"], rows=rows,
        citations=citations, note=note,
    )


def guidance_delivery(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """Management credibility measured on their own public record.

    The cheapest test available on a listed target, and one a private company cannot
    be given: they have told the market what they would do, repeatedly, and the
    results were published.
    """
    pairs = ctx.statements("guidance", "guided", "outlook", "raised", "lowered",
                           "beat", "missed", "reiterated", figure=FIGURE, limit=6)
    attested = [(i, s) for i, s in pairs
                if any(c.source_kind.is_attested or c.source_kind.is_public_record
                       for c in i.citations)]
    use = attested or pairs
    if not use:
        return _gap(spec)
    citations = _citations_of(use)
    if not citations:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Statement to the market", "Source kind", "Source"],
        rows=[[s, i.citations[0].source_kind.value.replace("_", " ") if i.citations else "-",
               _cite(i)] for i, s in use],
        citations=citations,
        note="A guidance-against-delivery track record needs the guided figure and "
             "the reported outcome side by side for each period; these are the "
             "statements located so far." if not attested else
             "Drawn from the public record, which is attested and independently "
             "readable - the strongest evidence available before the data room opens.",
    )


def influence_rights(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """For a minority stake: whether the buyer can affect the plan it is funding."""
    pairs = ctx.statements("board seat", "governance", "consent", "standstill",
                           "shareholder agreement", "information rights", "observer",
                           "veto", limit=5)
    if not pairs:
        return _gap(spec)
    citations = _citations_of(pairs)
    if not citations:
        return _gap(spec)
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Right under negotiation", "Sourcing", "Source"],
        rows=[[s, _sourcing(i), _cite(i)] for i, s in pairs],
        citations=citations,
        note="Without secured rights the plan is a forecast of what incumbent "
             "management will choose to do, and should be underwritten as such.",
    )


def completion_conditions(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    """Everything between an agreed price and actually owning the company."""
    return _risk_table(ctx, spec, (RiskCategory.DEAL_COMPLETION,))


_BUILDERS: dict[str, Callable[[ExhibitContext, ExhibitSpec], Exhibit]] = {
    "market_sizing": market_sizing, "market_growth": market_growth, "pestel": pestel,
    "landscape": landscape, "feature_bench": feature_bench, "market_share": market_share,
    "cohort_retention": cohort_retention, "nps": nps, "concentration": concentration,
    "win_loss": win_loss, "arr_bridge": arr_bridge,
    "forecast_vs_base": forecast_vs_base, "growth_levers": growth_levers,
    "pricing_elasticity": pricing_elasticity, "margin_erosion": margin_erosion,
    "substitution": substitution, "regulatory_heatmap": regulatory_heatmap,
    "market_context": market_context, "ownership_control": ownership_control,
    "mnpi_status": mnpi_status, "consensus_vs_plan": consensus_vs_plan,
    "guidance_delivery": guidance_delivery,
    "influence_rights": influence_rights,
    "completion_conditions": completion_conditions,
}


def _build_one(spec: ExhibitSpec, ctx: ExhibitContext) -> Exhibit:
    builder = _BUILDERS[spec.builder]
    try:
        exhibit = builder(ctx, spec)
    except Exception:
        # A builder that fails must not take the deck down, and must not quietly
        # vanish either - the exhibit is still owed, so it becomes a gap.
        return _gap(spec, "Could not be built from the material supplied.")
    if not exhibit.key:
        exhibit = exhibit.model_copy(update={"key": spec.key})
    return exhibit


def build_computed(ctx: ExhibitContext) -> list[Exhibit]:
    """The quantitative exhibits, computed where the parsed tables allow it.

    Called by the Analyst in Phase 3. Exhibits that cannot be computed are returned
    as gaps so the request is raised at the point the data was actually missing.
    """
    return [
        _build_one(spec, ctx) for spec in CATALOGUE if spec.key in COMPUTED_KEYS
    ]


def is_presentable(exhibit: Exhibit) -> bool:
    """Whether an exhibit may appear in the report.

    An exhibit earns its place only by having data *and* a source. A chart drawn from
    absent data is the most persuasive thing in a deck and the least defensible, and a
    table nobody can trace back to a document is an assertion wearing a table's
    clothes. Everything excluded here still leaves the engagement as a logged data
    request, so an omitted exhibit is visible as an outstanding gap in Section 8
    rather than silently dropped.
    """
    if exhibit.status is ExhibitStatus.GAP:
        return False
    if not (exhibit.rows or exhibit.series):
        return False
    return bool(exhibit.citations)


def build_for_section(
    section: int, ctx: ExhibitContext, precomputed: Optional[dict[str, Exhibit]] = None
) -> list[Exhibit]:
    """Every standard exhibit for one section, built and then filtered.

    `precomputed` supplies exhibits the Analyst already computed; the Synthesizer has
    no computation tool, so without them the quantitative ones can only be gaps.
    """
    return [e for e in build_all_for_section(section, ctx, precomputed)
            if is_presentable(e)]


def build_all_for_section(
    section: int, ctx: ExhibitContext, precomputed: Optional[dict[str, Exhibit]] = None
) -> list[Exhibit]:
    """Build every standard exhibit, presentable or not.

    The unfiltered list is what the gap log is written from - the report omits an
    unsourced exhibit, but the engagement still owes the data.
    """
    precomputed = precomputed or {}
    out: list[Exhibit] = []
    for spec in specs_for_section(section, ctx.shape):
        if spec.key in precomputed:
            out.append(precomputed[spec.key])
            continue
        if spec.builder not in _BUILDERS:
            continue
        out.append(_build_one(spec, ctx))
    return out
