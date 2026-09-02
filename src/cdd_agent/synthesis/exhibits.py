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
from dataclasses import dataclass
from typing import Callable, Optional

from cdd_agent.knowledge.four_question_test import classify
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
)


def specs_for_section(section: int) -> list[ExhibitSpec]:
    return [s for s in CATALOGUE if s.section == section]


# --------------------------------------------------------------------- context
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?<=\.)\s(?=[A-Z])")

# A figure of the shape the exhibit's title promises. An exhibit headed "TAM / SAM /
# SOM" that shows a sentence with no magnitude in it is asserting, by placement, that
# a size was supplied.
MONEY = re.compile(
    r"[$\u00a3\u20ac]\s?\d[\d,.]*\s*(?:bn|b\b|billion|m\b|mm|million|k\b)?"
    r"|\b\d[\d,.]*\s*(?:bn|billion|million|trillion)\b",
    re.I,
)
PERCENT = re.compile(r"\d[\d,.]*\s?%|\bcagr\b", re.I)
FIGURE = re.compile(r"\d")

@dataclass
class ExhibitContext:
    tree: HypothesisTree
    matrix: EvidenceMatrix
    register: RiskRegister
    computation: Optional[StructuredComputationTool]
    strategic_buyer: bool = False

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
        """
        needles = [t.lower() for t in terms]
        out: list[tuple] = []
        seen: set[str] = set()
        for item in self.matrix.items:
            text = "\n".join(
                [item.claim] + [c.quoted_text for c in item.citations if c.quoted_text]
            )
            for sentence in _SENTENCE_SPLIT.split(text):
                sentence = " ".join(sentence.split())
                if not (12 <= len(sentence) <= 320):
                    continue
                # A quotable finding is a whole declarative sentence. Retrieval hands
                # back chunk-boundary fragments ("Now the platform vendors bundle")
                # and interview prompts ("Q: How do you evaluate vendors?"); quoting
                # either as a finding attributes to the source something it did not
                # assert.
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
                out.append((item, sentence))
                if len(out) >= limit:
                    return out
        return out


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
    ("Environmental", ("environmental", "energy", "sustainab")),
    ("Legal", ("regulat", "compliance", "licence", "license", "gdpr", "hipaa",
               "privacy", "litigation")),
)


def pestel(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    rows, citations = [], []
    for factor, markers in _PESTEL:
        hits = ctx.evidence_about(*markers)
        if hits:
            rows.append([factor, "evidenced",
                         " ".join(hits[0].claim.split())[:150],
                         hits[0].citations[0].short() if hits[0].citations else "-"])
            citations += hits[0].citations[:1]
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
    citations = [
        c for row in levers for item in ctx.matrix.for_hypothesis(row[0])
        for c in item.citations
    ]
    if not citations:
        return _gap(spec, "No evidence yet supports the levers the thesis depends on.")
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["Hypothesis", "Lever under test", "Evidence status", "Items"],
        rows=levers, citations=citations[:6],
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


def _risk_table(ctx: ExhibitContext, spec: ExhibitSpec,
                categories: tuple[RiskCategory, ...]) -> Exhibit:
    risks = [r for r in ctx.register.ranked() if r.category in categories]
    if not risks:
        return _gap(spec)
    # Carry through the citations of the evidence each risk was raised from. A risk
    # table with no traceable source is an assertion, and an exhibit that cannot be
    # sourced does not belong in the report at all.
    wanted = {eid for r in risks for eid in r.evidence_ids}
    citations = [
        c for item in ctx.matrix.items if item.id in wanted for c in item.citations
    ]
    if not citations:
        return _gap(spec, "The findings in this category carry no traceable source.")
    return Exhibit(
        title=spec.title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["ID", "Finding", "Sev", "Lik", "Score", "Flags"],
        rows=[[r.id, r.description, str(r.severity), str(r.likelihood), str(r.score),
               "management data only" if r.management_data_only else ""] for r in risks],
        citations=citations[:6],
        note="Drawn from the Risk Register, so it cannot disagree with Section 8.",
    )


def margin_erosion(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    return _risk_table(ctx, spec,
                       (RiskCategory.UNIT_ECONOMICS, RiskCategory.RETENTION))


def substitution(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    return _risk_table(ctx, spec, (RiskCategory.COMPETITIVE_DISRUPTION,))


def regulatory_heatmap(ctx: ExhibitContext, spec: ExhibitSpec) -> Exhibit:
    return _risk_table(ctx, spec, (RiskCategory.REGULATORY,))


_BUILDERS: dict[str, Callable[[ExhibitContext, ExhibitSpec], Exhibit]] = {
    "market_sizing": market_sizing, "market_growth": market_growth, "pestel": pestel,
    "landscape": landscape, "feature_bench": feature_bench, "market_share": market_share,
    "cohort_retention": cohort_retention, "nps": nps, "concentration": concentration,
    "win_loss": win_loss, "arr_bridge": arr_bridge,
    "forecast_vs_base": forecast_vs_base, "growth_levers": growth_levers,
    "pricing_elasticity": pricing_elasticity, "margin_erosion": margin_erosion,
    "substitution": substitution, "regulatory_heatmap": regulatory_heatmap,
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
    for spec in specs_for_section(section):
        if spec.key in precomputed:
            out.append(precomputed[spec.key])
            continue
        if spec.builder not in _BUILDERS:
            continue
        out.append(_build_one(spec, ctx))
    return out
