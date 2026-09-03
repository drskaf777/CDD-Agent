"""The standard CDD exhibit set.

The point of these tests is the discipline, not the layout: an exhibit is computed,
evidenced, or declared missing. A market-sizing waterfall with numbers nobody supplied
would be the most convincing thing in the deck and the most dangerous, so the tests
that matter are the ones asserting what the builders *refuse* to draw.
"""

from __future__ import annotations

import pytest

from cdd_agent.retrieval.ingestion import StructuredTable
from cdd_agent.schemas.common import Citation, ConfidenceTag, SourceKind
from cdd_agent.schemas.deck import Exhibit, ExhibitStatus
from cdd_agent.schemas.evidence import EvidenceItem, EvidenceMatrix
from cdd_agent.schemas.risk import RiskRegister
from cdd_agent.synthesis.exhibits import (
    CATALOGUE,
    ExhibitContext,
    build_all_for_section,
    build_for_section,
    is_presentable,
)
from cdd_agent.tools.structured_computation import StructuredComputationTool


def _ctx(tree, *, tables=(), matrix=None, register=None) -> ExhibitContext:
    return ExhibitContext(
        tree=tree,
        matrix=matrix or EvidenceMatrix(engagement_id="e", created_by="t"),
        register=register or RiskRegister(engagement_id="e", created_by="t"),
        computation=StructuredComputationTool(tables) if tables else None,
    )


def test_catalogue_covers_the_standard_report_sections():
    sections = {s.section for s in CATALOGUE}
    # Market, competitive, customer, financial, valuation, risk.
    assert {2, 3, 4, 6, 7, 8} <= sections
    assert len(CATALOGUE) >= 15
    assert len({s.key for s in CATALOGUE}) == len(CATALOGUE), "duplicate exhibit keys"
    # Every exhibit names the data that would build it, so a gap is always actionable.
    assert all(s.requires.strip() for s in CATALOGUE)


def test_market_sizing_is_never_invented(tree):
    """The single most quoted exhibit, and the easiest to fabricate."""
    sizing = next(e for e in build_all_for_section(2, _ctx(tree)) if "TAM" in e.title)
    assert sizing.status is ExhibitStatus.GAP
    assert not sizing.rows, "a gap must carry no data rows"
    assert not sizing.series, "a gap must carry no chart series"
    assert "bottom-up" in sizing.gap_request
    # And it is left out of the report entirely rather than shown as a placeholder.
    assert not any("TAM" in e.title for e in build_for_section(2, _ctx(tree)))


def test_every_unbuildable_exhibit_states_what_would_close_it(tree):
    for section in (2, 3, 4, 6, 7, 8):
        for exhibit in build_all_for_section(section, _ctx(tree)):
            if exhibit.status is ExhibitStatus.GAP:
                assert exhibit.gap_request, f"{exhibit.title} is a silent gap"
                assert not exhibit.rows and not exhibit.series


def test_nothing_unsourced_reaches_the_report(tree):
    """Data and a citation, or it does not appear. No exceptions."""
    for section in (2, 3, 4, 6, 7, 8):
        for exhibit in build_for_section(section, _ctx(tree)):
            assert exhibit.status is not ExhibitStatus.GAP
            assert exhibit.rows or exhibit.series, f"{exhibit.title} has no data"
            assert exhibit.citations, f"{exhibit.title} cannot be sourced"


def test_an_exhibit_with_data_but_no_citation_is_still_excluded():
    """A table nobody can trace to a document is an assertion in a table's clothes."""
    unsourced = Exhibit(title="Made up", kind="table", status=ExhibitStatus.EVIDENCED,
                        columns=["a"], rows=[["1"]])
    assert not is_presentable(unsourced)


def test_concentration_is_computed_against_company_revenue(tree):
    rows = [{"customer": f"C{i}", "arr": str(1000 - i * 50)} for i in range(20)]
    customers = StructuredTable(source_file="customer_revenue.csv",
                                name="customer_revenue",
                                columns=["customer", "arr"], rows=rows)
    listed = sum(1000 - i * 50 for i in range(20))
    company = listed * 4
    totals = StructuredTable(source_file="revenue_summary.csv", name="revenue_summary",
                             columns=["total_revenue"],
                             rows=[{"total_revenue": str(company)}])
    exhibits = build_for_section(4, _ctx(tree, tables=[customers, totals]))
    conc = next(e for e in exhibits if "concentration" in e.title.lower())
    assert conc.status is ExhibitStatus.COMPUTED
    top5 = sum(1000 - i * 50 for i in range(5))
    assert conc.series[0].values[0] == pytest.approx(top5 / company, rel=1e-6)
    # A computed exhibit cites the file and columns it was computed from.
    assert conc.citations and conc.citations[0].source_file == "customer_revenue.csv"


def test_concentration_is_omitted_when_company_revenue_is_unknown(tree):
    """A customer schedule alone cannot say what share of the company it is."""
    rows = [{"customer": f"C{i}", "arr": str(1000 - i * 50)} for i in range(20)]
    customers = StructuredTable(source_file="customer_revenue.csv",
                                name="customer_revenue",
                                columns=["customer", "arr"], rows=rows)
    built = build_for_section(4, _ctx(tree, tables=[customers]))
    assert not any("concentration" in e.title.lower() for e in built)


def test_market_share_reports_hhi_when_competitor_data_exists(tree):
    rows = [{"competitor": n, "revenue": v} for n, v in
            [("A", "4000"), ("B", "3000"), ("C", "2000"), ("D", "1000")]]
    table = StructuredTable(source_file="competitor_share.csv", name="competitor_share",
                            columns=["competitor", "revenue"], rows=rows)
    exhibits = build_for_section(3, _ctx(tree, tables=[table]))
    share = next(e for e in exhibits if "HHI" in e.title)
    assert share.status is ExhibitStatus.COMPUTED
    # 40^2 + 30^2 + 20^2 + 10^2 = 3000
    assert "3,000" in share.note or "3000" in share.note
    assert "concentrated" in share.note


def test_cohort_curves_are_indexed_to_each_cohort_base(tree):
    rows = [
        {"cohort": "2024", "period": "0", "net_arr": "100"},
        {"cohort": "2024", "period": "1", "net_arr": "120"},
        {"cohort": "2025", "period": "0", "net_arr": "200"},
        {"cohort": "2025", "period": "1", "net_arr": "180"},
    ]
    table = StructuredTable(source_file="cohorts.csv", name="cohort_retention",
                            columns=list(rows[0]), rows=rows)
    exhibits = build_for_section(4, _ctx(tree, tables=[table]))
    cohorts = next(e for e in exhibits if "cohort" in e.title.lower())
    assert cohorts.status is ExhibitStatus.COMPUTED
    assert {s.name for s in cohorts.series} == {"2024", "2025"}
    by_name = {s.name: s.values for s in cohorts.series}
    assert by_name["2024"][1] == pytest.approx(1.2)
    assert by_name["2025"][1] == pytest.approx(0.9)


def test_risk_exhibits_cannot_disagree_with_the_register(tree):
    """Section 8 exhibits are drawn from the register, not restated beside it."""
    from cdd_agent.schemas.risk import RiskCategory, RiskItem

    register = RiskRegister(engagement_id="e", created_by="t", risks=[
        RiskItem(id="RISK-001", engagement_id="e", created_by="Risk Auditor",
                 category=RiskCategory.COMPETITIVE_DISRUPTION,
                 description="Platform vendors bundling workload protection",
                 severity=4, likelihood=3),
    ])
    # A risk with no traceable evidence cannot be shown, however real it is.
    unsourced = build_for_section(8, _ctx(tree, register=register))
    assert not any("substitution" in e.title.lower() for e in unsourced)

    # Give the risk a source, and the exhibit earns its place - citing that source.
    matrix = EvidenceMatrix(engagement_id="e", created_by="t")
    matrix.add(EvidenceItem(
        id="EV-1", engagement_id="e", created_by="Analyst", hypothesis_id="H2",
        claim="Platform vendors bundle workload protection", tag=ConfidenceTag.PARTIALLY_CONFIRMED,
        source_kind=SourceKind.DATA_ROOM,
        citations=[Citation(source_kind=SourceKind.DATA_ROOM,
                            source_file="Expert_Call.txt", locator="Q&A turn 1")]))
    register.risks[0].evidence_ids = ["EV-1"]
    subst = next(e for e in build_for_section(8, _ctx(tree, register=register, matrix=matrix))
                 if "substitution" in e.title.lower())
    assert subst.status is ExhibitStatus.EVIDENCED
    assert any("Platform vendors" in cell for row in subst.rows for cell in row)
    assert subst.citations[0].source_file == "Expert_Call.txt"


def test_a_failing_builder_degrades_to_a_gap_not_a_crash(tree, monkeypatch):
    import cdd_agent.synthesis.exhibits as ex

    def boom(ctx, spec):
        raise RuntimeError("builder exploded")

    monkeypatch.setitem(ex._BUILDERS, "concentration", boom)
    conc = next(e for e in build_all_for_section(4, _ctx(tree))
                if "concentration" in e.title.lower())
    assert conc.status is ExhibitStatus.GAP
    assert conc.gap_request, "a failed build still owes the data request"
    assert not any("concentration" in e.title.lower()
                   for e in build_for_section(4, _ctx(tree)))


def test_gap_exhibits_become_dated_information_gaps(context, tree):
    """The deck and Section 8 must agree about what is still owed."""
    import datetime as _dt

    from cdd_agent.agents.synthesizer import Synthesizer
    from cdd_agent.schemas.risk import GapOwner, InformationGap

    matrix = context.memory.evidence_matrix()
    register = context.memory.risk_register()
    # The synthesis gate refuses an unevidenced Tier-1 hypothesis without a dated gap,
    # which is the guardrail working. Satisfy it so this test can reach the exhibits.
    for h in tree.tier_1():
        register.gaps.append(InformationGap(
            id=f"GAP-{h.id}", engagement_id="e", created_by="test",
            hypothesis_id=h.id, request=f"Evidence for {h.id}",
            owner=GapOwner.MANAGEMENT,
            target_close_date=_dt.date.today() + _dt.timedelta(days=7),
            blocking=True))
    deck, _ = Synthesizer(context).run(tree, matrix, register)

    # Nothing unsourced is rendered ...
    assert not any(e.status is ExhibitStatus.GAP
                   for s in deck.slides for e in s.exhibits)
    # ... but every omitted exhibit is still owed, as a dated request.
    from cdd_agent.synthesis.exhibits import CATALOGUE as CAT
    gap_titles = [s.requires for s in CAT if s.key in
                  {"tam_sam_som", "landscape", "nps", "win_loss"}]

    saved = context.memory.risk_register()
    logged = {g.request for g in saved.gaps}
    for request in gap_titles:
        assert request in logged, f"exhibit gap never reached the register: {request[:60]}"
    for gap in saved.gaps:
        if gap.request in gap_titles:
            assert gap.target_close_date is not None, "an undated gap is not actionable"


def _matrix_with(claim: str, quoted: str = "") -> EvidenceMatrix:
    m = EvidenceMatrix(engagement_id="e", created_by="t")
    m.add(EvidenceItem(
        id="EV-1", engagement_id="e", created_by="Analyst", hypothesis_id="H1",
        claim=claim, tag=ConfidenceTag.PARTIALLY_CONFIRMED,
        source_kind=SourceKind.DATA_ROOM,
        citations=[Citation(source_kind=SourceKind.DATA_ROOM, source_file="Deck.txt",
                            locator="slide 3", quoted_text=quoted)]))
    return m


def test_discussing_the_market_is_not_the_same_as_sizing_it(tree):
    """The failure this guards is subtle: every cell cited, and still misleading.

    An exhibit headed TAM / SAM / SOM asserts by its title that a size was supplied.
    Filling it with a cited sentence that merely mentions the market lets the reader
    infer a number nobody gave.
    """
    discussed = _matrix_with("The addressable market is expanding as workloads move.")
    assert not any("TAM" in e.title
                   for e in build_for_section(2, _ctx(tree, matrix=discussed)))

    sized = _matrix_with("Management sizes the addressable market at $6.1bn.")
    sizing = next(e for e in build_for_section(2, _ctx(tree, matrix=sized))
                  if "TAM" in e.title)
    assert sizing.rows[0][1] == "Management sizes the addressable market at $6.1bn."
    assert sizing.rows[0][0] == "TAM", "the layer is read from the sentence, not assumed"


def test_growth_exhibit_requires_a_rate_not_a_mood(tree):
    vibes = _matrix_with("The business grew strongly across every region.")
    assert not any("CAGR" in e.title
                   for e in build_for_section(2, _ctx(tree, matrix=vibes)))
    rate = _matrix_with("Revenue grew at a 34% CAGR over the period.")
    assert any("CAGR" in e.title for e in build_for_section(2, _ctx(tree, matrix=rate)))


def test_fragments_and_interview_prompts_are_never_quoted_as_findings(tree):
    """Retrieval returns chunk-boundary fragments and questions; neither is a finding."""
    noise = _matrix_with(
        "Q: How do you evaluate competitive vendors today? "
        "Now the competitive vendors bundle"
    )
    for exhibit in build_for_section(3, _ctx(tree, matrix=noise)):
        for row in exhibit.rows:
            assert "?" not in str(row[0]), "an interview prompt asserts nothing"
            assert str(row[0]).endswith((".", "!")), f"fragment quoted: {row[0]!r}"


def test_the_agents_own_bookkeeping_is_never_quoted_as_a_source(tree):
    """The Analyst wraps passages in "Retrieved evidence bearing on H1: ...".

    Quoted verbatim, that scaffolding reads as though the board deck asserted it.
    """
    m = _matrix_with(
        "Retrieved evidence bearing on GROWTH-H1: The company grew at a 21% CAGR."
    )
    growth = next(e for e in build_for_section(2, _ctx(tree, matrix=m))
                  if "CAGR" in e.title)
    assert growth.rows[0][0] == "The company grew at a 21% CAGR."
    assert not any("Retrieved evidence" in str(c) for r in growth.rows for c in r)
