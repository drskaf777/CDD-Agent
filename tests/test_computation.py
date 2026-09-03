"""Structured computation: real arithmetic, or an error - never an approximation."""

from __future__ import annotations

import pytest

from cdd_agent.retrieval.ingestion import StructuredTable, parse_number
from cdd_agent.tools.structured_computation import (
    ComputationError,
    StructuredComputationTool,
)


def _arr_table() -> StructuredTable:
    rows = [
        {"period": "2026-Q1", "opening_arr": "1000", "new": "100", "expansion": "50",
         "contraction": "20", "churn": "30"},
        {"period": "2026-Q2", "opening_arr": "1100", "new": "80", "expansion": "40",
         "contraction": "25", "churn": "35"},
    ]
    return StructuredTable(
        source_file="arr.csv", name="arr", columns=list(rows[0]), rows=rows
    )


def _revenue_table() -> StructuredTable:
    rows = [
        {"customer": f"C{i}", "arr": str(1000 - i * 50)} for i in range(20)
    ]
    return StructuredTable(
        source_file="rev.csv", name="rev", columns=["customer", "arr"], rows=rows
    )


def test_arr_bridge_reconciles_rather_than_reading_a_summary_row():
    tool = StructuredComputationTool([_arr_table()])
    result = tool.arr_bridge(
        "arr",
        period_column="period",
        new_column="new",
        expansion_column="expansion",
        contraction_column="contraction",
        churn_column="churn",
        opening_column="opening_arr",
    )
    # 1000 + 100 + 50 - 20 - 30 = 1100; 1100 + 80 + 40 - 25 - 35 = 1160
    assert result.table[0]["closing"] == 1100
    assert result.value == 1160
    assert result.citation.source_file == "arr.csv"


def test_contraction_stored_as_a_negative_is_not_double_counted():
    table = _arr_table()
    table.rows[0]["contraction"] = "-20"
    table.rows[0]["churn"] = "-30"
    tool = StructuredComputationTool([table])
    result = tool.arr_bridge(
        "arr", period_column="period", new_column="new", expansion_column="expansion",
        contraction_column="contraction", churn_column="churn", opening_column="opening_arr",
    )
    assert result.table[0]["closing"] == 1100


def test_customer_concentration_is_a_share_of_company_revenue():
    """Reported from a live deck: "top 20 = 100% of revenue".

    True of the customer file, false of the company. The denominator has to be total
    company revenue, or the top-5 figure understates concentration by however much
    revenue sits outside the listed accounts.
    """
    tool = StructuredComputationTool([_revenue_table()])
    listed = sum(1000 - i * 50 for i in range(20))
    top5 = sum(1000 - i * 50 for i in range(5))
    company = listed * 4  # the file lists a quarter of the business

    result = tool.customer_concentration("rev", "customer", "arr",
                                         total_revenue=company)
    assert result.value == pytest.approx(top5 / company, rel=1e-6)
    shares = {r["bucket"]: r["share_of_total"] for r in result.table}
    assert shares["top 20"] == pytest.approx(listed / company, rel=1e-6)
    assert shares["top 20"] < 0.30, "top 20 must not read as 100% of the company"
    assert "of company revenue" in result.note


def test_concentration_refuses_to_guess_the_denominator():
    """Summing the customer file to get company revenue is the tempting shortcut,
    and it is the bug. Without a stated total this must raise, so the exhibit
    becomes a request rather than a chart against the wrong base."""
    tool = StructuredComputationTool([_revenue_table()])
    with pytest.raises(ComputationError, match="total company revenue"):
        tool.customer_concentration("rev", "customer", "arr")


def test_customers_exceeding_company_revenue_is_refused():
    tool = StructuredComputationTool([_revenue_table()])
    with pytest.raises(ComputationError, match="exceeds the stated"):
        tool.customer_concentration("rev", "customer", "arr", total_revenue=100.0)


def test_total_revenue_is_read_not_summed():
    """It must never manufacture a total by adding up the customer schedule."""
    tool = StructuredComputationTool([_revenue_table()])
    assert tool.total_company_revenue() is None


def test_cohort_retention_is_relative_to_each_cohort_base():
    rows = [
        {"cohort": "2024", "period": "0", "net_arr": "100"},
        {"cohort": "2024", "period": "1", "net_arr": "120"},
        {"cohort": "2025", "period": "0", "net_arr": "200"},
        {"cohort": "2025", "period": "1", "net_arr": "180"},
    ]
    table = StructuredTable(
        source_file="cohorts.csv", name="cohorts", columns=list(rows[0]), rows=rows
    )
    tool = StructuredComputationTool([table])
    result = tool.retention_cohorts(
        "cohorts", cohort_column="cohort", period_column="period", value_column="net_arr"
    )
    latest = {r["cohort"]: r["retention"] for r in result.table if r["period"] == "1"}
    assert set(latest) == {"2024", "2025"}, "cohort labels must survive as labels"
    assert latest["2024"] == pytest.approx(1.2)
    assert latest["2025"] == pytest.approx(0.9)


def test_missing_column_raises_rather_than_guessing():
    tool = StructuredComputationTool([_revenue_table()])
    with pytest.raises(ComputationError, match="missing column"):
        tool.customer_concentration("rev", "customer", "revenue_usd")


def test_unknown_table_raises():
    tool = StructuredComputationTool([_revenue_table()])
    with pytest.raises(ComputationError, match="no structured table"):
        tool.aggregate("cohorts", "arr")


def test_currency_and_percent_strings_are_parsed():
    assert parse_number("78%") == pytest.approx(0.78)
    assert parse_number("$1,200") == pytest.approx(1200.0)
    assert parse_number("(450)") == pytest.approx(-450.0)
    assert parse_number("not a number") == "not a number"


def test_label_columns_are_not_coerced_to_numbers():
    """A cohort of "2024" is a label. Coercing it to 2024.0 breaks its own grouping."""
    rows = [{"cohort": "2024", "net_arr": "100"}]
    table = StructuredTable(
        source_file="c.csv", name="c", columns=["cohort", "net_arr"], rows=rows
    )
    assert table.to_records()[0]["cohort"] == "2024"


def test_sensitivity_grid_covers_both_drivers():
    tool = StructuredComputationTool([])
    result = tool.sensitivity(
        base_value=1000.0, drivers={"growth": [-0.1, 0.0], "margin": [-0.05, 0.05]}
    )
    assert len(result.table) == 4
    assert result.table[0]["value"] == pytest.approx(1000 * 0.9 * 0.95)
