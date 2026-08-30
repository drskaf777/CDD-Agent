"""Structured Computation tool - addresses the computation limitation (Checkpoint 2.1).

Numbers that will anchor an investment recommendation come from an actual calculation
over parsed data, never from a plausible-sounding completion. Every result carries the
source file and columns it was computed from, so the citation on a computed claim
points at real inputs rather than at the model's own arithmetic.

Operates on `StructuredTable` records - the schema'd side of ingestion. Structured data
is deliberately not embedded (Checkpoint 3.1 s 4): exact-match and aggregation beat
semantic similarity for numeric lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from cdd_agent.retrieval.ingestion import StructuredTable
from cdd_agent.schemas.common import Citation, SourceKind, Tier


class ComputationError(ValueError):
    """A computation could not be performed on the data supplied.

    Raised rather than returning an approximate answer: a silently degraded number is
    exactly the failure this tool exists to prevent.
    """


@dataclass
class ComputationResult:
    name: str
    value: Optional[float]
    table: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    citation: Optional[Citation] = None
    note: str = ""

    def render(self) -> str:
        head = f"{self.name}: {self.value:,.4g}" if self.value is not None else self.name
        return f"{head}{(' - ' + self.note) if self.note else ''}"


class StructuredComputationTool:
    """Cohort builds, revenue bridges, concentration schedules, sensitivity tables."""

    name = "structured_computation"
    description = (
        "Compute a metric from parsed data-room tables (cohort retention, ARR bridge, "
        "customer concentration, sensitivity grid, aggregate). Returns real arithmetic "
        "with the source file and columns used."
    )

    def __init__(self, tables: Sequence[StructuredTable]) -> None:
        self._tables = {t.name.lower(): t for t in tables}
        self._by_file = {t.source_file.lower(): t for t in tables}

    # ------------------------------------------------------------------ lookup
    def table(self, name: str) -> StructuredTable:
        key = name.lower()
        if key in self._tables:
            return self._tables[key]
        if key in self._by_file:
            return self._by_file[key]
        matches = [t for k, t in self._tables.items() if key in k]
        if len(matches) == 1:
            return matches[0]
        raise ComputationError(
            f"no structured table named {name!r}; available: {sorted(self._tables)}"
        )

    def frame(self, name: str) -> pd.DataFrame:
        table = self.table(name)
        df = pd.DataFrame(table.to_records())
        if df.empty:
            raise ComputationError(f"table {name!r} has no rows")
        return df

    def available(self) -> list[str]:
        return sorted(self._tables)

    def _cite(self, name: str, columns: Iterable[str], detail: str) -> Citation:
        table = self.table(name)
        return Citation(
            source_kind=SourceKind.COMPUTATION,
            source_file=table.source_file,
            locator=f"{detail} over columns {', '.join(columns)}",
            document_date=table.document_date,
            document_tier=Tier.DEAL_CRITICAL,
            quoted_text="",
        )

    # -------------------------------------------------------------- operations
    def aggregate(self, table: str, column: str, how: str = "sum") -> ComputationResult:
        df = self.frame(table)
        _require(df, [column], table)
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            raise ComputationError(f"column {column!r} in {table!r} holds no numeric values")
        value = float(getattr(series, how)())
        return ComputationResult(
            name=f"{how} of {column}",
            value=value,
            citation=self._cite(table, [column], f"{how} aggregate"),
        )

    def customer_concentration(
        self, table: str, customer_column: str, revenue_column: str, top_n: Sequence[int] = (5, 10, 20)
    ) -> ComputationResult:
        """Top-N revenue concentration - the first screen in the risk taxonomy."""
        df = self.frame(table)
        _require(df, [customer_column, revenue_column], table)
        df = df.copy()
        df[revenue_column] = pd.to_numeric(df[revenue_column], errors="coerce")
        grouped = (
            df.groupby(customer_column)[revenue_column].sum().sort_values(ascending=False)
        )
        total = float(grouped.sum())
        if total <= 0:
            raise ComputationError(f"total {revenue_column} in {table!r} is not positive")
        rows = [
            {
                "bucket": f"top {n}",
                "revenue": round(float(grouped.head(n).sum()), 2),
                "share_of_total": round(float(grouped.head(n).sum()) / total, 4),
            }
            for n in top_n
            if n <= len(grouped)
        ]
        top5_share = next((r["share_of_total"] for r in rows if r["bucket"] == "top 5"), None)
        return ComputationResult(
            name="customer concentration",
            value=top5_share,
            table=rows,
            columns=["bucket", "revenue", "share_of_total"],
            citation=self._cite(table, [customer_column, revenue_column], "concentration schedule"),
            note=f"{len(grouped)} customers, total {total:,.0f}",
        )

    def arr_bridge(
        self,
        table: str,
        *,
        period_column: str,
        new_column: str,
        expansion_column: str,
        contraction_column: str,
        churn_column: str,
        opening_column: Optional[str] = None,
    ) -> ComputationResult:
        """Contract-level ARR waterfall: gross new, expansion, contraction, churn.

        The spec calls this the single most-scrutinized exhibit in SaaS CDD and the
        fastest way to catch bookings-vs-billings issues, so it reconciles explicitly:
        the closing balance is computed, not taken from a summary row.
        """
        df = self.frame(table)
        cols = [period_column, new_column, expansion_column, contraction_column, churn_column]
        _require(df, cols, table)
        df = df.sort_values(period_column).copy()
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        opening = 0.0
        if opening_column and opening_column in df.columns:
            opening = float(pd.to_numeric(df[opening_column], errors="coerce").iloc[0] or 0.0)

        rows: list[dict[str, Any]] = []
        balance = opening
        for _, r in df.iterrows():
            # Contraction and churn are recorded as magnitudes; subtract their absolute value
            # so a file that stores them as negatives does not double-count the sign.
            delta = (
                float(r[new_column])
                + float(r[expansion_column])
                - abs(float(r[contraction_column]))
                - abs(float(r[churn_column]))
            )
            closing = balance + delta
            rows.append(
                {
                    "period": r[period_column],
                    "opening": round(balance, 2),
                    "new": round(float(r[new_column]), 2),
                    "expansion": round(float(r[expansion_column]), 2),
                    "contraction": round(-abs(float(r[contraction_column])), 2),
                    "churn": round(-abs(float(r[churn_column])), 2),
                    "closing": round(closing, 2),
                }
            )
            balance = closing

        return ComputationResult(
            name="ARR bridge",
            value=round(balance, 2),
            table=rows,
            columns=["period", "opening", "new", "expansion", "contraction", "churn", "closing"],
            citation=self._cite(table, cols, "ARR waterfall"),
            note=f"closing ARR {balance:,.0f} across {len(rows)} periods",
        )

    def retention_cohorts(
        self,
        table: str,
        *,
        cohort_column: str,
        period_column: str,
        value_column: str,
    ) -> ComputationResult:
        """Cohort retention as a share of each cohort's first observed value."""
        df = self.frame(table)
        _require(df, [cohort_column, period_column, value_column], table)
        df = df.copy()
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        rows: list[dict[str, Any]] = []
        for cohort, group in df.groupby(cohort_column):
            group = group.sort_values(period_column)
            base = float(group[value_column].iloc[0] or 0.0)
            if base <= 0:
                continue
            for _, r in group.iterrows():
                rows.append(
                    {
                        "cohort": cohort,
                        "period": r[period_column],
                        "value": round(float(r[value_column] or 0.0), 2),
                        "retention": round(float(r[value_column] or 0.0) / base, 4),
                    }
                )
        if not rows:
            raise ComputationError(f"no cohort in {table!r} has a positive base period")
        latest = [r for r in rows if r["period"] == max(x["period"] for x in rows)]
        avg_latest = sum(r["retention"] for r in latest) / len(latest) if latest else None
        return ComputationResult(
            name="cohort retention",
            value=round(avg_latest, 4) if avg_latest is not None else None,
            table=rows,
            columns=["cohort", "period", "value", "retention"],
            citation=self._cite(
                table, [cohort_column, period_column, value_column], "cohort build"
            ),
            note=f"{df[cohort_column].nunique()} cohorts",
        )

    def sensitivity(
        self,
        *,
        base_value: float,
        drivers: dict[str, Sequence[float]],
        source_table: Optional[str] = None,
    ) -> ComputationResult:
        """Multiplicative sensitivity grid over one or two drivers."""
        if not drivers or len(drivers) > 2:
            raise ComputationError("sensitivity takes one or two drivers")
        names = list(drivers)
        rows: list[dict[str, Any]] = []
        if len(names) == 1:
            a = names[0]
            for x in drivers[a]:
                rows.append({a: x, "value": round(base_value * (1 + x), 2)})
            columns = [a, "value"]
        else:
            a, b = names
            for x in drivers[a]:
                for y in drivers[b]:
                    rows.append(
                        {a: x, b: y, "value": round(base_value * (1 + x) * (1 + y), 2)}
                    )
            columns = [a, b, "value"]
        citation = (
            self._cite(source_table, names, "sensitivity grid") if source_table else None
        )
        return ComputationResult(
            name="sensitivity",
            value=base_value,
            table=rows,
            columns=columns,
            citation=citation,
            note=f"base {base_value:,.0f}",
        )


def _require(df: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ComputationError(
            f"table {table!r} is missing column(s) {missing}; has {list(df.columns)}"
        )
