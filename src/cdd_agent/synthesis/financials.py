"""Financial time series read out of filing prose.

A listed target publishes its numbers in sentences, not spreadsheets: "total revenue
was $838.8 million, $720.4 million and $596.4 million in the years ended December 31,
2025, 2024 and 2023". Every figure a revenue chart needs is in there, and until now
the system could not plot it because nothing had parsed a CSV.

The extraction is deliberately deterministic rather than model-driven. A model asked
to pull figures out of a filing will occasionally produce a plausible number that is
not in the text, and a chart is exactly where that is least detectable - nobody
re-reads a filing to check a bar. So this matches known reporting phrasings and takes
the values straight out of the matched span. A figure that cannot be read out of the
source verbatim does not become a data point, which means the chart can be thin but
never invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Money as filings write it: $838.8 million, $1.2 billion, $(18.2) million for losses.
_MONEY = r"\$\s?\(?\d[\d,]*(?:\.\d+)?\)?\s*(?:million|billion|bn|m\b)?"
_YEAR = r"(?:19|20)\d{2}"

# The metrics worth charting, and the words a filing uses for each. Order matters:
# the first match wins, so the more specific phrasings come first.
METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Revenue", ("total revenue", "revenue was", "revenues were")),
    ("Gross profit", ("gross profit",)),
    ("Non-GAAP operating income", ("non-gaap income from operations",
                                   "non-gaap operating income")),
    ("Operating income / (loss)", ("loss from operations", "income from operations")),
    ("Free cash flow", ("free cash flow",)),
    ("Net income / (loss)", ("net loss was", "net income was")),
)


# A filing states annual and quarterly figures in the same voice: "revenue was
# $838.8 million in the year ended December 31, 2025" and "revenue was $228.6 million
# for the three months ended March 31, 2026". Both parse identically, and putting the
# quarter on an annual chart understates the year by three quarters while looking
# entirely normal. So the period basis has to be read from the sentence, and a
# sentence that does not say which basis it is on yields nothing.
_ANNUAL = ("year ended", "years ended", "fiscal year", "full year", "annual period")
_QUARTERLY = ("three months ended", "quarter ended", "quarterly period",
              "six months ended", "nine months ended", "months ended")


def _basis(sentence: str) -> str:
    low = sentence.lower()
    # Quarterly first: "three months ended December 31, 2025" also contains no
    # annual cue, but "year ended" can appear in the same sentence as a comparative.
    if any(cue in low for cue in _QUARTERLY):
        return "quarterly"
    if any(cue in low for cue in _ANNUAL):
        return "annual"
    return "unstated"


@dataclass(frozen=True)
class Point:
    """One figure, and the words it was read out of."""
    metric: str
    period: str
    value: float          # normalised to millions of the reporting currency
    unit: str
    quoted: str
    source_file: str
    locator: str
    basis: str = "annual"


def _to_millions(raw: str) -> Optional[float]:
    text = raw.lower().replace("$", "").replace(",", "").strip()
    negative = "(" in text
    text = text.replace("(", "").replace(")", "").strip()
    scale = 1.0
    for suffix, factor in (("billion", 1000.0), ("bn", 1000.0),
                           ("million", 1.0), ("m", 1.0)):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            scale = factor
            break
    try:
        value = float(text) * scale
    except ValueError:
        return None
    return -value if negative else value


def _sentences(text: str) -> list[str]:
    # Filing figures carry decimal points, so a naive split on "." shears them in
    # half. Split only where a period is followed by whitespace and a capital.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()]


def extract(text: str, *, source_file: str, locator: str) -> list[Point]:
    """Every figure this text states plainly enough to plot."""
    points: list[Point] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _sentences(text):
        low = sentence.lower()
        metric = next((name for name, cues in METRICS if any(c in low for c in cues)), None)
        if metric is None:
            continue
        basis = _basis(sentence)
        if basis != "annual":
            # A quarterly or unlabelled figure is not wrong, it is just not a year.
            continue
        values = re.findall(_MONEY, sentence)
        years = re.findall(_YEAR, sentence)
        if not values or not years:
            continue
        # "A, B and C in the years ended December 31, 2025, 2024 and 2023" - the
        # filing lists figures newest-first and the years in the same order, so the
        # pairing is positional. Any other shape is left alone rather than guessed at.
        if len(values) != len(years):
            if len(values) == 1 and len(years) >= 1:
                pairs = [(values[0], years[0])]
            else:
                continue
        else:
            pairs = list(zip(values, years, strict=True))
        for raw, year in pairs:
            amount = _to_millions(raw)
            if amount is None:
                continue
            key = (metric, year)
            if key in seen:
                continue
            seen.add(key)
            points.append(Point(metric=metric, period=f"FY{year}", value=amount,
                                unit="m", quoted=" ".join(sentence.split())[:300],
                                source_file=source_file, locator=locator))
    return points


def series_for(points: list[Point], metric: str) -> list[Point]:
    """One metric, oldest period first, one figure per period."""
    chosen: dict[str, Point] = {}
    for p in points:
        if p.metric == metric:
            chosen.setdefault(p.period, p)
    return sorted(chosen.values(), key=lambda p: p.period)


def margin_series(points: list[Point], numerator: str) -> list[tuple[str, float]]:
    """A margin only where both halves were stated for the same period.

    Never carried forward from an adjacent year and never interpolated: a margin is
    the quotient of two figures that were actually reported together, or it is absent.
    """
    revenue = {p.period: p.value for p in series_for(points, "Revenue") if p.value}
    top = {p.period: p.value for p in series_for(points, numerator)}
    return [(period, top[period] / revenue[period])
            for period in sorted(set(revenue) & set(top))]
