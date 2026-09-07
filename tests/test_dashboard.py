"""The Executive Dashboard is a view, never a second synthesis path.

The risk this guards is specific: the dashboard is the page a partner is most likely
to read and least likely to check. If it could assemble its own exhibits it would be
the least verified surface in the system while looking like the most authoritative.
So it may only re-present what the draft already published under the data-and-citation
rule, and it must name what it is not showing.
"""

from __future__ import annotations

from cdd_agent.schemas.deck import Deck, Exhibit, ExhibitStatus, Slide
from cdd_agent.schemas.common import Citation, SourceKind
from cdd_agent.synthesis import dashboard as D


def _exhibit(key: str, title: str) -> Exhibit:
    return Exhibit(
        key=key, title=title, kind="table", status=ExhibitStatus.EVIDENCED,
        columns=["a"], rows=[["1"]],
        citations=[Citation(source_kind=SourceKind.PUBLIC_FILING,
                            source_file="FRSH_10-K.txt", locator="p1")])


def _deck(*exhibits: Exhibit, headline: str = "The headline from the draft.") -> Deck:
    return Deck(
        engagement_id="e", created_by="Synthesizer", title="CDD - Target",
        slides=[Slide(section_number=1, section_title="Executive Summary",
                      so_what_headline=headline, exhibits=list(exhibits))])


def test_the_dashboard_only_shows_what_the_draft_published():
    deck = _deck(_exhibit("market_growth", "Growth"), _exhibit("nps", "NPS"))
    d = D.build("e", deck, None, None, None, D.requests_by_key())
    shown = {e.key for s in d.sections for e in s.exhibits}
    assert shown == {"market_growth", "nps"}
    assert d.shown == 2


def test_an_exhibit_the_draft_withheld_is_named_not_drawn():
    """The whole point: absence has to be visible, and actionable."""
    d = D.build("e", _deck(_exhibit("market_growth", "Growth")), None, None, None,
                D.requests_by_key())
    market = next(s for s in d.sections if s.number == 2)
    assert [e.key for e in market.exhibits] == ["market_growth"]
    assert market.withheld, "the missing sizing exhibit must be named"
    assert any("bottom-up" in w for w in market.withheld)
    # Every withheld entry is a data request, not a blank placeholder.
    assert all(len(w) > 20 for s in d.sections for w in s.withheld)


def test_the_dashboard_does_not_write_its_own_headline():
    """An uncited conclusion on the most-read page is the worst possible output."""
    d = D.build("e", _deck(_exhibit("nps", "NPS"), headline="Verbatim from Section 1."),
                None, None, None, D.requests_by_key())
    assert d.headline == "Verbatim from Section 1."

    blank = D.build("e", _deck(_exhibit("nps", "NPS"), headline=""), None, None, None,
                    D.requests_by_key())
    assert blank.headline == "", "no headline is better than an invented one"


def test_no_deck_means_no_dashboard():
    d = D.build("e", None, None, None, None, D.requests_by_key())
    assert d.sections == [] and d.shown == 0


def test_every_layout_key_exists_in_the_catalogue():
    """A typo in the layout would silently drop a section of the analysis."""
    from cdd_agent.synthesis.exhibits import CATALOGUE

    known = {s.key for s in CATALOGUE}
    laid_out = {k for _, _, _, keys in D.LAYOUT for k in keys}
    assert laid_out <= known, f"layout references unknown exhibits: {laid_out - known}"


def test_the_layout_covers_the_standard_report_sections():
    numbers = [n for n, _, _, _ in D.LAYOUT]
    assert numbers == [1, 2, 3, 4, 5, 6]
    assert all(title.strip() and question.strip() for _, title, question, _ in D.LAYOUT)
