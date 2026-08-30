"""Render a Deck to reviewable output.

Markdown is the default because the draft is meant to be read and corrected before it
becomes slides - and because every claim's citation and tag stay visible inline, which
a polished deck tends to hide in speaker notes.
"""

from __future__ import annotations

from pathlib import Path

from cdd_agent.schemas.deck import Deck, Slide


def deck_to_markdown(deck: Deck) -> str:
    lines: list[str] = [
        f"# {deck.title}",
        "",
        f"> **{deck.draft_notice}**",
        "",
        f"_Generated {deck.created_at:%Y-%m-%d %H:%M UTC} by {deck.created_by}. "
        f"Groundedness: {deck.groundedness():.0%}._",
        "",
    ]
    for slide in deck.slides:
        lines += _slide_to_markdown(slide)
    return "\n".join(lines)


def _slide_to_markdown(slide: Slide) -> list[str]:
    lines = [
        f"## {slide.section_number}. {slide.section_title}",
        "",
        f"**{slide.so_what_headline}**",
        "",
    ]
    for claim in slide.claims:
        flag = " `management data only`" if claim.management_data_only else ""
        cites = "; ".join(c.short() for c in claim.citations)
        lines.append(f"- {claim.text} `[{claim.tag.value}]`{flag}")
        if cites:
            lines.append(f"  - Source: {cites}")
    if slide.claims:
        lines.append("")
    for exhibit in slide.exhibits:
        lines += _exhibit_to_markdown(exhibit)
    return lines


def _exhibit_to_markdown(exhibit) -> list[str]:  # type: ignore[no-untyped-def]
    lines = [f"**Exhibit - {exhibit.title}**", ""]
    if exhibit.columns:
        lines.append("| " + " | ".join(exhibit.columns) + " |")
        lines.append("|" + "|".join(["---"] * len(exhibit.columns)) + "|")
        for row in exhibit.rows:
            cells = [str(c).replace("|", "\\|") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
    if exhibit.note:
        lines += ["", f"_{exhibit.note}_"]
    lines.append("")
    return lines


def write_markdown(deck: Deck, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deck_to_markdown(deck), encoding="utf-8")
    return path
