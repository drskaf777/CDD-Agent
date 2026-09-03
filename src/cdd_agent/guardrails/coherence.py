"""Detective guardrail: do this engagement artifacts agree on which company it is?

Every artifact in an engagement is derived from the one before it - the hypothesis
tree from the Deal Profile Brief, the evidence from the tree, the deck from all
three. Nothing re-checks that chain once it is built, so replacing an upstream
artifact leaves the downstream ones intact, plausible, and about a different company.

That is not hypothetical. A control labelled "Reload profile" loaded a demo fixture
over a Deal Profile Brief built from real filings. Every count on the dashboard stayed
right, the deck still rendered, and the next Phase-1 run decomposed the wrong target
against the right evidence. The visible symptom was that it felt slow.

The checks here are deliberately structural rather than clever. Each one compares a
value one artifact *copied* from another at the moment it was built, so a divergence
means an upstream artifact was replaced after the fact - there is no other way for
them to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Incoherence:
    """One disagreement between artifacts, and what closes it."""

    artifact: str
    detail: str
    remedy: str

    def render(self) -> str:
        return f"[{self.artifact}] {self.detail} {self.remedy}"

    def to_dict(self) -> dict[str, str]:
        return {"artifact": self.artifact, "detail": self.detail, "remedy": self.remedy}


def _norm(text: str) -> str:
    """Collapse the differences that are not disagreements.

    A trailing full stop is not a different thesis. The first version of this check
    fired on exactly that - the tree had "...installed base." and the brief had
    "...installed base" - and a guardrail that cries wolf over punctuation teaches
    people to click past it, which costs more than it saves.
    """
    return " ".join((text or "").split()).casefold().rstrip(".;:,!? ")


def _divergence(a: str, b: str, window: int = 45) -> str:
    """Show where two strings part company, not their first ninety characters.

    Reporting the head of each string is useless when they agree for a paragraph and
    differ at the end - the report then prints two identical-looking quotes and
    invites the reader to conclude the checker is broken.
    """
    a_norm, b_norm = " ".join(a.split()), " ".join(b.split())
    limit = min(len(a_norm), len(b_norm))
    at = next((i for i in range(limit) if a_norm[i] != b_norm[i]), limit)
    start = max(0, at - window // 2)
    lead = "..." if start else ""
    return (f"they agree for {at} characters, then differ: "
            f"{lead}{a_norm[start:at + window]!r} vs {lead}{b_norm[start:at + window]!r}")


def check_engagement(
    engagement_id: str,
    profile: Optional[Any] = None,
    tree: Optional[Any] = None,
    deck: Optional[Any] = None,
    matrix: Optional[Any] = None,
    register: Optional[Any] = None,
    table_files: Optional[list[str]] = None,
    ingested_files: Optional[list[str]] = None,
) -> list[Incoherence]:
    """Every way the artifacts can disagree about the target, in one pass."""
    found: list[Incoherence] = []

    # 1. The tree copies the thesis verbatim when Phase 1 runs. If they differ now,
    #    the profile was replaced or edited after the decomposition was built, and the
    #    tree is testing a thesis nobody is pursuing.
    if profile is not None and tree is not None:
        stated = _norm(profile.thesis.one_sentence_thesis)
        decomposed = _norm(tree.root_thesis)
        if stated and decomposed and stated != decomposed:
            found.append(Incoherence(
                artifact="hypothesis tree",
                detail=(
                    "was decomposed from a different thesis than the Deal Profile "
                    "Brief now states - "
                    + _divergence(tree.root_thesis,
                                  profile.thesis.one_sentence_thesis)
                ),
                remedy=(
                    "Re-run Phase 1 against the current brief, or restore the brief "
                    "the tree was built from. Do not advance: every data request and "
                    "evidence rating below this point tests the wrong claim."
                ),
            ))

    # 2. The deck names the target in its title. A deck about another company is the
    #    single most damaging artifact this system can emit, because it looks finished.
    if profile is not None and deck is not None:
        target = _norm(profile.target.legal_name)
        if target and target not in _norm(deck.title):
            found.append(Incoherence(
                artifact="deck",
                detail=(
                    f"is titled {deck.title!r}, which does not name "
                    f"{profile.target.legal_name!r}."
                ),
                remedy="Re-run synthesis so the draft is written for the current target.",
            ))

    # 3. Each artifact records the engagement it belongs to. A mismatch means one was
    #    written into the wrong engagement entirely.
    for name, artifact in (("deal profile", profile), ("hypothesis tree", tree),
                           ("evidence matrix", matrix), ("risk register", register),
                           ("deck", deck)):
        owner = getattr(artifact, "engagement_id", None)
        if owner and owner != engagement_id:
            found.append(Incoherence(
                artifact=name,
                detail=f"belongs to engagement {owner!r}, not {engagement_id!r}.",
                remedy="It was written into the wrong engagement; do not rely on it.",
            ))
    # 4. Structured tables are not stamped with an engagement, so nothing above
    #    catches them being loaded from the wrong folder. What does catch it is the
    #    filenames: a table parsed for this engagement should have arrived with its
    #    data room. This fired for real - a customer schedule from another engagement
    #    was computed into a deck and rendered as that company concentration risk.
    if ingested_files is not None and table_files:
        strays = sorted(set(table_files) - set(ingested_files))
        if strays:
            found.append(Incoherence(
                artifact="structured tables",
                detail=(
                    f"include {', '.join(strays[:4])}, which "
                    f"{'was' if len(strays) == 1 else 'were'} never ingested for "
                    f"{engagement_id!r}. Any computed exhibit built on "
                    f"{'it' if len(strays) == 1 else 'them'} describes another "
                    f"engagement data."
                ),
                remedy=(
                    "Re-ingest this engagement own data room and re-run Phase 3, so "
                    "the computed exhibits are rebuilt or correctly dropped."
                ),
            ))
    # 5. Evidence accumulates across Phase-3 loops by design, so re-ingesting a
    #    corrected data room does not retire what the previous one produced. Items
    #    citing a document this engagement no longer holds are the residue, and they
    #    are indistinguishable from good evidence once they reach the deck - one item
    #    was found citing two companies documents at once.
    if ingested_files is not None and matrix is not None:
        held = set(ingested_files)
        stale: set[str] = set()
        for item in getattr(matrix, "items", []):
            for citation in getattr(item, "citations", []):
                source = getattr(citation, "source_file", "")
                kind = getattr(citation, "source_kind", None)
                # Knowledge-Base references are cross-engagement by design and never
                # appear in a data-room ingestion.
                if kind is not None and getattr(kind, "is_public_record", False) \
                        and source.startswith("kb_"):
                    continue
                if source and source not in held and not source.startswith("kb_") \
                        and "intake" not in source.lower():
                    stale.add(source)
        if stale:
            found.append(Incoherence(
                artifact="evidence matrix",
                detail=(
                    f"cites {', '.join(sorted(stale)[:4])}, which this engagement has "
                    f"not ingested. Evidence accumulates across loops, so these "
                    f"survived a data-room correction."
                ),
                remedy=(
                    "Clear the evidence matrix and re-run Phase 3 so every rating is "
                    "grounded in documents this engagement actually holds."
                ),
            ))
    return found


def raise_if_incoherent(engagement_id: str, **artifacts: Any) -> None:
    """Refuse to proceed while the artifacts disagree about the target."""
    problems = check_engagement(engagement_id, **artifacts)
    if problems:
        raise EngagementIncoherent(
            "This engagement artifacts disagree about which company is being "
            "diligenced:\n" + "\n".join("  - " + p.render() for p in problems)
        )


class EngagementIncoherent(RuntimeError):
    """Raised when downstream work would be built on a replaced upstream artifact."""
