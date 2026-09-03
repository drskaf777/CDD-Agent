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
