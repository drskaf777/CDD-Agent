"""Short-term (working) and long-term memory.

Checkpoint 2.1 draws the line by lifetime, not by storage technology:

* Working memory holds what a single reasoning step needs in view - the hypothesis
  being tested, the chunks just retrieved, the last observation. It is deliberately
  ephemeral, so a stale retrieval cannot leak into a later step unnoticed.
* Long-term memory holds what must survive the multi-week engagement, and (for
  corrections) what must survive the engagement itself so judgment compounds across
  deals rather than resetting each time.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from cdd_agent.schemas.common import Citation
from cdd_agent.schemas.data_request import DataRequestChecklist
from cdd_agent.schemas.deal_profile import DealProfile
from cdd_agent.schemas.deck import Deck
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree, ThesisSearchResult
from cdd_agent.schemas.risk import RiskRegister
from cdd_agent.state.store import Collection, StateStore


@dataclass
class WorkingMemory:
    """Scratch state for one ReAct step. Cleared between hypotheses."""

    active_hypothesis_id: Optional[str] = None
    current_document: Optional[str] = None
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    last_observation: Optional[str] = None
    step: int = 0

    def focus(self, hypothesis_id: str) -> None:
        self.active_hypothesis_id = hypothesis_id
        self.retrieved_chunks = []
        self.last_observation = None
        self.current_document = None

    def observe(self, observation: str, chunks: list[dict[str, Any]] | None = None) -> None:
        self.last_observation = observation
        if chunks is not None:
            self.retrieved_chunks = chunks
        self.step += 1


class Correction(BaseModel):
    """A user correction, retained across engagements.

    Checkpoint 2.1: a correction to a risk rating or a mis-prioritized data request
    should recalibrate how the agent scopes the next deal in the same sub-sector.

    The recalibration must not carry the deal it came from. `from_value` and
    `to_value` are whatever the reviewer typed, which on a live engagement is a client
    figure - a retention rate, a concentration percentage, a customer name. Replaying
    those into another engagement prompt puts one client confidential data into
    another client work product, which is an NDA breach whether or not anyone notices.

    So cross-engagement recall is opt-in and value-free: `shareable` has to be set
    deliberately, and even then only the shape of the correction travels. Within its
    own engagement a correction is returned intact.
    """

    engagement_id: str
    sub_sector: str
    artifact: str
    field_path: str
    from_value: str
    to_value: str
    note: str = ""
    at: _dt.datetime
    shareable: bool = Field(
        default=False,
        description="Cleared for reuse on other engagements. Off by default: the "
        "person recording the correction is the only one who can say it carries no "
        "client-confidential detail.",
    )

    def redacted(self) -> "Correction":
        """The same correction with everything deal-specific removed."""
        return self.model_copy(update={
            "engagement_id": "",
            "from_value": "",
            "to_value": "",
            "note": "",
        })

    def render_for_prompt(self) -> str:
        """How a correction is allowed to appear in a prompt.

        Rendering lives here rather than at the call site so that a redacted
        correction cannot be reassembled into a sentence containing values that were
        deliberately dropped.
        """
        if not self.from_value and not self.to_value:
            return (
                f"{self.artifact}.{self.field_path}: corrected on a prior engagement "
                f"in this sub-sector. Values withheld - treat as a signal that this "
                f"field is commonly got wrong, not as a number."
            )
        return f"{self.artifact}.{self.field_path}: {self.from_value} -> {self.to_value}. {self.note}".strip()


class LongTermMemory:
    """Typed accessor over the state store for the five persistent artifacts."""

    def __init__(self, store: StateStore, engagement_id: str) -> None:
        self.store = store
        self.engagement_id = engagement_id

    # ----------------------------------------------------------- deal profile
    def save_deal_profile(self, profile: DealProfile, *, agent: str) -> None:
        self.store.put(
            self.engagement_id, Collection.DEAL_PROFILE, "current", profile, agent=agent
        )

    def deal_profile(self) -> Optional[DealProfile]:
        raw = self.store.get(self.engagement_id, Collection.DEAL_PROFILE, "current")
        return DealProfile.model_validate(raw) if raw else None

    # -------------------------------------------------------------- ToT search
    def save_thesis_search(self, result: ThesisSearchResult, *, agent: str) -> None:
        """Persist the whole search, pruned branches and reasons included."""
        self.store.put(
            self.engagement_id, Collection.THESIS_SEARCH, "current", result, agent=agent
        )

    def thesis_search(self) -> Optional[ThesisSearchResult]:
        raw = self.store.get(self.engagement_id, Collection.THESIS_SEARCH, "current")
        return ThesisSearchResult.model_validate(raw) if raw else None

    def save_hypothesis_tree(self, tree: HypothesisTree, *, agent: str) -> None:
        self.store.put(
            self.engagement_id, Collection.HYPOTHESIS_TREE, "current", tree, agent=agent
        )

    def hypothesis_tree(self) -> Optional[HypothesisTree]:
        raw = self.store.get(self.engagement_id, Collection.HYPOTHESIS_TREE, "current")
        return HypothesisTree.model_validate(raw) if raw else None

    # ------------------------------------------------------------ data request
    def save_data_request(self, checklist: DataRequestChecklist, *, agent: str) -> None:
        self.store.put(
            self.engagement_id, Collection.DATA_REQUEST, "current", checklist, agent=agent
        )

    def data_request(self) -> Optional[DataRequestChecklist]:
        raw = self.store.get(self.engagement_id, Collection.DATA_REQUEST, "current")
        return DataRequestChecklist.model_validate(raw) if raw else None

    # --------------------------------------------------------- evidence matrix
    def save_evidence_matrix(self, matrix: EvidenceMatrix, *, agent: str) -> None:
        self.store.put(
            self.engagement_id, Collection.EVIDENCE_MATRIX, "current", matrix, agent=agent
        )

    def evidence_matrix(self) -> EvidenceMatrix:
        raw = self.store.get(self.engagement_id, Collection.EVIDENCE_MATRIX, "current")
        if raw:
            return EvidenceMatrix.model_validate(raw)
        return EvidenceMatrix(engagement_id=self.engagement_id, created_by="system")

    # ----------------------------------------------------------- risk register
    def save_risk_register(self, register: RiskRegister, *, agent: str) -> None:
        self.store.put(
            self.engagement_id, Collection.RISK_REGISTER, "current", register, agent=agent
        )

    def risk_register(self) -> RiskRegister:
        raw = self.store.get(self.engagement_id, Collection.RISK_REGISTER, "current")
        if raw:
            return RiskRegister.model_validate(raw)
        return RiskRegister(engagement_id=self.engagement_id, created_by="system")

    # ------------------------------------------------------------ citation log
    def log_citation(self, citation: Citation, hypothesis_id: str, *, agent: str) -> str:
        """Chunk -> source log (Architecture v6.7, slide 1).

        Kept separate from the Evidence Matrix so a later session can audit what was
        retrieved even for chunks that never became a claim.
        """
        return self.store.append(
            self.engagement_id,
            Collection.CITATION_LOG,
            {"hypothesis_id": hypothesis_id, **citation.model_dump(mode="json")},
            agent=agent,
        )

    def citation_log(self) -> list[dict[str, Any]]:
        return [doc for _, doc in self.store.list(self.engagement_id, Collection.CITATION_LOG)]

    # ---------------------------------------------------------------- the deck
    def save_deck(self, deck: Deck, *, agent: str) -> None:
        self.store.put(self.engagement_id, Collection.DECK, "current", deck, agent=agent)

    def deck(self) -> Optional[Deck]:
        raw = self.store.get(self.engagement_id, Collection.DECK, "current")
        return Deck.model_validate(raw) if raw else None

    # ----------------------------------------- cross-engagement user corrections
    def record_correction(self, correction: Correction, *, agent: str) -> str:
        return self.store.append(
            self.engagement_id, Collection.CORRECTION, correction, agent=agent
        )

    def corrections_for_sub_sector(self, sub_sector: str) -> list[Correction]:
        """Prior corrections in the same sub-sector.

        This engagement own corrections come back intact. Corrections belonging to
        other engagements come back only if someone marked them shareable, and always
        redacted - the values a reviewer typed on another deal are that client data,
        and the recalibration this exists for does not need them.
        """
        out: list[Correction] = []
        for eng in self.store.engagements():
            for _, doc in self.store.list(eng, Collection.CORRECTION):
                if doc.get("sub_sector", "").lower() != sub_sector.lower():
                    continue
                correction = Correction.model_validate(doc)
                if correction.engagement_id == self.engagement_id:
                    out.append(correction)
                elif correction.shareable:
                    out.append(correction.redacted())
        return sorted(out, key=lambda c: c.at)
