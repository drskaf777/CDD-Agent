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

from pydantic import BaseModel

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
    Stored with the sub-sector so it can be recalled as context, not replayed blindly.
    """

    engagement_id: str
    sub_sector: str
    artifact: str
    field_path: str
    from_value: str
    to_value: str
    note: str = ""
    at: _dt.datetime


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
        """Recall prior corrections in the same sub-sector, across all engagements."""
        out: list[Correction] = []
        for eng in self.store.engagements():
            for _, doc in self.store.list(eng, Collection.CORRECTION):
                if doc.get("sub_sector", "").lower() == sub_sector.lower():
                    out.append(Correction.model_validate(doc))
        return sorted(out, key=lambda c: c.at)
