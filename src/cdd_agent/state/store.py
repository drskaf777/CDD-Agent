"""Shared state store - the backbone of inter-agent communication.

Checkpoint 5.1: hand-offs go through a shared artifact store rather than agent-to-agent
chat, which is what keeps the interface auditable and lets new agents be added at the
same interface without redesigning how existing agents communicate.

Checkpoint 6.1 runtime-monitoring guardrail: every write is timestamped and attributed
to an agent. The audit log is append-only even when a write overwrites a document, so
the full run is reconstructable after the fact.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from cdd_agent.config import get_settings


class Collection(str, Enum):
    """Long-Term Memory contents, per Architecture v6.7 slide 1."""

    DEAL_PROFILE = "deal_profile"
    THESIS_SEARCH = "thesis_search"       # all ToT branches incl. pruned, with reasons
    HYPOTHESIS_TREE = "hypothesis_tree"   # the selected branch
    DATA_REQUEST = "data_request"
    EVIDENCE_MATRIX = "evidence_matrix"
    RISK_REGISTER = "risk_register"
    CITATION_LOG = "citation_log"         # chunk -> source, per the v6.7 diagram
    TRACE = "trace"                       # Thought -> Action -> Observation steps
    EXHIBIT = "exhibit"                   # quantitative exhibits computed in Phase 3
    STRUCTURED = "structured"             # parsed tabular data-room files
    DECK = "deck"
    ESCALATION = "escalation"
    CORRECTION = "correction"             # user corrections, replayed across engagements
    METRICS = "metrics"


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    engagement_id: str
    collection: str
    key: str
    agent: str
    action: str
    at: _dt.datetime
    payload_digest: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    engagement_id TEXT NOT NULL,
    collection    TEXT NOT NULL,
    key           TEXT NOT NULL,
    body          TEXT NOT NULL,
    version       INTEGER NOT NULL,
    updated_at    TEXT NOT NULL,
    updated_by    TEXT NOT NULL,
    PRIMARY KEY (engagement_id, collection, key)
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    engagement_id  TEXT NOT NULL,
    collection     TEXT NOT NULL,
    key            TEXT NOT NULL,
    agent          TEXT NOT NULL,
    action         TEXT NOT NULL,
    at             TEXT NOT NULL,
    body           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_engagement ON audit_log (engagement_id, seq);
"""


class StateStore:
    """SQLite-backed document store: one file, atomic writes, full audit trail."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        settings.ensure_dirs()
        self.db_path = Path(db_path) if db_path else settings.state_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ writes
    def put(
        self,
        engagement_id: str,
        collection: Collection | str,
        key: str,
        document: Any,
        *,
        agent: str,
    ) -> int:
        """Write a document. `agent` is mandatory - anonymous writes are impossible."""
        if not agent or not agent.strip():
            raise ValueError("every state-store write must be attributed to an agent")
        coll = _coll(collection)
        body = _to_json(document)
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT version FROM documents WHERE engagement_id=? AND collection=? "
                "AND key=?",
                (engagement_id, coll, key),
            ).fetchone()
            version = (row["version"] + 1) if row else 1
            self._conn.execute(
                "INSERT INTO documents (engagement_id, collection, key, body, version,"
                " updated_at, updated_by) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(engagement_id, collection, key) DO UPDATE SET"
                " body=excluded.body, version=excluded.version,"
                " updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                (engagement_id, coll, key, body, version, now, agent),
            )
            self._conn.execute(
                "INSERT INTO audit_log (engagement_id, collection, key, agent, action,"
                " at, body) VALUES (?,?,?,?,?,?,?)",
                (
                    engagement_id,
                    coll,
                    key,
                    agent,
                    "create" if version == 1 else "update",
                    now,
                    body,
                ),
            )
        return version

    def append(
        self,
        engagement_id: str,
        collection: Collection | str,
        document: Any,
        *,
        agent: str,
    ) -> str:
        """Append to a log-style collection under an auto-generated key."""
        coll = _coll(collection)
        with self._lock:
            n = self._conn.execute(
                "SELECT COUNT(*) c FROM documents WHERE engagement_id=? AND collection=?",
                (engagement_id, coll),
            ).fetchone()["c"]
        key = f"{n + 1:06d}"
        self.put(engagement_id, coll, key, document, agent=agent)
        return key

    # ------------------------------------------------------------------- reads
    def get(
        self, engagement_id: str, collection: Collection | str, key: str
    ) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT body FROM documents WHERE engagement_id=? AND collection=? AND key=?",
            (engagement_id, _coll(collection), key),
        ).fetchone()
        return json.loads(row["body"]) if row else None

    def list(
        self, engagement_id: str, collection: Collection | str
    ) -> list[tuple[str, dict[str, Any]]]:
        rows = self._conn.execute(
            "SELECT key, body FROM documents WHERE engagement_id=? AND collection=?"
            " ORDER BY key",
            (engagement_id, _coll(collection)),
        ).fetchall()
        return [(r["key"], json.loads(r["body"])) for r in rows]

    def engagements(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT engagement_id FROM documents ORDER BY engagement_id"
        ).fetchall()
        return [r["engagement_id"] for r in rows]

    def audit(self, engagement_id: str, limit: int = 500) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT seq, engagement_id, collection, key, agent, action, at, body"
            " FROM audit_log WHERE engagement_id=? ORDER BY seq DESC LIMIT ?",
            (engagement_id, limit),
        ).fetchall()
        return [
            AuditEntry(
                seq=r["seq"],
                engagement_id=r["engagement_id"],
                collection=r["collection"],
                key=r["key"],
                agent=r["agent"],
                action=r["action"],
                at=_dt.datetime.fromisoformat(r["at"]),
                payload_digest=_digest(r["body"]),
            )
            for r in rows
        ]

    # --------------------------- confidentiality carry-through (spec s VIII)
    def purge_engagement(
        self, engagement_id: str, *, agent: str, keep_audit: bool = True
    ) -> int:
        """Delete a deal's documents at engagement close.

        The audit log is retained by default so the run stays reconstructable for
        internal review; pass keep_audit=False when the NDA requires full teardown.
        """
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM documents WHERE engagement_id=?", (engagement_id,)
            )
            deleted = cur.rowcount
            self._conn.execute(
                "INSERT INTO audit_log (engagement_id, collection, key, agent, action,"
                " at, body) VALUES (?,?,?,?,?,?,?)",
                (
                    engagement_id,
                    "*",
                    "*",
                    agent,
                    "purge",
                    _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    json.dumps({"documents_deleted": deleted}),
                ),
            )
            if not keep_audit:
                self._conn.execute(
                    "DELETE FROM audit_log WHERE engagement_id=? AND action != 'purge'",
                    (engagement_id,),
                )
        return deleted

    def close(self) -> None:
        self._conn.close()


def _coll(collection: Collection | str) -> str:
    return collection.value if isinstance(collection, Collection) else collection


def _to_json(document: Any) -> str:
    if hasattr(document, "model_dump_json"):
        return document.model_dump_json()
    return json.dumps(document, default=str, sort_keys=True)


def _digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def iter_collections() -> Iterable[Collection]:
    return list(Collection)
