"""MCP server exposing the shared state store.

Run with::

    python -m cdd_agent.state.mcp_server

This is the access layer described in Checkpoint 4.1 s 2.4 and drawn on Architecture
v6.7 slide 2 ("MCP - access layer, not the store"). It deliberately exposes only the
store's read/write operations. It does not expose retrieval, computation, or outreach:
those are role-scoped tools, and widening this surface would quietly defeat the
tool-access-limits guardrail in Checkpoint 6.1.

Every write still requires an ``agent`` argument, so crossing the process boundary
does not create a hole in the attribution guarantee.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from cdd_agent.state.store import StateStore

mcp = FastMCP("cdd-state")
_store = StateStore()


@mcp.tool()
def state_read(engagement_id: str, collection: str, key: str) -> str:
    """Read one document from the shared state store."""
    return json.dumps(_store.get(engagement_id, collection, key))


@mcp.tool()
def state_write(
    engagement_id: str, collection: str, key: str, document: dict[str, Any], agent: str
) -> str:
    """Write one document. `agent` attributes the write in the audit log."""
    version = _store.put(engagement_id, collection, key, document, agent=agent)
    return json.dumps(version)


@mcp.tool()
def state_list_keys(engagement_id: str, collection: str) -> str:
    """List document keys in a collection."""
    return json.dumps([k for k, _ in _store.list(engagement_id, collection)])


@mcp.tool()
def state_list_documents(engagement_id: str, collection: str) -> str:
    """List every document in a collection, with its key."""
    return json.dumps(
        [{"key": k, "document": d} for k, d in _store.list(engagement_id, collection)]
    )


@mcp.tool()
def state_audit(engagement_id: str, limit: int = 100) -> str:
    """Read the attributed, timestamped audit trail for an engagement."""
    entries = _store.audit(engagement_id, limit=limit)
    return json.dumps(
        [
            {
                "seq": e.seq,
                "collection": e.collection,
                "key": e.key,
                "agent": e.agent,
                "action": e.action,
                "at": e.at.isoformat(),
                "digest": e.payload_digest,
            }
            for e in entries
        ]
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
