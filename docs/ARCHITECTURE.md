# Architecture

This document maps the implementation back to the specifications it was built from, and
records the places where a specification statement needed a judgment call in code.

Reference documents:

* *CDD Agent Design Specification v2.2* (July 23, 2026) — Sections I–VIII
* Checkpoint 2.1 — Reasoning, Memory, and Tool Use
* Checkpoint 3.1 — RAG and Retrieval Design Integration
* Checkpoint 4.1 — Tree-of-Thought Integration
* Checkpoint 5.1 — Multi-Agent Architecture and Coordination
* Checkpoint 6.1 — Safety Guardrails and Human Intervention
* Architecture Diagram v6.7 — Safety Update (4 slides)

---

## 1. Module map

| Specification element | Module |
|---|---|
| Five-phase operating model (spec § II) | `orchestration/controller.py` |
| Worked engagement, run in two gated steps | `cli.py` (`cdd demo`), `demo/` |
| Local interface and shareable report | `web/api.py`, `web/static/index.html`, `web/report.py` |
| Diagnostic intake protocol, Categories A–G (§ III) | `knowledge/intake_questions.py`, `schemas/deal_profile.py`, `agents/intake.py` |
| Enhanced master outline, universal + tailored (§ IV) | `knowledge/outline.py` |
| Data-request catalogue and tiering (§ V) | `knowledge/data_request_catalog.py`, `agents/analyst.py` |
| Ingestion and synthesis sequence (§ VI) | `retrieval/ingestion.py`, `agents/synthesizer.py` |
| Standing risk taxonomy and gap logic (§ VII) | `knowledge/risk_taxonomy.py`, `agents/risk_auditor.py` |
| Governance and guardrails (§ VIII) | `guardrails/` |
| ReAct loop, memory split, tool categories (CP 2.1) | `agents/analyst.py`, `state/memory.py`, `tools/` |
| Dual vector indexes, chunking, hybrid retrieval (CP 3.1) | `retrieval/` |
| ToT beam search, Generator/Critic/Controller (CP 4.1) | `agents/thesis_architect/` |
| Five agents, hybrid coordination, artifact hand-off (CP 5.1) | `agents/`, `state/store.py` |
| Guardrails, metrics, intervention triggers (CP 6.1) | `guardrails/`, `evaluation/metrics.py` |

## 2. The framework split, and why it is real

Checkpoint 4.1 § 2.4 assigns frameworks by role. The assignment is not cosmetic here:

| Role | Framework | Module | Why |
|---|---|---|---|
| Thought Generator | LangChain (LCEL) | `thesis_architect/generator.py` | A structured one-shot transform. Templated prompting plus structured-output parsing; no persistent persona needed. |
| Critic / Evaluator | CrewAI | `thesis_architect/critic.py` | Needs its own persona and its own context, so it cannot grade its own output. |
| Decision Maker / Controller | LangChain (conditional routing) | `thesis_architect/beam_search.py` | Deterministic control logic — an orchestration branch, not a persona. Implemented as pure functions with no I/O, which is what makes the routing rules directly testable. |
| Memory / State Manager | State store + MCP | `state/store.py`, `state/access.py`, `state/mcp_server.py` | The store holds the branches and scores; MCP exposes its read/write operations as one protocol both frameworks can call. |

The store/MCP distinction is preserved exactly as the checkpoint draws it. `StateStore`
is the persistence layer (the same one Long-Term Memory uses — not new infrastructure).
`StateAccess` is the protocol; `LocalStateAccess` satisfies it in-process and
`MCPStateAccess` satisfies it across a process boundary. The Thesis Architect does not
know which is in play, which is the point of MCP being an access layer rather than the
store.

## 3. Two places the specifications needed reconciling

**ToT depth.** Checkpoint 4.1 § 2.2 caps depth at 2 (root → Tier-1 → assumptions);
§ 2.3 says depth is "1 level, not iteratively deepened", and Architecture v6.7 slide 3
says "depth capped at 1". These describe two different things, so the code names them
separately: `tree_max_depth = 2` is the depth of the produced artifact, and
`search_expansion_levels = 1` is how many levels the beam search expands. Both are in
`config.py`.

**Data-request tiering.** Section V fixes a tier per catalogue item, but tier is defined
by how load-bearing an item is *to the hypothesis tree*, and the tree differs per deal.
`Analyst.generate_data_request` therefore demotes a Tier-1 catalogue item to Tier 2 when
nothing in the selected tree depends on it, rather than dropping it. The checklist stays
complete; the blocking set stays honest.

## 4. Guardrails: where each one is enforced

Checkpoint 6.1's preventive layer is enforced in code paths, not in prompt text, because
a guardrail a model is asked to respect is a suggestion.

| Guardrail | Enforced by | Mechanism |
|---|---|---|
| Input checks (NDA / Category F) | `guardrails/authorization.py` | `ToolAuthorization.check` runs before any tool is constructed; `available_tools` filters the model's tool list so a forbidden tool is never offered |
| Output constraints | `schemas/deck.py`, `schemas/evidence.py`, `guardrails/output_contract.py` | A `Claim` or `EvidenceItem` cannot be constructed uncited unless tagged No Data; `check_deck` re-validates the whole deck before it is saved |
| Source verification | `retrieval/indexes.py`, `agents/risk_auditor.py` | Supersession filtered by version group + date *before* ranking; the Auditor separately flags undated version conflicts, which the date filter cannot resolve |
| Tool access limits | `guardrails/authorization.py`, `tools/registry.py` | `ROLE_TOOLS` scopes each role; the registry builds only the permitted tools |
| Escalation rules | `guardrails/escalation.py`, `orchestration/controller.py` | Five hard-coded triggers; the Controller never advances past a blocking one |
| Runtime monitoring | `state/store.py` | `put()` requires an `agent` argument and refuses an empty one; every write appends to `audit_log` |

## 5. The confidence schema is the spine

`ConfidenceTag` (Confirmed / Partially Confirmed / Contradicted / No Data) appears at
every layer, which is what lets a failure in one become an auditable state rather than
a silent degradation:

* The ReAct Observation step emits it (CP 2.1).
* A below-floor retrieval maps onto No Data instead of returning a weak match (CP 3.1 § 5).
* ToT hypotheses carry it as a placeholder before evidence arrives (CP 4.1 § 2.2).
* The synthesis gate reads it: a Tier-1 hypothesis must clear Partially Confirmed or
  carry a dated gap.
* The output contract requires it on every claim (CP 6.1).

Two rating rules live in `EvidenceMatrix.rating` and are worth stating explicitly:

* **Contradicted dominates.** A contradicting finding is the most decision-relevant
  state; averaging it against supporting items would bury it.
* **Management data alone cannot reach Confirmed.** It degrades to Partially Confirmed,
  because management's own base case is what is being tested (spec § I, § VIII).

## 6. Coordination

Sequential backbone with two embedded loops, per Checkpoint 5.1:

```
Intake ──▶ Thesis Architect ──▶ Analyst ──▶ Risk Auditor ──▶ Synthesizer
             │  ↺ generate/score/prune       ↑         │
             │  (beam search, width 3)       └─────────┘
             │                                ↺ flagged gap routes back
             └─ human-approval gate before Phase 2
```

Communication is one-way structured hand-off through the shared store — Deal Profile
Brief, Hypothesis Tree, Evidence Matrix, Risk Register — not free-form dialogue.
Two-way exchange exists only where one role must grade or push back on another's work:
Critic ↔ Generator, and Risk Auditor ↔ Analyst.

The Analyst ↔ Auditor loop is bounded by `max_auditor_rounds` (default 3). The loop is a
deliberate latency-for-reliability trade; unbounded, it would trade away the reliability
it was added to buy.

## 7. The interface

`web/api.py` is deliberately thin: every endpoint calls the same agents the CLI calls,
so the interface cannot do anything the pipeline would not do and cannot skip a gate.
There is no endpoint that selects a framing without recording who selected it, and none
that synthesises without the output contract running first — a 422 carrying the
violations is the guardrail working, not an error to route around.

The read model is a single snapshot endpoint rather than a dozen fine-grained ones. A
diligence UI is read *across* its panels — the evidence against the risks against the
trace — so serving them from one consistent read avoids showing a matrix from one
moment beside a register from another.

**The trace is a first-class artifact.** The audit log already recorded that the
Evidence Matrix changed and who changed it; that is not enough to review a judgment
call. `Collection.TRACE` stores each Thought → Action → Observation step, so a reviewer
can see which hypothesis was weakest, what was asked, and what came back. The report
reproduces both in full.

## 8. Evaluation

`evaluation/metrics.py` computes the six metrics from Checkpoint 6.1. Two of them are
returned as `None` with `needs_human=True`, on purpose:

* **Calibration** — whether sampled Confirmed tags hold up under human review. The
  module draws the sample; it does not grade itself.
* **Numeric correctness** — computed figures are queued for spot-check against source
  documents, with the inputs recorded.

`groundedness` excludes No Data claims from its denominator. Counting a logged gap as an
ungrounded claim would create pressure to assert rather than to log, which is precisely
backwards.

## 9. Known limits

* **Document extraction** handles `.txt` / `.md` and `.csv` / `.tsv` / `.json`. Real
  data rooms are PDF and Office files; `retrieval/ingestion.py` reports unsupported
  files as skipped rather than silently ignoring them. Adding an extractor is a
  contained change at `ingest_directory`.
* **Primary research** commissions and records a scoped, authorized interview
  programme; it does not contact anyone. Placing the calls is a human step, and the
  returned notes enter through `InterviewNote`.
* **The Synthesizer's section mapping** is keyword-overlap based. It is deterministic
  and inspectable, but a hypothesis whose vocabulary does not match a section's key
  elements will land in the summary sections rather than the specific one.
* **Offline mode is machinery, not judgment.** See the README.
* **Python 3.14 cannot run the Critic.** CrewAI declares `>=3.10,<3.14`, so it is an
  optional `[critic]` extra rather than a base dependency - pinning `requires-python` to
  match made the entire package uninstallable on 3.14, including the parts unrelated to
  the Critic. LangChain, langchain-anthropic, Chroma, and MCP all run on 3.14. Without
  the extra, `get_crew_llm` raises a message naming the extra; it does not fall back to
  the deterministic rubric, because a Critic that is quietly not the Critic defeats the
  separation it exists for. The MCP layer targets `mcp>=2`, where `FastMCP` was renamed
  `MCPServer`.
* **What has and has not been executed.** The offline pipeline, both indexes, the
  guardrails, and the routing rules are covered by the test suite and were run end to
  end on the demo engagement. The model-backed paths — the LCEL Generator, the CrewAI
  Critic, the Synthesizer's headline pass — are written against the documented APIs but
  have not been run against a live model here.
