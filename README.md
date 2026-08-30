# CDD Agent

An AI-based commercial due diligence agent: it tests whether an acquisition target's
market, customers, and growth plan support the price being paid, and produces a **draft**
investment-committee deliverable in which every claim carries a source citation and a
confidence tag.

Implementation of the *CDD Agent Design Specification v2.2* and Checkpoints 1.1–6.1
(CMU Agentic AI Program capstone, M. A. Skaf). Architecture diagram v6.7 is the
reference picture; `docs/ARCHITECTURE.md` maps every module back to the section of the
specs it implements.

---

## What it does

Five phases, five agents, one shared state store:

```
Phase 0        Phase 1              Phase 2-3            Phase 3          Phase 4
Intake    →    Thesis Architect  →  Analyst          ⇄   Risk Auditor  →  Synthesizer
               (Tree of Thought)    (ReAct + RAG)        (audits it)      (writes it)

               ↑ Controller gates every hand-off; Phase 5 risk governance runs throughout
```

* **Intake Agent** — runs the diagnostic intake protocol (Categories A–G) and produces
  the Deal Profile Brief. It has no data-room access at all.
* **Thesis Architect** — the one place Tree of Thought applies. Generates three framings
  of the thesis (growth-led, margin-led, risk-led), scores each with a separate Critic
  persona, prunes, and routes. Ties and all-pruned outcomes go to a human.
* **Analyst** — generates the tiered data-request checklist, then runs the ReAct
  evidence loop over the least-supported Tier-1 hypothesis, grounded in retrieval.
* **Risk Auditor** — a separate persona that screens the Evidence Matrix against the
  standing risk taxonomy and pushes work back to the Analyst.
* **Synthesizer** — populates the enhanced master outline. It cannot retrieve anything,
  so it can only say what the store already supports.

## Why it is built this way

Three design decisions carry most of the weight, and each has a failure mode behind it:

1. **The Critic is not the Generator, and the Auditor is not the Analyst.** A role that
   grades its own work produces an audit worth nothing. This is why the Critic runs on
   CrewAI with its own persona and context, and why the Auditor has no outreach tool.
2. **A retrieval near-miss becomes "No Data", not a low-confidence citation.** The most
   dangerous failure here is *grounded-but-wrong*: a real, correctly cited, superseded
   figure. It looks legitimate, which is what makes it worse than a hallucination. So
   supersession is a metadata filter applied before ranking, and anything below the
   similarity floor is reported as an explicit gap.
3. **An uncited claim is a schema violation, not a style problem.** `Claim` and
   `EvidenceItem` refuse to construct without a citation unless they are tagged No Data,
   and the whole deck is checked again before it is saved.

## Install

**Python 3.11–3.13.** The upper bound is CrewAI's, not this project's: CrewAI declares
`>=3.10,<3.14` and publishes no wheels for 3.14, so the Critic persona cannot be
installed on a 3.14 interpreter. Everything else in the stack — LangChain,
langchain-anthropic, Chroma, MCP — runs fine on 3.14, and the pipeline itself runs there
in offline mode. If `python --version` reports 3.14, install a 3.13 interpreter before
creating the virtualenv, or the `pip install` below will refuse.

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -e ".[dev]"
```

Set credentials (either works — an `ant auth login` profile needs no env var):

```bash
cp .env.example .env   # then set ANTHROPIC_API_KEY
```

Default model is `claude-opus-5`; override with `CDD_MODEL`.

## Run the demo engagement

Project Sentinel is a worked B2B cybersecurity SaaS deal with a cross-sell thesis and a
sponsor buyer — the illustrative case from Section V.B of the design specification. Its
data room contains three deliberate traps:

* A superseded board deck reporting 124% NRR sits alongside the current one reporting
  118%. Retrieval must never cite the old one.
* A contract extract shows 25% of ARR in agreements that step down at renewal — the
  detail the blended NRR figure obscures.
* An expert transcript headed *"recruited independently"* sits in the seller's data
  room. The agent still treats it as management-supplied, so nothing in the demo reaches
  a Confirmed rating. Independence is a property of how evidence was sourced, not of
  what a document says about itself, and a curated reference call is the exact bias the
  outside-in standard exists to catch.

```bash
cdd seed-kb
cdd run project-sentinel --briefing demo/briefing.md --data-room demo/data_room --approve-phase1
```

Or phase by phase, with the gates where they belong:

```bash
cdd intake project-sentinel --briefing demo/briefing.md
cdd thesis project-sentinel
cdd approve project-sentinel --by "M. Skaf"
cdd request project-sentinel
cdd ingest project-sentinel demo/data_room
cdd analyze project-sentinel --data-room demo/data_room
cdd audit project-sentinel
cdd synthesize project-sentinel --out demo/output/draft.md
```

Useful alongside:

```bash
cdd status project-sentinel   # artifacts present + the attributed audit trail
cdd config                    # every design parameter and which checkpoint set it
cdd questions                 # the full intake protocol
cdd purge project-sentinel --confirm   # confidentiality carry-through at close
```

## Offline mode

`CDD_OFFLINE=1` runs the whole pipeline with no API calls. Each agent has a
deterministic path built from the knowledge modules, so orchestration, guardrails,
retrieval, and the artifact contracts all execute end to end.

It is not a simulated model. Offline runs produce structurally valid artifacts with real
citations and no judgment — useful for CI and for demonstrating the machinery, and never
to be shown as diligence. The test suite runs entirely in this mode.

```bash
pytest
```

## Where the human is required

Five triggers, deliberately narrow — over-triggering produces a reviewer who
rubber-stamps:

| Trigger | Behaviour |
|---|---|
| Tier-1 hypothesis below Partially Confirmed at synthesis | Blocks |
| Phase-1 tie within 0.5 points, or all three framings pruned | Blocks; asks for a choice or a clarification |
| Risk Auditor flags conflicting versions of one source | Blocks |
| An action would exceed intake's NDA/access constraints | Hard block — raises, never warns |
| The final go/no-go recommendation | Always, unconditionally |

## Layout

```
src/cdd_agent/
  agents/        five role-bearing agents; thesis_architect/ holds the ToT search
  orchestration/ the Controller — routing logic, not a persona
  guardrails/    authorization, output contract, escalation triggers
  retrieval/     chunking, the two vector indexes, ingestion
  tools/         the four tool categories + role-scoped registry
  knowledge/     outline, risk taxonomy, intake protocol, data-request catalogue
  schemas/       the artifacts, with the citation/confidence contract in the types
  state/         SQLite store, memory, and the MCP access layer
  evaluation/    the six metrics from Checkpoint 6.1
demo/            Project Sentinel: briefing + data room
docs/            ARCHITECTURE.md — module-to-specification map
```

## Status

The output is a structured working draft for partner/MD review. It accelerates the
analytical heavy lifting; it does not make the judgment call on recommendation, price,
or deal-breaker severity. The agent stays inside the commercial workstream — financial,
legal, tax, and technical diligence are separate workstreams whose findings it
references but does not replicate.
