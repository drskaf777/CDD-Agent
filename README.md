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

Python 3.11 or later.

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -e ".[dev]"
```

**The Critic needs Python 3.11–3.13.** CrewAI is the Critic's framework, but it declares
`>=3.10,<3.14` and ships no wheels for 3.14, so it is an optional extra rather than a
base dependency — otherwise the whole package would be uninstallable on 3.14, including
the parts that have nothing to do with the Critic. On a 3.11–3.13 interpreter:

```bash
pip install -e ".[dev,critic]"
```

Without it, everything runs except the model-backed Critic, which raises a clear error
naming the extra rather than quietly substituting something else for the persona.

**Two conflicts to know about if you install the Critic.** CrewAI constrains two of this
project's own dependencies, and pip resolves them silently:

| Package | Plain install | With `[critic]` | Consequence |
|---|---|---|---|
| `chromadb` | 1.5.x | 1.1.x (CrewAI pins `~=1.1.0`) | A persisted index is **not** readable across the two |
| `mcp` | 2.x | 1.x on CrewAI ≥ 1.15 | `FastMCP` vs `MCPServer` — both are supported, no action needed |

The `mcp` split is handled: `state/mcp_server.py` imports whichever name exists.

The `chromadb` split is not something code can paper over — Chroma cannot read its own
persisted format across versions, and it fails with a Rust panic rather than a Python
error. The index is therefore version-stamped, and opening a mismatched one raises a
readable `IndexVersionMismatch` telling you to rebuild. If you want both interpreters
working at once, give each its own directory:

```bash
CDD_CHROMA_DIR=./data/chroma-313 CDD_DATA_DIR=./data313 cdd demo
```

## Running it live

Offline mode exercises the machinery. A live run needs three things together, and
`cdd preflight` checks all of them before you depend on it:

1. **`ANTHROPIC_API_KEY` in the environment.** Not an `ant auth login` profile —
   CrewAI's Anthropic provider reads the variable directly and ignores the CLI
   profile, so the Critic fails without it even on an authenticated machine.
2. **Python 3.11–3.13**, so CrewAI is installed and the Critic can actually run.
3. **`CDD_OFFLINE=0`.**

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:CDD_OFFLINE="0"
.venv313\Scripts\cdd preflight --data-room demo/data_room
```

**Identity-linked keys need a workspace id.** A key scoped to a person rather than a
workspace is rejected with `400 anthropic-workspace-id is required`. Set the workspace
it should act in — Console → Settings → Workspaces, the id looks like `wrkspc_...`:

```powershell
$env:CDD_WORKSPACE_ID="wrkspc_..."
```

The header is then sent by both the LangChain client and CrewAI's Anthropic provider.
Ordinary keys carry the workspace implicitly and need nothing here.

Preflight makes one minimal model call — a fraction of a cent — because a key being
*present* is not the same as a key that *works*. Pass `--no-call` to skip it.

Because CrewAI pins an older chromadb, a 3.13 environment needs its own index
directory (see the conflict table above):

```powershell
$env:CDD_CHROMA_DIR="./data/chroma-313"; $env:CDD_DATA_DIR="./data313"
```

Default model is `claude-opus-5`; override with `CDD_MODEL`. Live intake reads the
briefing for real, so `cdd run --briefing demo/briefing.md` works without the fixture.

**CrewAI traces.** The Critic runs on CrewAI, which collects execution traces and — while
the preference is unset — asks whether to show them, with a 20-second timeout, on *every*
kickoff. Phase 1 scores three branches in three separate crews, so that is three
interactive prompts in a non-interactive pipeline. The preference is therefore always
stated explicitly. Traces are **on** by default; turn them off with:

```powershell
$env:CDD_CREWAI_TRACING="false"
```

If you answered "N" to that prompt at some point, CrewAI saved it. Setting
`CDD_CREWAI_TRACING=true` overrides the saved preference on the next run.

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

Run it in two commands — and the fact that it takes two is the point:

```bash
cdd demo
```

That seeds the knowledge base, loads the Deal Profile Brief, and runs the Phase-1 beam
search. It then **stops**, because all three framings score within the 0.5-point tie
band and ties are not auto-resolved by reranking. Pick one, as the deal team would:

```bash
cdd demo --pick risk --no-reset
```

Phases 2–4 then run: data request, ingestion, the ReAct evidence loop, the Risk Auditor,
and synthesis into `demo/output/project-sentinel-draft.md`, followed by the metric set.

`cdd demo` loads the Deal Profile from `demo/deal_profile.json` rather than running
Phase-0 intake on `demo/briefing.md`. Offline mode has no model to extract a thesis from
prose, and the intake agent will not invent one — so `cdd run --briefing demo/briefing.md`
halts at the Phase-0 gate by design. With an API key set and `CDD_OFFLINE=0`, that
command runs intake for real and the fixture is unnecessary.

## Listed targets

A public company has already been read by everyone who reads filings, and is repriced
daily on the result. Restating that analysis is not diligence, so for a listed target
the work shifts to the differential — what this engagement concludes that the price
does not already reflect — and to the constraints that only exist because there is a
share price and other shareholders.

Three structures are supported, and they are kept distinct because they ask different
questions of the same company:

| Structure | What changes |
| --- | --- |
| Significant minority stake | The buyer compels nothing. The plan being underwritten belongs to incumbent management and cannot be replaced, so the diligence is about whether that plan works, what influence is actually secured, and whether a block that size can be exited. |
| Controlling stake, listing retained | Minority shareholders continue alongside. Related-party value capture is limited, the public-company cost base stays in the base case, and free-float and listing requirements bind. |
| Take-private | The offer must win a vote at a premium to an unaffected price, which sets a floor the base case has to clear. Completion becomes a commercial question: change-of-control consents that can be withheld are revenue at risk on close. Delisting removes the public-company cost base, which is a real lever and must be evidenced rather than assumed. |

The structure is a Phase-1 prerequisite, not a detail. A listed target with no named
structure will not start the beam search, because a decomposition written before the
structure is known tests the wrong question.

**MNPI is a hard gate, not a warning.** Reading a data room on a listed issuer puts the
firm in possession of material non-public information and restricts everyone briefed
from trading. That is not a decision an agent may take on the firm behalf, so until
compliance records the acknowledgement the data-room tools are withheld entirely — not
offered and refused, but absent from the tool list. Market search stays available,
because the public record is precisely what may be worked on before wall-crossing.
Contact with issuer insiders defaults to denied under Reg FD: the exposure of a leaky
call lands on the asset being bought.

**Two source kinds, and one deliberate omission.** A filing is *attested* — audited,
certified, legally exposed — but it is still the issuer own account of itself, so it
carries extra weight without becoming independent. Sell-side research is explicitly
**not** independent: analysts are guided by the company, so consensus agreeing with the
plan is one source counted twice, which is the failure the four-way confidence schema
exists to prevent. Documents are classified by filename, because a data room routinely
holds the last 10-K beside the board pack, and ambiguity resolves towards the more
restrictive reading.

Everything is scoped. A private deal sees none of it — no listed-target risk categories
counting against its coverage metric, no public exhibits, no structure module — and the
data request stops asking management for what they published last quarter, marking
those items retrieved rather than requested.

### The listed demo

Project Atlas is a worked listed-target deal on Meridian Data Systems, a fictional
NASDAQ-listed enterprise data-integration company. Its data room mixes a public record
(annual report, earnings call, proxy, broker note) with a confidential board pack, and
the trap is the gap between them: the internal plan carries revenue above the range
guided to the market, and the reported segments do not correspond to how the business
is actually run.

The same company is briefed three ways, one per structure:

```bash
cdd run project-atlas-tp  --briefing demo/public/briefing-take-private.md \
                          --data-room demo/public/data_room --approve-phase1
cdd approve project-atlas-tp --by "you" --branch growth   # ties are not auto-resolved
cdd request project-atlas-tp
cdd ingest  project-atlas-tp demo/public/data_room
cdd analyze project-atlas-tp --data-room demo/public/data_room
cdd audit   project-atlas-tp
cdd synthesize project-atlas-tp
```

Swap the briefing for `briefing-minority.md` or `briefing-control.md` to see the same
company produce a different deck. The take-private run raises *deal completion and
approvals* and carries a completion-conditions exhibit; the minority run raises
*governance and control rights* and carries an influence-rights exhibit instead. Both
share the unaffected price, the register, consensus against the management plan,
guidance against delivery, and the MNPI position.

To run it against a real company, drop that company actual filings into a data-room
folder and write the briefing to match. Filings are recognised by filename — `10-K`,
`10-Q`, `DEF-14A`, `earnings-call`, `annual-report` — so they are cited as attested
public record rather than as confidential management material.

### Running against a real listed company

The filings are fetched rather than committed, so the repository carries no copy of
another company disclosures and the data room is always current:

```bash
python scripts/fetch_edgar.py FRSH --email you@example.com
```

That resolves the ticker to a CIK, pulls the latest 10-K, 10-Q and proxy, and writes
them to `demo/real/frsh/data_room/` with a provenance header naming the EDGAR URL.
The `--email` is required because SEC returns 403 to undeclared automated traffic; it
is sent only to sec.gov, in the User-Agent header their access policy asks for.

Published consensus and market data are not on EDGAR. Add them as one document and
name it so it contains `analyst` or `consensus` - the ingester then classifies it as
sell-side research, which the system treats as *not* independent corroboration, on
the grounds that covering analysts are guided by the company.

Then write a briefing and run the pipeline as above.
`demo/real/frsh/briefing-take-private.md` is a worked example on Freshworks Inc. It
opens by saying what it is: the company, its filings and the market data are real; the
buyer, thesis and hold assumptions are a screening exercise, and no approach exists.
Keeping that line in the briefing matters, because everything downstream inherits it -
a deck that reads as though a live take-private were underway would be the most
misleading artifact the system could produce.

## The interface

```bash
cdd serve
```

The same interface drives a live run — it reads `CDD_OFFLINE` like everything else, and
the mode badge shows the model name instead of "offline mode". Serve from the 3.13
environment so the Critic is available:

```powershell
$env:CDD_OFFLINE="0"; $env:CDD_DATA_DIR="./data313"; $env:CDD_CHROMA_DIR="./data/chroma-313"
.venv313\Scripts\cdd serve
```

Live phases take minutes rather than milliseconds — Phase 1 alone is three generations
and three separate Critic crews — so the running indicator counts elapsed seconds. A
spinner with no elapsed time is indistinguishable from a hang.

Opens a local app at http://127.0.0.1:8000 (localhost only — an engagement's data room
is client-confidential and this server has no authentication).

The left rail is the five-phase pipeline as a live stepper; the centre is tabbed:

| Tab | What it shows |
|---|---|
| Overview | Deal Profile Brief, headline metrics, and each role's tool authorization under intake Category F |
| Hypotheses | The three ToT framings side by side with their Critic scores — and the gate where you pick one |
| Data request | The tiered checklist, each item traced to the hypotheses it supports |
| Evidence | The Evidence Matrix: hypothesis → data → confidence, with citations and similarity scores |
| Risks & gaps | The register ranked by severity × likelihood, plus taxonomy coverage and open gaps |
| Draft | The generated deck, tags and sources inline |
| **Trace** | Every Thought → Action → Observation step and every attributed state-store write |

**Run pipeline** advances the phases in order and stops at any gate that needs a person,
which makes it the spine of a live demo: it runs, it halts at the Phase-1 tie, you choose
a framing, it carries on to the draft. The rail shows which agent is working; completed
phases fill the spine.

The Trace tab is the point of the whole thing. Reasoning steps record *why* the Analyst
went where it went; state writes record *what* changed and which agent changed it.
Filter to either, or read them interleaved. Nothing is anonymous and nothing is
retro-fitted: the store refuses an unattributed write.

Clearing the Phase-1 gate needs a name — it is written into the artifact provenance and
the audit trail, because a gate cleared by nobody in particular is not a gate.

**Export a shareable report:**

```bash
cdd export project-sentinel --out report.html
```

One self-contained HTML file — no external assets beyond the webfont — carrying the
hypothesis tree, the draft findings, the risk register, and the full trace. The
Export report button in the app does the same thing.

## Phase by phase

Or drive it from the terminal, with the gates where they belong:

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
  public/        Project Atlas: a listed target, briefed three ways
  real/          worked run on a real listed company; filings are fetched
docs/            ARCHITECTURE.md — module-to-specification map
```

## Status

The output is a structured working draft for partner/MD review. It accelerates the
analytical heavy lifting; it does not make the judgment call on recommendation, price,
or deal-breaker severity. The agent stays inside the commercial workstream — financial,
legal, tax, and technical diligence are separate workstreams whose findings it
references but does not replicate.
