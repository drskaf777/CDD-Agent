"""Self-contained HTML report for a finished engagement.

The local app is an instrument; this is the document that comes off it. Same palette
and type system, different register: single column, serif headings, no controls.

Two output modes from one renderer:

* ``standalone=True`` wraps the page in a full HTML document, for `cdd export` and the
  `/export` endpoint - a file you can open or email.
* ``standalone=False`` emits head-less body content, for publishing as an Artifact,
  where the host supplies the document shell.

Nothing here recomputes anything. It reads the stored artifacts, so the report cannot
disagree with the run that produced it.
"""

from __future__ import annotations

import datetime as _dt
import html
from typing import Any, Iterable, Optional

from cdd_agent.agents.base import AgentContext
from cdd_agent.knowledge.risk_taxonomy import applicable_categories
from cdd_agent.schemas.common import ConfidenceTag

_TAG_CLASS = {
    ConfidenceTag.CONFIRMED.value: "confirmed",
    ConfidenceTag.PARTIALLY_CONFIRMED.value: "partial",
    ConfidenceTag.CONTRADICTED.value: "contradicted",
    ConfidenceTag.NO_DATA.value: "nodata",
}

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700"
    '&family=IBM+Plex+Serif:wght@500;600&display=swap">'
)

_CSS = """
:root{
 --ground:#f5f6f8;--surface:#fff;--surface-2:#eef0f4;--sunken:#e7eaef;
 --ink:#111721;--ink-2:#26303d;--ink-muted:#5b6675;--ink-faint:#8b95a4;
 --rule:#dbe0e8;--rule-strong:#c3cad6;--accent:#7a2e39;--accent-soft:#f3e4e6;
 --confirmed:#1c6f55;--confirmed-bg:#e2f1eb;--partial:#8a6011;--partial-bg:#f8eeda;
 --contradicted:#9c352a;--contradicted-bg:#fae3df;--nodata:#66717f;--nodata-bg:#e9ecf1;
 --sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
 --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
 --serif:"IBM Plex Serif",ui-serif,Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0e1218;--surface:#161b23;--surface-2:#1c222c;--sunken:#11161d;
 --ink:#e9edf3;--ink-2:#cbd3de;--ink-muted:#98a3b3;--ink-faint:#6e7988;
 --rule:#252c37;--rule-strong:#333c4a;--accent:#d2868f;--accent-soft:#332024;
 --confirmed:#57bf99;--confirmed-bg:#152b25;--partial:#dda63c;--partial-bg:#2e2716;
 --contradicted:#ef8b7c;--contradicted-bg:#33201d;--nodata:#8b95a4;--nodata-bg:#1d232c;
}}
:root[data-theme="dark"]{
 --ground:#0e1218;--surface:#161b23;--surface-2:#1c222c;--sunken:#11161d;
 --ink:#e9edf3;--ink-2:#cbd3de;--ink-muted:#98a3b3;--ink-faint:#6e7988;
 --rule:#252c37;--rule-strong:#333c4a;--accent:#d2868f;--accent-soft:#332024;
 --confirmed:#57bf99;--confirmed-bg:#152b25;--partial:#dda63c;--partial-bg:#2e2716;
 --contradicted:#ef8b7c;--contradicted-bg:#33201d;--nodata:#8b95a4;--nodata-bg:#1d232c;
}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
 font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:920px;margin:0 auto;padding:44px 24px 80px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--accent);margin:0 0 8px}
h1{font-family:var(--serif);font-size:34px;line-height:1.15;letter-spacing:-.015em;
 margin:0 0 10px;text-wrap:balance}
.lede{color:var(--ink-muted);margin:0 0 6px;max-width:62ch}
.stamp{border:1px solid var(--accent);border-left-width:3px;background:var(--accent-soft);
 border-radius:6px;padding:13px 16px;margin:22px 0 30px;color:var(--ink-2);font-size:14px}
.stamp strong{color:var(--accent)}
h2{font-family:var(--serif);font-size:23px;letter-spacing:-.01em;margin:38px 0 4px;
 padding-top:20px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-size:15px;margin:22px 0 6px}
.headline{font-weight:600;color:var(--ink);margin:0 0 12px;font-size:16px;max-width:62ch}
p{max-width:66ch}
.muted{color:var(--ink-muted)}.faint{color:var(--ink-faint)}
.mono{font-family:var(--mono)}
.pill{display:inline-flex;align-items:center;border-radius:999px;padding:1px 8px;
 font-family:var(--mono);font-size:10.5px;font-weight:600;letter-spacing:.02em;line-height:1.8}
.pill.confirmed{background:var(--confirmed-bg);color:var(--confirmed)}
.pill.partial{background:var(--partial-bg);color:var(--partial)}
.pill.contradicted{background:var(--contradicted-bg);color:var(--contradicted)}
.pill.nodata{background:var(--nodata-bg);color:var(--nodata)}
.pill.neutral{background:var(--sunken);color:var(--ink-muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
 background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden;margin:24px 0}
.stat{background:var(--surface);padding:14px 16px}
.stat .k{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-faint)}
.stat .v{font-size:25px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.02em;margin-top:2px}
.stat .d{font-size:12px;color:var(--ink-muted)}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface);margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
 color:var(--ink-faint);font-weight:600;padding:9px 13px;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:10px 13px;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;text-align:right;font-family:var(--mono)}
td.id{font-family:var(--mono);font-size:12px;color:var(--ink-muted);white-space:nowrap}
.claim{border-left:2px solid var(--rule);padding:3px 0 3px 13px;margin:0 0 11px;max-width:70ch}
.claim .src{display:block;font-family:var(--mono);font-size:11px;color:var(--ink-faint);margin-top:4px}
.trace{display:flex;flex-direction:column;gap:8px;margin:14px 0}
.tr{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--partial);
 border-radius:6px;padding:11px 14px}
.tr-head{font-family:var(--mono);font-size:11px;color:var(--ink-faint);display:flex;gap:10px;flex-wrap:wrap}
.tr-row{display:grid;grid-template-columns:78px minmax(0,1fr);gap:10px;margin-top:5px;font-size:13px}
.tr-k{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-faint)}
.tr-v{color:var(--ink-2);white-space:pre-wrap;word-break:break-word}
.filtered{margin-top:8px;font-size:12.5px;color:var(--confirmed);background:var(--confirmed-bg);
 border-radius:5px;padding:7px 11px}
footer{margin-top:46px;padding-top:18px;border-top:1px solid var(--rule);
 font-size:12.5px;color:var(--ink-faint);font-family:var(--mono)}
@media print{body{background:#fff}.wrap{max-width:none;padding:0}}
"""


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{round(x * 100)}%"


def _stat(k: str, v: Any, d: str = "") -> str:
    return (
        f'<div class="stat"><div class="k">{_e(k)}</div>'
        f'<div class="v">{_e(v)}</div><div class="d">{_e(d)}</div></div>'
    )


def _table(columns: Iterable[str], rows: Iterable[Iterable[str]], classes: Optional[list[str]] = None) -> str:
    cols = list(columns)
    classes = classes or [""] * len(cols)
    head = "".join(f"<th>{_e(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{classes[i] if i < len(classes) else ""}">{cell}</td>'
            for i, cell in enumerate(r)
        ) + "</tr>"
        for r in rows
    )
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _tag(value: str) -> str:
    return f'<span class="pill {_TAG_CLASS.get(value, "neutral")}">{_e(value)}</span>'


def render_report(ctx: AgentContext, *, standalone: bool = True, include_trace: bool = True) -> str:
    """Render the engagement as a shareable document."""
    mem = ctx.memory
    profile = mem.deal_profile()
    tree = mem.hypothesis_tree()
    matrix = mem.evidence_matrix()
    register = mem.risk_register()
    deck = mem.deck()
    applicable = applicable_categories(ctx.is_strategic_buyer)

    target = profile.target.legal_name if profile else ctx.engagement_id
    title = f"{target} — Commercial Due Diligence"
    parts: list[str] = ['<div class="wrap">']

    # ------------------------------------------------------------- masthead
    parts.append('<p class="eyebrow">Commercial due diligence · working draft</p>')
    parts.append(f"<h1>{_e(target)}</h1>")
    if profile:
        parts.append(f'<p class="lede">{_e(profile.thesis.one_sentence_thesis)}</p>')
        parts.append(
            f'<p class="lede faint">{_e(profile.sector.sub_sector)} · '
            f'{_e(profile.buyer.buyer_type.value)} · {_e(profile.target.deal_stage.value)}'
            + (f' · IC {_e(profile.target.ic_date)}' if profile.target.ic_date else "")
            + "</p>"
        )
    parts.append(
        '<div class="stamp"><strong>Draft for partner/MD review.</strong> '
        "Not an investment-committee recommendation. Every claim below carries a source "
        "citation and a confidence tag; a claim tagged No Data is a logged gap, not an "
        "omission. The go/no-go judgment is human and has not been made here.</div>"
    )

    # ---------------------------------------------------------------- summary
    tier1 = tree.tier_1() if tree else []
    resolved = sum(1 for h in tier1 if matrix.rating(h.id) is not ConfidenceTag.NO_DATA)
    open_gaps = [g for g in register.gaps if not g.resolved]
    parts.append(
        '<div class="stats">'
        + _stat("Tier-1 resolved", f"{resolved}/{len(tier1)}" if tier1 else "—",
                tree.framing_label if tree else "no tree")
        + _stat("Evidence items", len(matrix.items),
                f"{sum(1 for i in matrix.items if i.is_independent)} independent")
        + _stat("Risks", len(register.risks), f"coverage {_pct(register.coverage(applicable))}")
        + _stat("Open gaps", len(open_gaps),
                f"{sum(1 for g in open_gaps if g.blocking)} blocking")
        + _stat("Groundedness", _pct(deck.groundedness()) if deck else "—", "cited claims")
        + "</div>"
    )

    # ------------------------------------------------- hypothesis tree status
    if tree:
        parts.append("<h2>Deal thesis and hypothesis tree</h2>")
        parts.append(
            f'<p class="muted">Selected framing: <strong>{_e(tree.framing_label)}</strong>. '
            f'Provenance: <span class="mono">{_e(tree.created_by)}</span>.</p>'
        )
        parts.append(_table(
            ["ID", "Tier-1 hypothesis", "Status", "Items", "Triangulated"],
            [
                [_e(h.id), _e(h.statement), _tag(matrix.rating(h.id).value),
                 str(len(matrix.for_hypothesis(h.id))),
                 "yes" if matrix.triangulated(h.id) else "no"]
                for h in tier1
            ],
            ["id", "", "", "num", ""],
        ))

    # -------------------------------------------------------------- the deck
    if deck:
        parts.append("<h2>Draft findings</h2>")
        for slide in deck.slides:
            parts.append(f"<h3>{slide.section_number}. {_e(slide.section_title)}</h3>")
            parts.append(f'<p class="headline">{_e(slide.so_what_headline)}</p>')
            for claim in slide.claims:
                src = "; ".join(c.short() for c in claim.citations)
                flag = ' <span class="pill nodata">mgmt data only</span>' if claim.management_data_only else ""
                parts.append(
                    f'<div class="claim">{_e(claim.text)} {_tag(claim.tag.value)}{flag}'
                    + (f'<span class="src">{_e(src)}</span>' if src else "")
                    + "</div>"
                )
            for exhibit in slide.exhibits:
                if not exhibit.columns:
                    continue
                parts.append(
                    f'<p class="faint mono" style="font-size:10.5px;letter-spacing:.07em;'
                    f'text-transform:uppercase;margin:16px 0 0">{_e(exhibit.title)}</p>'
                )
                parts.append(_table(exhibit.columns, [[_e(c) for c in r] for r in exhibit.rows]))
                if exhibit.note:
                    parts.append(f'<p class="faint" style="font-size:12px">{_e(exhibit.note)}</p>')

    # ------------------------------------------------------- risks and gaps
    if register.risks or open_gaps:
        parts.append("<h2>Risk register and outstanding gaps</h2>")
        uncovered = [c.value for c in applicable if c not in register.categories_evaluated()]
        parts.append(
            f'<p class="muted">Taxonomy coverage {_pct(register.coverage(applicable))}'
            + (f'. Not evaluated on this deal: {_e(", ".join(uncovered))}.' if uncovered else ".")
            + " Coverage is measured against the standing taxonomy, not against what "
            "happened to surface.</p>"
        )
        if register.risks:
            parts.append(_table(
                ["ID", "Category", "Finding", "Sev", "Lik", "Score", "Flags"],
                [
                    [_e(r.id), _e(r.category.value), _e(r.description), str(r.severity),
                     str(r.likelihood), f"<strong>{r.score}</strong>",
                     '<span class="pill nodata">mgmt data only</span>' if r.management_data_only else ""]
                    for r in register.ranked()
                ],
                ["id", "muted", "", "num", "num", "num", ""],
            ))
        if open_gaps:
            parts.append(_table(
                ["ID", "Request", "Owner", "Target close", "Status"],
                [
                    [_e(g.id), _e(g.request), _e(g.owner.value),
                     _e(g.target_close_date or "undated"),
                     ('<span class="pill contradicted">blocking</span> ' if g.blocking else "")
                     + ('<span class="pill partial">confirmatory</span>' if g.carried_to_confirmatory else "")]
                    for g in open_gaps
                ],
                ["id", "", "muted", "mono", ""],
            ))

    # ------------------------------------------------------------ the trace
    if include_trace:
        from cdd_agent.state.store import Collection

        trace = [doc for _, doc in ctx.store.list(ctx.engagement_id, Collection.TRACE)]
        audit = ctx.store.audit(ctx.engagement_id, limit=200)
        if trace or audit:
            parts.append("<h2>Trace and audit history</h2>")
            parts.append(
                '<p class="muted">Reasoning steps record why the Analyst went where it '
                "went; the audit trail records what changed in the shared state store "
                "and which agent changed it. Both are reproduced in full so a reviewer "
                "can audit the judgment, not only the conclusion.</p>"
            )
        if trace:
            parts.append('<div class="trace">')
            for t in trace:
                parts.append(
                    '<div class="tr"><div class="tr-head">'
                    f'<span>Step {_e(t.get("step"))}</span>'
                    f'<span>{_e(t.get("hypothesis_id"))}</span>'
                    f'<span>{_e(t.get("phase"))}</span>'
                    f'{_tag(t.get("tag", ""))}</div>'
                    + "".join(
                        f'<div class="tr-row"><span class="tr-k">{k}</span>'
                        f'<span class="tr-v">{_e(t.get(k))}</span></div>'
                        for k in ("thought", "action", "observation") if t.get(k)
                    )
                    + (
                        # The supersession filter is the guard against citing a real
                        # but stale figure. A reviewer needs to see it fired.
                        '<div class="filtered">Supersession filter dropped '
                        f'{len(t.get("superseded_filtered") or [])} stale version(s) '
                        "before ranking: "
                        f'{_e("; ".join(t.get("superseded_filtered") or []))}</div>'
                        if t.get("superseded_filtered") else ""
                    )
                    + "</div>"
                )
            parts.append("</div>")
        if audit:
            parts.append(_table(
                ["Seq", "When (UTC)", "Agent", "Collection", "Key", "Action"],
                [
                    [str(a.seq), a.at.strftime("%Y-%m-%d %H:%M:%S"), _e(a.agent),
                     _e(a.collection), _e(a.key), _e(a.action)]
                    for a in audit
                ],
                ["num", "mono", "", "muted", "id", "mono"],
            ))

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"<footer>Engagement {_e(ctx.engagement_id)} · generated {generated} · "
        f"{'offline deterministic run' if ctx.settings.offline else _e(ctx.settings.model)}"
        "</footer>"
    )
    parts.append("</div>")
    body = "\n".join(parts)

    if not standalone:
        return f"<title>{_e(title)}</title>\n{_FONTS}\n<style>{_CSS}</style>\n{body}"
    return (
        "<!doctype html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title>{_FONTS}<style>{_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
