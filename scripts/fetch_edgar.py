"""Build a data room for a listed target from its own SEC filings.

The point of running against a real company is that every citation in the deck
resolves to a document anyone can open. So this fetches the filings rather than
shipping a copy of them: the repository stays lean, the filings stay current, and
nobody is redistributing a company own disclosures.

    python scripts/fetch_edgar.py FRSH --email you@example.com

SEC requires a declared contact address on automated requests and will return 403
without one, so --email is mandatory. It is sent only to sec.gov, in the User-Agent
header their access policy asks for.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.sec.gov"
WANTED = {
    "10-K": ("{t}_10-K_{d}.txt", "Form 10-K annual report", 3),
    "10-Q": ("{t}_10-Q_{d}.txt", "Form 10-Q quarterly report", 4),
    "DEF 14A": ("{t}_DEF-14A_proxy_{d}.txt", "Definitive Proxy Statement (Schedule 14A)", 2),
    # The 8-K itself is a cover page; the earnings release is an exhibit to it, and
    # that is where guidance lives. Guidance against delivery, quarter by quarter, is
    # the cheapest test of management credibility a listed target offers - and it
    # needs several quarters to say anything at all.
    "8-K": ("{t}_earnings-call_{d}.txt", "quarterly earnings release (Exhibit 99.1 to Form 8-K)", 6),
}
NOISE = re.compile(r"^(iso4217:|xbrli:|utr:|http://|\d{10}$|[0-9-]+$|true$|false$)", re.I)
EARNINGS_HINTS = ("earningsrelease", "earnings-release", "earningsrelea",
                  "quarterlyearnings", "pressrelease", "exhibit99", "ex-99", "ex99")


def earnings_exhibit(cik: str, accession: str, email: str) -> str | None:
    """The earnings release filed as an exhibit, if this 8-K carries one.

    Most 8-Ks are not earnings, so a miss here is normal and simply skipped.
    """
    listing = json.loads(get(
        f"{BASE}/Archives/edgar/data/{int(cik)}/{accession}/index.json", email))
    names = [item.get("name", "") for item in listing.get("directory", {}).get("item", [])]
    for name in names:
        stem = name.lower().replace("_", "").replace("-", "")
        if name.lower().endswith((".htm", ".html")) and any(h.replace("-", "") in stem
                                                            for h in EARNINGS_HINTS):
            return name
    return None


def get(url: str, email: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": f"CDD-Agent {email}", "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    if response.headers.get("Content-Encoding") == "gzip":
        import gzip

        raw = gzip.decompress(raw)
    time.sleep(0.2)  # SEC asks for no more than ten requests a second
    return raw


def to_text(html: bytes, header: str) -> str:
    """Flatten a filing to text, keeping table cells apart so figures stay readable."""
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" | ")
    for tr in soup.find_all("tr"):
        tr.insert_after("\n")
    lines = [" ".join(l.split()) for l in soup.get_text("\n").replace("\u00a0", " ").split("\n")]
    lines = [l for l in lines if l and l not in {"|", "| |"} and not NOISE.match(l)]
    # Everything before the cover page is XBRL plumbing, not the filing.
    for i, line in enumerate(lines):
        if "SECURITIES AND EXCHANGE COMMISSION" in line:
            lines = lines[i:]
            break
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return header.rstrip() + "\n\n" + re.sub(r"(\|\s*){2,}", " | ", body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker")
    ap.add_argument("--email", required=True,
                    help="Contact address for the SEC User-Agent header. Required by "
                         "their access policy; sent only to sec.gov.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    out = args.out or Path("demo/real") / ticker.lower() / "data_room"
    out.mkdir(parents=True, exist_ok=True)

    tickers = json.loads(get(f"{BASE}/files/company_tickers.json", args.email))
    match = next((r for r in tickers.values() if r["ticker"].upper() == ticker), None)
    if match is None:
        print(f"{ticker}: no CIK on EDGAR", file=sys.stderr)
        return 1
    cik = str(match["cik_str"]).zfill(10)
    print(f"{ticker} = {match['title']} (CIK {cik})")

    subs = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik}.json", args.email))
    recent = subs["filings"]["recent"]
    rows = list(zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"],
                    recent["filingDate"], strict=True))

    written = 0
    for form, (pattern, label, limit) in WANTED.items():
        hits = [r for r in rows if r[0] == form][:limit]
        if not hits:
            print(f"  {form}: none found")
            continue
        for _, accession, document, date in hits:
            folder = accession.replace("-", "")
            if form == "8-K":
                document = earnings_exhibit(cik, folder, args.email)
                if document is None:
                    continue  # an 8-K without an earnings release is not wanted
            url = f"{BASE}/Archives/edgar/data/{int(cik)}/{folder}/{document}"
            header = (f"SOURCE DOCUMENT - {match['title']} ({ticker}), {label}, filed "
                      f"{date}. Retrieved from SEC EDGAR: {url}\n"
                      f"This is the company own filed text, not a summary.")
            name = pattern.format(t=ticker, d=date)
            (out / name).write_text(to_text(get(url, args.email), header),
                                    encoding="utf-8")
            words = len((out / name).read_text(encoding="utf-8").split())
            print(f"  {form:8} {date}  -> {name} ({words:,} words)")
            written += 1

    print(f"\n{written} filing(s) in {out}")
    print("Published consensus and market data are not on EDGAR. Add them as a separate "
          "document if wanted; name it so it contains 'analyst' or 'consensus' and the "
          "ingester will classify it as sell-side research rather than public record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
