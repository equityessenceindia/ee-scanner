"""
Concall key points without any AI service (free, runs in the daily GitHub job).

1. Results snapshot from the Screener quarterly table: revenue, operating margin, net profit, EPS
   versus the same quarter a year ago.
2. "What management said": the transcript PDF (BSE link from Screener) is read with pypdf and the
   sentences that carry numbers about guidance, order book, capex/capacity and margins are picked
   out, management remarks first. Safe-harbour boilerplate and operator lines are dropped.
The result is written in the style of a short Substack note: headline, numbers, quotes, links.
"""
from __future__ import annotations

import io
import re

import numpy as np

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Referer": "https://www.bseindia.com/"}

# ------------------------------------------------------------------ Screener concall links
LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.S)
MON_RE = re.compile(r">\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(20\d\d)\s*<")
MON = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def concall_docs(page: str) -> list[dict]:
    """[{month:'2026-11', label:'Nov 2026', transcript, ppt, rec}] newest first, from a Screener page."""
    i = page.find('class="documents concalls')
    if i == -1:
        return []
    j = page.find("</ul>", i)
    seg = page[i:j if j != -1 else i + 40000]
    out = []
    for li in LI_RE.findall(seg):
        m = MON_RE.search(li)
        if not m:
            continue
        d = {"month": f"{m.group(2)}-{MON[m.group(1)]:02d}", "label": f"{m.group(1)} {m.group(2)}"}
        for href, text in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>\s*([A-Za-z ]+?)\s*</a>', li, re.S):
            t = text.strip().lower()
            if t.startswith("transcript"):
                d["transcript"] = href
            elif t == "ppt":
                d["ppt"] = href
            elif t == "rec":
                d["rec"] = href
        out.append(d)
    return out


# ------------------------------------------------------------------ results snapshot
def _f(x):
    return x if (x is not None and isinstance(x, (int, float)) and np.isfinite(x)) else None


def results_snapshot(q: dict) -> dict:
    """Latest quarter vs the same quarter last year from Screener's quarterly table."""
    labels = q.get("q_labels") or []
    if len(labels) < 5:
        return {}
    out = {"quarter": labels[-1], "lines": [], "yoy": {}}

    def pair(key):
        v = q.get(key) or []
        if len(v) < 5:
            return None, None
        return _f(v[-1]), _f(v[-5])

    s, s0 = pair("sales")
    if s is not None and s0:
        g = (s / s0 - 1) * 100
        out["yoy"]["sales"] = round(g, 1)
        out["lines"].append(f"Revenue Rs {s:,.0f} cr, {g:+.0f}% YoY")
    o, o0 = pair("opm")
    if o is not None and o0 is not None:
        out["lines"].append(f"Operating margin {o:.1f}% (vs {o0:.1f}% a year ago, {o - o0:+.1f} pts)")
        out["yoy"]["opm_pts"] = round(o - o0, 1)
    p, p0 = pair("np")
    if p is not None and p0 is not None:
        if p0 > 0:
            g = (p / p0 - 1) * 100
            out["yoy"]["profit"] = round(g, 1)
            out["lines"].append(f"Net profit Rs {p:,.0f} cr, {g:+.0f}% YoY")
        else:
            out["lines"].append(f"Net profit Rs {p:,.0f} cr (Rs {p0:,.0f} cr a year ago)")
    e, e0 = pair("eps")
    if e is not None:
        out["lines"].append(f"EPS Rs {e:.2f}" + (f" (Rs {e0:.2f} a year ago)" if e0 is not None else ""))
    return out


# ------------------------------------------------------------------ transcript reading
TOPICS = [   # checked in this order; the first topic that matches a sentence owns it
    ("Orders", r"\border ?book\b|\border inflow|\border pipeline|\bbacklog\b|\bL1\b|\bbid pipeline"),
    ("Capex & capacity", r"\bcapex\b|\bcapacity\b|\bexpansion\b|\bnew plant\b|\bcommission|\butili[sz]ation\b|\bgreenfield\b|\bbrownfield\b"),
    ("Guidance", r"\bguidance\b|\bguide\b|\bguiding\b|\boutlook\b|\bexpect(?:ing|ed)?\b|\btarget(?:ing)?\b|"
                 r"\baim(?:ing)?\b|\bgoing forward\b|\bnext (?:year|financial year|fiscal)\b"),
    ("Margins", r"\bmargin|\bebitda\b|\bgross profit\b|\braw material|\bpricing\b"),
    ("Demand & growth", r"\bgrowth\b|\bgr[eo]w\b|\bvolumes?\b|\bdemand\b|\bmarket share\b|\bexports?\b|\brevenue\b"),
]
FWD = re.compile(r"\bFY ?2[7-9]\b|\bFY ?3[0-2]\b|\bwill\b|\bexpect|\btarget|\bguid", re.I)
NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent|per cent|crore|crores|cr\b|bps|basis points|x\b|times|billion|million|mn\b|lakh|MW|GW|tonnes|TPA|MTPA)", re.I)
BOILER = re.compile(r"forward[- ]looking|safe harbo|disclaimer|ladies and gentlemen|thank you|"
                    r"press star|question[- ]and[- ]answer|operator|moderator|good (?:morning|afternoon|evening)|"
                    r"welcome to|conference call|joining us|next question|line of", re.I)
QUESTION_HINT = re.compile(r"\?\s*$|^(?:my|so my|just|could you|can you|what is|what was|how do|how much|sir,? )", re.I)


def pdf_text(data: bytes, max_pages: int = 40) -> str:
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(data))
    parts = []
    for p in r.pages[:max_pages]:
        try:
            parts.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _sentences(text: str) -> list[str]:
    t = re.sub(r"-\n", "", text)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"Page \d+ of \d+", " ", t)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", t)
    return [p.strip() for p in parts if 40 <= len(p.strip()) <= 420]


def key_points(text: str, per_topic: int = 2, total: int = 7) -> dict:
    """Pick numeric, forward-looking management sentences, grouped by topic."""
    sents = _sentences(text)
    n = len(sents)
    picked, seen = {}, set()
    scored = []
    for i, s in enumerate(sents):
        if BOILER.search(s) or QUESTION_HINT.search(s):
            continue
        nums = len(NUM_RE.findall(s))
        if not nums:
            continue
        for topic, pat in TOPICS:
            hits = len(re.findall(pat, s, re.I))
            if not hits:
                continue
            score = hits * 2 + min(nums, 3)
            if topic == "Guidance":
                score += 3
            if FWD.search(s):          # forward-looking statements matter most
                score += 2
            if i < n * 0.45:           # opening remarks are management's prepared view
                score += 1
            scored.append((score, i, topic, s))
            break
    scored.sort(key=lambda x: (-x[0], x[1]))
    count = 0
    for score, i, topic, s in scored:
        key = re.sub(r"\W+", "", s.lower())[:80]
        if key in seen or len(picked.get(topic, [])) >= per_topic:
            continue
        seen.add(key)
        s = s if len(s) <= 300 else s[:297].rsplit(" ", 1)[0] + " ..."
        picked.setdefault(topic, []).append((i, s))
        count += 1
        if count >= total:
            break
    return {t: [s for _, s in sorted(v)] for t, v in picked.items()}


def fetch_transcript_points(url: str, timeout: int = 45) -> dict:
    import requests
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError("not a PDF")
    return key_points(pdf_text(r.content))


# ------------------------------------------------------------------ the note
def headline(snap: dict, points: dict) -> str:
    y = snap.get("yoy", {})
    bits = []
    if "sales" in y:
        bits.append(f"revenue {y['sales']:+.0f}%")
    if "profit" in y:
        bits.append(f"profit {y['profit']:+.0f}%")
    if "opm_pts" in y:
        bits.append(f"margin {y['opm_pts']:+.1f} pts")
    h = (snap.get("quarter", "Latest quarter") + ": " + ", ".join(bits) + " YoY") if bits else ""
    tone = ""
    g = " ".join(points.get("Guidance", [])).lower()
    if re.search(r"\b(raise|upgrad|higher than|ahead of|maintain|reiterat|confident)\w*", g):
        tone = "Management sounds confident on guidance."
    elif re.search(r"\b(lower|cut|revis\w* down|muted|soft|slowdown|challenging|headwind)\w*", g):
        tone = "Management flagged a softer outlook."
    return (h + (" " + tone if tone else "")).strip()
