#!/usr/bin/env python3
"""
Flag & Pole scanner (weekly and monthly charts) for NSE small & micro caps.

A stock makes the list when:
  1. It is a small/micro cap with market cap >= 500 crore (same universe as the Elliott scanner).
  2. On the WEEKLY or MONTHLY chart it shows a bull flag:
       pole = a sharp rise (weekly >= 35% in 2-12 weeks, monthly >= 60% in 2-9 months),
       flag = a sideways / down-sloping consolidation after the pole top that holds at least
              50% of the pole and does not make a new high, still live today
              (in the flag, near its top line, or breaking out in the last 1-2 bars).
  3. There is an UNUSUAL VOLUME bar at the top of the pole or inside the flag
     (weekly >= 3x the 20-week average, monthly >= 2.5x the 12-month average).
  4. The company holds concalls regularly (Screener.in concall list: at least 4 calls in the
     last 24 months and the latest within 7 months).

Run:  python flag_pole_scanner.py
      python flag_pole_scanner.py --symbols NEOGEN ELECON
Output: output/flag_pole_<date>.xlsx and .html.  Not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import elliott_scanner as es

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
CACHE_DIR = HERE / "cache"


@dataclass
class TF:
    code: str
    name: str
    rule: str
    pole_min: int
    pole_max: int
    pole_gain: float
    flag_min: int
    flag_max: int
    max_retrace: float
    vol_avg: int
    vol_spike: float


TIMEFRAMES = {
    "W": TF("W", "Weekly", "W-FRI", 2, 12, 0.35, 3, 20, 0.50, 20, 3.0),
    "M": TF("M", "Monthly", "ME", 2, 9, 0.60, 2, 10, 0.50, 12, 2.5),
}


# --------------------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------------------
def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    try:
        out = df.resample(rule).agg(agg)
    except ValueError:                       # pandas < 2.2 uses "M" for month end
        out = df.resample(rule.replace("ME", "M")).agg(agg)
    return out.dropna(subset=["Close"])


# --------------------------------------------------------------------------------------
# Pattern detection
# --------------------------------------------------------------------------------------
def _line(y: np.ndarray):
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0, float(y[0]) if len(y) else np.nan
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def find_flag(bars: pd.DataFrame, tf: TF) -> dict | None:
    """Best live bull flag on these bars, or None."""
    H, L, C = bars["High"].values, bars["Low"].values, bars["Close"].values
    V = bars["Volume"].astype(float).values
    n = len(bars)
    if n < tf.pole_min + tf.flag_min + 3:
        return None
    best = None
    for brk in (0, 1, 2):                    # bars of breakout at the end
        e = n - 1 - brk                      # last bar of the flag
        for t in range(max(tf.pole_min, e - tf.flag_max), e - tf.flag_min + 1):
            top = H[t]
            flag = slice(t + 1, e + 1)
            if H[flag].max() > top * 1.02:   # flag must not make a new high
                continue
            s_lo = max(0, t - tf.pole_max)
            s_hi = t - tf.pole_min
            if s_hi < s_lo:
                continue
            win = L[s_lo:s_hi + 1]
            m = win.min()
            s = s_lo + int(np.where(win <= m * 1.03)[0][-1])   # last bar of the base
            base = L[s]
            if base <= 0 or L[s:t + 1].min() < base:
                continue
            gain = top / base - 1
            if gain < tf.pole_gain:
                continue
            if H[s:t + 1].max() > top:       # top of pole must be the pole's high
                continue
            height = top - base
            pole_len = t - s
            flag_len = e - t
            if flag_len > max(4 * pole_len, tf.flag_min + 2):
                continue
            if C[t] < base + 0.6 * height and C[min(t + 1, e)] < base + 0.6 * height:
                continue                     # pole must close strong
            f_low = L[flag].min()
            retr = (top - f_low) / height
            if retr > tf.max_retrace:
                continue
            slope, _ = _line(C[flag])
            if slope * flag_len > 0.15 * height:
                continue                     # flag must be flat or drift down
            # upper line of the flag (through its highs, starting at the pole top)
            hb, ha = _line(np.r_[top, H[flag]])
            upper = lambda i: ha + hb * (i - t)
            if brk == 0:
                if C[n - 1] > upper(n - 1) * 1.0 and C[n - 1] > H[flag].max():
                    continue                 # already broken out -> handled by brk>0
                status = "In flag"
                trig = max(upper(n), float(H[flag][-3:].max()) if flag_len >= 3 else top)
                trig = min(trig, top)
                if C[n - 1] >= trig * 0.95:
                    status = "Near breakout"
            else:
                b0 = e + 1
                if not (C[b0] > upper(b0) and C[b0] > H[flag].max() * 0.995):
                    continue
                if L[b0:].min() < f_low:
                    continue
                status = "Breaking out" if brk == 1 else "Broke out last bar"
                trig = float(max(upper(b0), H[flag].max()))
            # volume: bars at the pole top or inside the flag vs the average before each bar
            vs = pd.Series(V).replace(0, np.nan)
            avg = vs.shift(1).rolling(tf.vol_avg, min_periods=max(4, tf.vol_avg // 2)).mean()
            rv = (vs / avg).values
            w0, w1 = max(0, t - 1), e
            seg = rv[w0:w1 + 1]
            if np.isfinite(seg).any():
                k = w0 + int(np.nanargmax(seg))
                vx = float(rv[k])
                where = "Pole top" if k <= t else "In flag"
            else:
                k, vx, where = None, np.nan, ""
            quality = gain * (1 - retr) * (1.15 if status != "In flag" else 1.0)
            cand = dict(status=status, s=s, t=t, e=e, brk=brk, base=float(base), top=float(top),
                        gain=gain, pole_len=pole_len, flag_len=flag_len, flag_low=float(f_low),
                        flag_high=float(H[flag].max()), retrace=retr, trigger=float(trig),
                        hb=hb, ha=ha, vol_x=vx, vol_i=k, vol_where=where, quality=quality)
            if best is None or cand["quality"] > best["quality"]:
                best = cand
    return best


def analyse(sym: str, daily: pd.DataFrame, tfs: list[TF]) -> list[dict]:
    daily = daily.dropna(subset=["High", "Low", "Close"])
    rows = []
    for tf in tfs:
        bars = resample(daily, tf.rule)
        f = find_flag(bars, tf)
        if not f:
            continue
        idx = bars.index
        c = float(daily["Close"].iloc[-1])
        height = f["top"] - f["base"]
        target = f["trigger"] + height
        d = lambda i: idx[i].date().isoformat() if i is not None else ""
        rows.append({
            "symbol": sym, "tf": tf.name, "status": f["status"], "price": round(c, 2),
            "pole_start_date": d(f["s"]), "pole_start": round(f["base"], 2),
            "pole_top_date": d(f["t"]), "pole_top": round(f["top"], 2),
            "pole_gain_pct": round(f["gain"] * 100, 1), "pole_bars": f["pole_len"],
            "flag_bars": f["flag_len"], "flag_low": round(f["flag_low"], 2),
            "retrace_pct": round(f["retrace"] * 100, 1),
            "trigger": round(f["trigger"], 2),
            "to_trigger_pct": round((f["trigger"] / c - 1) * 100, 1),
            "target": round(target, 2), "to_target_pct": round((target / c - 1) * 100, 1),
            "invalidation": round(f["flag_low"], 2),
            "risk_pct": round((c / f["flag_low"] - 1) * 100, 1),
            "vol_x": round(f["vol_x"], 2) if np.isfinite(f["vol_x"]) else np.nan,
            "vol_date": d(f["vol_i"]), "vol_where": f["vol_where"],
            "vol_flag": bool(np.isfinite(f["vol_x"]) and f["vol_x"] >= tf.vol_spike),
            "vol_threshold": tf.vol_spike, "quality": round(f["quality"], 3),
            "_bars": bars, "_f": f, "_tf": tf,
        })
    return rows


# --------------------------------------------------------------------------------------
# Concalls (Screener.in company page, "Concalls" section)
# --------------------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
MONTH_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(20\d\d)\b")


def parse_concalls(page: str) -> list[str] | None:
    """Month-year of every concall listed on a Screener company page ('YYYY-MM'), newest first.
    None if the page has no concall section at all."""
    i = page.find("concalls")
    j = page.find(">Concalls<")
    if j == -1 and i == -1:
        return None
    start = j if j != -1 else i
    end_candidates = [page.find(m, start + 10) for m in ("</section>", 'class="documents ')]
    end_candidates = [x for x in end_candidates if x != -1]
    seg = page[start:min(end_candidates) if end_candidates else start + 30000]
    seen = []
    for m, y in MONTH_RE.findall(seg):
        k = f"{y}-{MONTHS[m[:3]]:02d}"
        if k not in seen:
            seen.append(k)
    return sorted(seen, reverse=True)


def concall_verdict(months: list[str] | None, today: dt.date, min_calls=4, window=24,
                    max_gap=7) -> dict:
    if months is None:
        return {"concall_status": "No concall section", "concalls_24m": 0, "last_concall": ""}
    cutoff = (today.year * 12 + today.month) - window
    recent = [m for m in months if int(m[:4]) * 12 + int(m[5:]) > cutoff]
    last = months[0] if months else ""
    gap = ((today.year * 12 + today.month) - (int(last[:4]) * 12 + int(last[5:]))) if last else 99
    ok = len(recent) >= min_calls and gap <= max_gap
    if not months:
        status = "No concalls listed"
    elif ok:
        status = "Regular"
    elif gap > max_gap:
        status = f"Stopped (last {gap} months ago)"
    else:
        status = "Irregular"
    return {"concall_status": status, "concalls_24m": len(recent), "last_concall": last}


def _fetch_screener(sym: str) -> str | None:
    import requests
    for url in (f"https://www.screener.in/company/{sym}/consolidated/",
                f"https://www.screener.in/company/{sym}/"):
        for attempt in range(3):
            try:
                r = requests.get(url, headers={"User-Agent": es.UA["User-Agent"]}, timeout=30)
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                if r.status_code == 200:
                    return r.text
                break                        # 404 etc: try the next URL
            except Exception:
                time.sleep(3)
    return None


def check_concalls(symbols: list[str], cache_file: Path, today: dt.date, workers=2) -> dict:
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            cache = {}
    out, todo = {}, []
    for s in symbols:
        c = cache.get(s)
        if c and (time.time() - c.get("ts", 0)) < 7 * 86400:
            out[s] = c
        else:
            todo.append(s)
    if todo:
        print(f"Checking concall history on Screener.in for {len(todo)} companies ...")

    def job(s):
        page = _fetch_screener(s)
        time.sleep(1.2)
        if page is None:
            return s, {"months": "fetch-failed"}
        return s, {"months": parse_concalls(page)}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(job, s) for s in todo]):
            s, res = f.result()
            res["ts"] = time.time()
            if res["months"] != "fetch-failed":
                cache[s] = res
            out[s] = res
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"  concalls: {done}/{len(todo)}", flush=True)
    try:
        cache_file.write_text(json.dumps(cache))
    except Exception:
        pass
    verdicts = {}
    for s, res in out.items():
        if res.get("months") == "fetch-failed":
            verdicts[s] = {"concall_status": "Unknown (Screener not reachable)",
                           "concalls_24m": np.nan, "last_concall": ""}
        else:
            verdicts[s] = concall_verdict(res.get("months"), today)
    return verdicts


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
COLS = [
    ("symbol", "Symbol"), ("name", "Company"), ("mcap_cr", "Mcap (cr)"), ("price", "Price"),
    ("tf", "Timeframe"), ("status", "Status"), ("concall_status", "Concalls"),
    ("concalls_24m", "Concalls in 24m"), ("last_concall", "Last concall"),
    ("vol_x", "Volume spike (x avg)"), ("vol_where", "Spike at"), ("vol_date", "Spike bar"),
    ("trigger", "Breakout level"), ("to_trigger_pct", "To breakout %"),
    ("target", "Measured target"), ("to_target_pct", "To target %"),
    ("invalidation", "Flag low (stop)"), ("risk_pct", "Risk %"),
    ("pole_gain_pct", "Pole gain %"), ("pole_bars", "Pole bars"),
    ("pole_start_date", "Pole start"), ("pole_start", "Pole start price"),
    ("pole_top_date", "Pole top"), ("pole_top", "Pole top price"),
    ("flag_bars", "Flag bars"), ("retrace_pct", "Flag retrace of pole %"),
    ("last_deliv_pct", "Latest delivery %"), ("avg20_deliv_pct", "20-day avg delivery %"),
    ("quality", "Pattern quality"),
]
STATUS_ORDER = {"Breaking out": 0, "Near breakout": 1, "Broke out last bar": 2, "In flag": 3}


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{k: r.get(k, np.nan) for k, _ in COLS} | {
        "_vol": r.get("vol_flag", False),
        "_cc": r.get("concall_status") == "Regular"} for r in rows])
    if df.empty:
        return df
    df["_st"] = df["status"].map(STATUS_ORDER).fillna(9)
    return df.sort_values(["_st", "vol_x"], ascending=[True, False])


def write_excel(df: pd.DataFrame, path: Path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    header = dict(COLS)
    hide = ["_vol", "_cc", "_st"]
    sheets = {
        "Final list": df[df["_vol"] & df["_cc"]],
        "Pattern+volume, concall not ok": df[df["_vol"] & ~df["_cc"]],
        "Pattern only (no vol spike)": df[~df["_vol"]],
    }
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, t in sheets.items():
            t.drop(columns=hide).rename(columns=header).to_excel(xw, sheet_name=name, index=False)
        pd.DataFrame({"Item": [
            "Pole", "Flag", "Status", "Volume spike", "Breakout level", "Measured target",
            "Flag low (stop)", "Concalls", "Timeframes", "Caveat"], "Meaning": [
            "Weekly: rise of 35%+ in 2-12 weeks. Monthly: 60%+ in 2-9 months. Must close strong.",
            "Sideways or down-sloping consolidation after the pole top; holds at least half the "
            "pole, no new high, lasts at most 4x the pole length.",
            "In flag / Near breakout (within 5% of the flag's top line) / Breaking out (this bar "
            "closed above the flag) / Broke out last bar.",
            "Largest bar volume at the pole top or inside the flag, versus the average of the "
            "previous 20 weeks (weekly, needs 3x) or 12 months (monthly, needs 2.5x).",
            "Flag's falling top line projected to the next bar (or the flag high); a close above "
            "it is the breakout.",
            "Breakout level + pole height (classic flag measured move).",
            "Lowest low of the flag - the pattern fails below it.",
            "Screener.in concall list: Regular = 4+ calls in the last 24 months and the latest "
            "within 7 months.",
            "The current week / month is still forming, so its volume is partial.",
            "Mechanical pattern scan. Check every chart by eye. Not investment advice."]}
        ).to_excel(xw, sheet_name="How to read", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for j, cell in enumerate(ws[1], start=1):
                ws.column_dimensions[get_column_letter(j)].width = {
                    "Company": 28, "Concalls": 24, "Status": 16, "Item": 18, "Meaning": 110
                }.get(cell.value, 12)
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions


def chart_svg(r: dict, w=330, h=150) -> str:
    bars, f = r["_bars"], r["_f"]
    show = 60
    start = max(0, min(f["s"] - 6, len(bars) - show))
    b = bars.iloc[start:]
    n = len(b)
    if n < 3:
        return ""
    H, L, C, V = b["High"].values, b["Low"].values, b["Close"].values, b["Volume"].values
    lo, hi = float(L.min()), float(H.max())
    pad, ph, vh = 14, 98, 30
    X = lambda i: pad + (w - 2 * pad) * (i + 0.5) / n
    Y = lambda p: pad + ph * (hi - p) / max(hi - lo, 1e-9)
    bw = max(1.0, (w - 2 * pad) / n * 0.6)
    parts = []
    s, t, e = f["s"] - start, f["t"] - start, f["e"] - start
    parts.append(f'<rect x="{X(t):.1f}" y="{Y(f["flag_high"]):.1f}" width="{X(e) - X(t):.1f}" '
                 f'height="{Y(f["flag_low"]) - Y(f["flag_high"]):.1f}" class="fz"/>')
    for i in range(n):
        cls = "pc" if s <= i <= t else "cd"
        parts.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(H[i]):.1f}" y2="{Y(L[i]):.1f}" class="{cls}"/>')
        parts.append(f'<line x1="{X(i) - bw / 2:.1f}" x2="{X(i) + bw / 2:.1f}" y1="{Y(C[i]):.1f}" '
                     f'y2="{Y(C[i]):.1f}" class="{cls}"/>')
    y0 = min(max(f["ha"], lo), hi)
    y1 = min(max(r["trigger"], lo), hi)
    parts.append(f'<line x1="{X(t):.1f}" x2="{X(n - 1):.1f}" y1="{Y(y0):.1f}" y2="{Y(y1):.1f}" class="tg"/>')
    vmax = float(np.nanmax(V)) or 1.0
    base_y = pad + ph + 8 + vh
    vi = (f["vol_i"] - start) if f.get("vol_i") is not None else -1
    for i in range(n):
        hh = vh * float(V[i]) / vmax
        cls = "vs" if (i == vi and r.get("vol_flag")) else "vb"
        parts.append(f'<rect x="{X(i) - bw / 2:.1f}" y="{base_y - hh:.1f}" width="{bw:.1f}" '
                     f'height="{hh:.1f}" class="{cls}"/>')
    return (f'<svg viewBox="0 0 {w} {base_y + 4:.0f}" class="ch" role="img" '
            f'aria-label="{html.escape(r["tf"])} chart with pole, flag and volume">{"".join(parts)}</svg>')


def write_html(rows: list[dict], path: Path, title: str):
    final = [r for r in rows if r.get("vol_flag") and r.get("concall_status") == "Regular"]
    other = [r for r in rows if r.get("vol_flag") and r.get("concall_status") != "Regular"]
    key = lambda r: (STATUS_ORDER.get(r["status"], 9), -(r.get("vol_x") or 0))
    final.sort(key=key)
    other.sort(key=key)

    def f(v, pct=False):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:+.1f}%" if pct else f"{v:,.2f}"

    def card(r):
        mc = r.get("mcap_cr")
        mc_s = f"{mc:,.0f} cr" if isinstance(mc, (int, float)) and np.isfinite(mc) else ""
        dl = ""
        if r.get("last_deliv_pct") is not None and np.isfinite(r.get("last_deliv_pct", np.nan)):
            dl = (f'<div><dt>Delivery</dt><dd>{r["last_deliv_pct"]:.0f}% '
                  f'<i>20d {r.get("avg20_deliv_pct", np.nan):.0f}%</i></dd></div>')
        return f"""
<article class="card" data-tf="{r['tf']}" data-st="{html.escape(r['status'])}">
 <header><div><a href="https://www.tradingview.com/chart/?symbol=NSE:{html.escape(r['symbol'])}&interval={'W' if r['tf'] == 'Weekly' else 'M'}"
   target="_blank" rel="noopener">{html.escape(r['symbol'])}</a>
   <span class="nm">{html.escape(str(r.get('name', '')))[:30]}</span></div>
   <span class="tf">{r['tf']}</span></header>
 <div class="tags"><span class="st st-{STATUS_ORDER.get(r['status'], 9)}">{html.escape(r['status'])}</span>
   <span class="cc">{html.escape(str(r.get('concall_status', '')))}{(' &middot; ' + str(int(r['concalls_24m'])) + ' in 24m') if isinstance(r.get('concalls_24m'), (int, float)) and np.isfinite(r.get('concalls_24m')) else ''}</span></div>
 <p class="dt">Pole +{r['pole_gain_pct']:.0f}% in {r['pole_bars']} bars to {f(r['pole_top'])} ({r['pole_top_date']}),
   flag {r['flag_bars']} bars holding {100 - r['retrace_pct']:.0f}% of it.
   Volume {r['vol_x']:.1f}x avg {r['vol_where'].lower()} ({r['vol_date']}).</p>
 {chart_svg(r)}
 <dl>
  <div><dt>Price</dt><dd>{f(r['price'])}</dd></div><div><dt>Mcap</dt><dd>{mc_s}</dd></div>
  <div><dt>Breakout</dt><dd>{f(r['trigger'])} <i>{f(r['to_trigger_pct'], True)}</i></dd></div>
  <div><dt>Target</dt><dd>{f(r['target'])} <i>{f(r['to_target_pct'], True)}</i></dd></div>
  <div><dt>Flag low</dt><dd>{f(r['invalidation'])} <i>{f(-r['risk_pct'], True)}</i></dd></div>
  {dl}
 </dl>
</article>"""

    tf_counts = {k: sum(1 for r in final if r["tf"] == k) for k in ("Weekly", "Monthly")}
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f6f2;--card:#fff;--tx:#1d1d1b;--mu:#6b6a65;--bd:#e3e1da;--ac:#1f4e78;--g:#1d7a4f;--gb:#e1f3e8;
--a:#8a5a07;--ab:#fbf0d9;--b:#185fa5;--bb:#e6f1fb;--pole:#1d7a4f;--fz:rgba(31,78,120,.10);--vb:#c9c7bf;--vs:#d85a30}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191917;--card:#232321;--tx:#ecebe6;--mu:#a3a199;--bd:#3a3936;
--ac:#85b7eb;--g:#7fd1a5;--gb:#173b2a;--a:#efc06a;--ab:#3d2b0c;--b:#85b7eb;--bb:#132c46;--pole:#7fd1a5;
--fz:rgba(133,183,235,.14);--vb:#55534d;--vs:#f0997b}}}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--tx)}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:28px 0 10px}}
.sub{{color:var(--mu);margin:0 0 14px}}.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}}
.chip{{border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:999px;padding:5px 12px;cursor:pointer;font:inherit}}
.chip.on{{border-color:var(--ac);color:var(--ac)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px}}
.card header{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
.card a{{color:var(--ac);font-weight:600;text-decoration:none;font-size:16px}}.nm{{color:var(--mu);margin-left:6px;font-size:12px}}
.tf{{font-size:12px;border:1px solid var(--bd);border-radius:6px;padding:0 7px}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px}}.st,.cc{{padding:2px 9px;border-radius:6px;font-size:12px;font-weight:600}}
.st-0,.st-2{{background:var(--gb);color:var(--g)}}.st-1{{background:var(--ab);color:var(--a)}}.st-3{{background:var(--bb);color:var(--b)}}
.cc{{background:transparent;border:1px solid var(--bd);color:var(--mu);font-weight:500}}
.dt{{margin:2px 0 6px;color:var(--mu);font-size:13px}}
.ch{{width:100%;height:auto;display:block;margin:4px 0}}.cd{{stroke:var(--mu);stroke-width:1}}.pc{{stroke:var(--pole);stroke-width:1.4}}
.fz{{fill:var(--fz)}}.tg{{stroke:var(--g);stroke-dasharray:3 3;stroke-width:.9}}.vb{{fill:var(--vb)}}.vs{{fill:var(--vs)}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;margin:6px 0}}dl div{{display:flex;justify-content:space-between;gap:6px}}
dt{{color:var(--mu);font-size:12px}}dd{{margin:0;font-variant-numeric:tabular-nums;font-size:13px}}dd i{{color:var(--mu);font-style:normal;font-size:11px}}
.note{{color:var(--mu);font-size:12px;margin-top:24px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">Bull flag on the weekly or monthly chart + unusual volume at the pole top or in the flag
+ regular concalls. Green = pole, shaded box = flag, dashed = breakout level, orange bar = volume spike.</p>
<div class="chips"><button class="chip on" data-f="*">All <b>{len(final)}</b></button>
<button class="chip" data-f="Weekly">Weekly <b>{tf_counts['Weekly']}</b></button>
<button class="chip" data-f="Monthly">Monthly <b>{tf_counts['Monthly']}</b></button>
<button class="chip" data-f="Near breakout">Near breakout</button><button class="chip" data-f="Breaking out">Breaking out</button></div>
<section class="grid" id="final">{''.join(card(r) for r in final) or '<p>No stocks passed all three filters today.</p>'}</section>
<h2>Pattern + volume, but concalls not regular or not verified ({len(other)})</h2>
<section class="grid">{''.join(card(r) for r in other) or '<p>None.</p>'}</section>
<p class="note">Mechanical pattern scan, not investment advice. Confirm each chart and the concall record before acting.</p>
</main><script>
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{{
 document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===b));const v=b.dataset.f;
 document.querySelectorAll('#final .card').forEach(k=>k.style.display=(v==='*'||k.dataset.tf===v||k.dataset.st===v)?'':'none');}});
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Flag & pole scanner (weekly/monthly) for NSE small & micro caps")
    ap.add_argument("--symbols", nargs="*", help="scan only these NSE symbols (no mcap filter)")
    ap.add_argument("--universe", help="CSV with a Symbol/NSE Code column and optional Market Cap")
    ap.add_argument("--min-mcap", type=float, default=500)
    ap.add_argument("--max-mcap", type=float, default=None)
    ap.add_argument("--exclude-top", type=int, default=250)
    ap.add_argument("--timeframes", default="W,M", help="W, M or W,M (default)")
    ap.add_argument("--weekly-spike", type=float, default=3.0, help="weekly volume spike, x 20-week avg")
    ap.add_argument("--monthly-spike", type=float, default=2.5, help="monthly volume spike, x 12-month avg")
    ap.add_argument("--min-concalls", type=int, default=4, help="concalls needed in the last 24 months")
    ap.add_argument("--skip-concalls", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)

    TIMEFRAMES["W"].vol_spike = a.weekly_spike
    TIMEFRAMES["M"].vol_spike = a.monthly_spike
    tfs = [TIMEFRAMES[x.strip().upper()] for x in a.timeframes.split(",") if x.strip()]
    cfg = es.Config(min_mcap_cr=a.min_mcap, max_mcap_cr=a.max_mcap, exclude_top=a.exclude_top)
    uni = es.build_universe(a, cfg)
    if uni.empty:
        sys.exit("Universe is empty.")
    print(f"Downloading 7y of daily prices for {len(uni)} stocks ...")
    prices = es.download_prices(uni["symbol"].tolist(), 7)
    meta = uni.set_index("symbol")[["name", "mcap_cr"]].to_dict("index")

    rows = []
    for sym, df in prices.items():
        try:
            for r in analyse(sym, df, tfs):
                r.update(meta.get(sym, {}))
                rows.append(r)
        except Exception as e:
            print(f"  {sym}: {e}")
    n_pat = len({r["symbol"] for r in rows})
    vol_syms = sorted({r["symbol"] for r in rows if r["vol_flag"]})
    print(f"Live flags: {len(rows)} ({n_pat} stocks); with a volume spike: {len(vol_syms)} stocks")

    today = dt.date.today()
    if not a.skip_concalls and vol_syms:
        CACHE_DIR.mkdir(exist_ok=True)
        verdicts = check_concalls(vol_syms, CACHE_DIR / "concalls.json", today)
        for r in rows:
            v = verdicts.get(r["symbol"])
            if v:
                r.update(v)
                if a.min_concalls != 4 and r["concall_status"] in ("Regular", "Irregular"):
                    r["concall_status"] = ("Regular" if r["concalls_24m"] >= a.min_concalls
                                           else "Irregular")
    # delivery % for the final candidates (uses the Elliott scanner's NSE file cache)
    try:
        cand = [r for r in rows if r["vol_flag"]]
        if cand:
            days = [d.date().isoformat() for d in prices[cand[0]["symbol"]].index[-20:]]
            bhav = es.load_delivery(set(days), CACHE_DIR / "bhav")
            if bhav:
                last = max(bhav)
                for r in cand:
                    s = r["symbol"]
                    rec = bhav[last].get(s)
                    if rec and np.isfinite(rec[2]):
                        r["last_deliv_pct"] = round(float(rec[2]), 1)
                    p = [bhav[d][s][2] for d in days if d in bhav and s in bhav[d]
                         and np.isfinite(bhav[d][s][2])]
                    if p:
                        r["avg20_deliv_pct"] = round(float(np.mean(p)), 1)
    except Exception as e:
        print(f"  delivery data skipped ({e})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = today.isoformat()
    frame = to_frame(rows)
    xlsx, page = out / f"flag_pole_{stamp}.xlsx", out / f"flag_pole_{stamp}.html"
    if frame.empty:
        print("No live flags found.")
        return
    write_excel(frame, xlsx)
    write_html(rows, page, f"Flag & pole scan - {stamp}")
    fin = frame[frame["_vol"] & frame["_cc"]]
    print(f"\nDone. Final list: {len(fin)} (flag + volume spike + regular concalls)")
    print(f"  Excel : {xlsx}\n  Report: {page}")
    if len(fin):
        print(fin[["symbol", "tf", "status", "vol_x", "trigger", "target", "invalidation"]]
              .head(20).to_string(index=False))


if __name__ == "__main__":
    main()
