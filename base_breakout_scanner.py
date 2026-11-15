#!/usr/bin/env python3
"""
Long-consolidation breakout scanner (NSE small & micro caps, mcap >= 500 crore).

A stock makes the final list when:
  1. On the WEEKLY chart it has a long base: at least 26 weeks (6 months, up to 5 years) in which
     weekly closes stayed inside a 40% range, the base is flat (not a slow uptrend), and the top of
     the range (closing-basis resistance) was tested at least twice before.
  2. It is coming out of that base now:
       Near breakout      - this week's close is within 5% below the resistance, or
       Breakout confirmed - a completed week closed above the resistance in the last 4 weeks and
                            price is still holding above it (not more than 15% extended), or
       Breaking out       - the current, unfinished week is above the resistance.
  3. Unusual volume near the breakout level: a week in the last 4 with volume >= 3x the base's
     average weekly volume, or a day in the last 20 sessions with volume >= 5x its 50-day average.
  4. Regular concalls (Screener.in; scaled for companies listed recently).

Run:  python base_breakout_scanner.py      (options: --min-weeks 39 --max-width 30 ...)
Not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import elliott_scanner as es
import flag_pole_scanner as fp
import ipo_flag_scanner as ip

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
CACHE_DIR = HERE / "cache"


class P:                     # parameters (overridden from the command line)
    min_weeks = 26
    max_weeks = 260
    max_width = 0.40
    near = 0.05
    max_brk = 4
    max_ext = 0.15
    touch = 0.03
    wk_spike = 3.0
    day_spike = 5.0


def _clusters(idx: np.ndarray, gap=3) -> int:
    if len(idx) == 0:
        return 0
    return 1 + int(np.sum(np.diff(idx) > gap))


def find_base(w: pd.DataFrame, complete: bool) -> dict | None:
    C, H, L, V = (w[c].values.astype(float) for c in ("Close", "High", "Low", "Volume"))
    n = len(w)
    best = None
    for k in range(0, P.max_brk + 1):
        e = n - 1 - k                          # last bar of the base
        if e + 1 < P.min_weeks:
            break
        # grow the base backwards while the closing range stays inside max_width
        hi = lo = C[e]
        length = 0
        for i in range(e, max(-1, e - P.max_weeks), -1):
            nh, nl = max(hi, C[i]), min(lo, C[i])
            if nh / nl - 1 > P.max_width:
                break
            hi, lo, length = nh, nl, length + 1
        if length < P.min_weeks:
            continue
        found = None
        for length in range(length, P.min_weeks - 1, -1):   # trim the left edge until flat
            s = e - length + 1
            base = slice(s, e + 1)
            res, sup = C[base].max(), C[base].min()
            if k == 0:
                # still inside: current close near the top; resistance from the earlier bars
                res = C[s:e - 1].max()
                if not (res * (1 - P.near) <= C[-1] <= res):
                    continue
            else:
                b = e + 1
                if C[b] <= res:
                    continue                   # first bar after the base must close above
                if C[-1] < res * 0.97 or C[b:].min() < res * 0.95:
                    continue                   # fell back into the base: failed breakout
                if C[-1] > res * (1 + P.max_ext):
                    continue                   # already too extended
            x = np.arange(length, dtype=float)
            y = C[base]
            slope = np.cov(x, y, bias=True)[0, 1] / np.var(x)
            if abs(slope * length) > 0.6 * (res - sup):
                continue                       # trending, not a consolidation
            tests = _clusters(np.where(C[s:e - 1] >= res * (1 - P.touch))[0])
            if tests < 2:
                continue
            found = (s, length, res, sup, tests)
            break
        if not found:
            continue
        s, length, res, sup, tests = found
        base = slice(s, e + 1)
        status = ("Near breakout" if k == 0 else
                  "Breaking out (week not closed)" if (k == 1 and not complete) else
                  "Breakout confirmed")
        base_vol = np.nanmean(np.where(V[base] > 0, V[base], np.nan))
        lo_i = max(0, n - 4)
        rv = V[lo_i:] / base_vol if base_vol and np.isfinite(base_vol) else np.full(n - lo_i, np.nan)
        j = int(np.nanargmax(rv)) if np.isfinite(rv).any() else None
        cand = dict(k=k, s=s, e=e, length=length, res=float(res), sup=float(sup),
                    wick_high=float(H[base].max()), width=res / sup - 1, tests=tests, status=status,
                    wk_vol_x=float(rv[j]) if j is not None else np.nan,
                    wk_vol_i=(lo_i + j) if j is not None else None, brk_i=(e + 1) if k else None)
        key = lambda c: (c["k"] > 0, c["length"])      # a real breakout beats "near"
        if best is None or key(cand) > key(best):
            best = cand
    return best


def daily_spike(d: pd.DataFrame, sessions=20, avg=50):
    v = d["Volume"].astype(float).replace(0, np.nan)
    rv = (v / v.shift(1).rolling(avg, min_periods=20).mean()).values
    seg = rv[-sessions:]
    if not np.isfinite(seg).any():
        return np.nan, None
    i = len(rv) - sessions + int(np.nanargmax(seg))
    return float(rv[i]), d.index[i].date().isoformat()


def analyse(sym, daily, now) -> dict | None:
    daily = daily.dropna(subset=["High", "Low", "Close"])
    w = fp.resample(daily, "W-FRI")
    if len(w) < P.min_weeks + 2:
        return None
    b = find_base(w, ip.last_bar_complete(daily, "W", now))
    if not b:
        return None
    c = float(daily["Close"].iloc[-1])
    dx, dd = daily_spike(daily)
    idx = w.index
    d = lambda i: idx[i].date().isoformat() if i is not None else ""
    height = b["res"] - b["sup"]
    return {
        "symbol": sym, "price": round(c, 2), "status": b["status"],
        "base_weeks": b["length"], "base_months": round(b["length"] / 4.345, 1),
        "base_start": d(b["s"]), "base_end": d(b["e"]),
        "resistance": round(b["res"], 2), "wick_high": round(b["wick_high"], 2),
        "support": round(b["sup"], 2), "width_pct": round(b["width"] * 100, 1), "tests": b["tests"],
        "breakout_week": d(b["brk_i"]),
        "vs_res_pct": round((c / b["res"] - 1) * 100, 1),
        "target": round(b["res"] + height, 2), "to_target_pct": round(((b["res"] + height) / c - 1) * 100, 1),
        "stop": round(b["res"] * 0.93 if b["k"] else b["sup"] + 0.5 * height, 2),
        "wk_vol_x": round(b["wk_vol_x"], 2) if np.isfinite(b["wk_vol_x"]) else np.nan,
        "wk_vol_week": d(b["wk_vol_i"]),
        "day_vol_x": round(dx, 2) if np.isfinite(dx) else np.nan, "day_vol_date": dd or "",
        "vol_ok": bool((np.isfinite(b["wk_vol_x"]) and b["wk_vol_x"] >= P.wk_spike) or
                       (np.isfinite(dx) and dx >= P.day_spike)),
        "_w": w, "_b": b,
    }


# --------------------------------------------------------------------------------------
COLS = [
    ("symbol", "Symbol"), ("name", "Company"), ("mcap_cr", "Mcap (cr)"), ("price", "Price"),
    ("status", "Status"), ("breakout_week", "Breakout week"), ("vs_res_pct", "Price vs resistance %"),
    ("base_months", "Base length (months)"), ("base_start", "Base from"), ("width_pct", "Base range %"),
    ("tests", "Resistance tests"), ("resistance", "Resistance (closing)"), ("wick_high", "Base high (wick)"),
    ("support", "Base low (closing)"), ("concall_status", "Concalls"), ("concalls_24m", "Concalls (24m)"),
    ("last_concall", "Last concall"), ("wk_vol_x", "Weekly vol (x base avg)"), ("wk_vol_week", "Spike week"),
    ("day_vol_x", "Daily vol (x 50d avg)"), ("day_vol_date", "Spike day"),
    ("spike_deliv_pct", "Spike-day delivery %"), ("avg20_deliv_pct", "20-day avg delivery %"),
    ("target", "Measured target"), ("to_target_pct", "To target %"), ("stop", "Suggested stop"),
    ("pe", "Stock P/E"), ("profit_growth_ttm", "Profit growth TTM %"),
]
ORDER = {"Breakout confirmed": 0, "Breaking out (week not closed)": 1, "Near breakout": 2}


def frame(rows):
    df = pd.DataFrame([{k: r.get(k, np.nan) for k, _ in COLS} |
                       {"_v": r["vol_ok"], "_c": r.get("concall_status") == "Regular"} for r in rows])
    if df.empty:
        return df
    df["_o"] = df["status"].map(ORDER)
    return df.sort_values(["_o", "base_months"], ascending=[True, False])


def write_excel(df, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    hdr, hide = dict(COLS), ["_v", "_c", "_o"]
    sheets = {"Final list": df[df["_v"] & df["_c"]],
              "Volume ok, concalls not regular": df[df["_v"] & ~df["_c"]],
              "No unusual volume": df[~df["_v"]]}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, t in sheets.items():
            t.drop(columns=hide).rename(columns=hdr).to_excel(xw, sheet_name=name, index=False)
        pd.DataFrame({"Item": ["Base", "Resistance", "Status", "Unusual volume", "Measured target",
                               "Suggested stop", "Concalls", "Caveat"], "Meaning": [
            f"Weekly closes stayed within a {P.max_width:.0%} range for {P.min_weeks}+ weeks; flat, not trending.",
            f"Highest weekly close of the base (closing basis), tested at least twice (within {P.touch:.0%}). "
            "'Base high (wick)' is the highest intraweek high for reference.",
            f"Near breakout = within {P.near:.0%} below resistance. Breakout confirmed = a finished week closed "
            f"above it in the last {P.max_brk} weeks and price holds above it (max {P.max_ext:.0%} extended). "
            "Breaking out = the current unfinished week is above it.",
            f"A week in the last 4 with volume >= {P.wk_spike:g}x the base's average week, or a day in the "
            f"last 20 sessions >= {P.day_spike:g}x its 50-day average. Delivery % is for that day (NSE).",
            "Resistance + base height (classic range breakout projection).",
            "Breakouts: 7% below resistance (back inside the base = failed). Near breakout: middle of the base.",
            "Screener.in: 4+ calls in 24 months and one within 7 months (fewer needed for recent listings).",
            "Mechanical scan - check each chart. Not investment advice."]}
        ).to_excel(xw, sheet_name="How to read", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for j, cell in enumerate(ws[1], 1):
                ws.column_dimensions[get_column_letter(j)].width = {
                    "Company": 28, "Status": 22, "Concalls": 22, "Item": 18, "Meaning": 110}.get(cell.value, 12)
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions


def chart(r, w=340, h=150):
    wk, b = r["_w"], r["_b"]
    start = max(0, min(b["s"] - 8, len(wk) - 130))
    x = wk.iloc[start:]
    n = len(x)
    H, L, C, V = (x[c].values.astype(float) for c in ("High", "Low", "Close", "Volume"))
    lo, hi = float(L.min()), float(H.max())
    pad, ph, vh = 12, 100, 28
    X = lambda i: pad + (w - 2 * pad) * (i + 0.5) / n
    Y = lambda p: pad + ph * (hi - p) / max(hi - lo, 1e-9)
    s, e = b["s"] - start, b["e"] - start
    p = [f'<rect x="{X(s):.1f}" y="{Y(b["res"]):.1f}" width="{X(e) - X(s):.1f}" '
         f'height="{Y(b["sup"]) - Y(b["res"]):.1f}" class="bz"/>',
         f'<line x1="{X(s):.1f}" x2="{X(n - 1):.1f}" y1="{Y(b["res"]):.1f}" y2="{Y(b["res"]):.1f}" class="rl"/>']
    pts = " ".join(f"{X(i):.1f},{Y(C[i]):.1f}" for i in range(n))
    p.append(f'<polyline points="{pts}" class="px"/>')
    vmax = float(np.nanmax(V)) or 1.0
    by = pad + ph + 8 + vh
    hot = (b["wk_vol_i"] - start) if (b["wk_vol_i"] is not None and r["wk_vol_x"] >= P.wk_spike) else -1
    bw = max(0.8, (w - 2 * pad) / n * 0.7)
    for i in range(n):
        hh = vh * (V[i] / vmax if np.isfinite(V[i]) else 0)
        p.append(f'<rect x="{X(i) - bw / 2:.1f}" y="{by - hh:.1f}" width="{bw:.1f}" height="{hh:.1f}" '
                 f'class="{"vs" if i == hot else "vb"}"/>')
    return (f'<svg viewBox="0 0 {w} {by + 4:.0f}" class="ch" role="img" '
            f'aria-label="weekly chart with the base and resistance">{"".join(p)}</svg>')


def write_html(rows, path, title):
    fin = sorted([r for r in rows if r["vol_ok"] and r.get("concall_status") == "Regular"],
                 key=lambda r: (ORDER[r["status"]], -r["base_weeks"]))

    def f(v, pct=False, d=2):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:+.1f}%" if pct else f"{v:,.{d}f}"

    def card(r):
        o = ORDER[r["status"]]
        vol = []
        if np.isfinite(r.get("wk_vol_x", np.nan)):
            vol.append(f'week of {r["wk_vol_week"]}: {r["wk_vol_x"]:.1f}x the base average')
        if np.isfinite(r.get("day_vol_x", np.nan)):
            dl = (f', delivery {r["spike_deliv_pct"]:.0f}%' if np.isfinite(r.get("spike_deliv_pct", np.nan)) else "")
            vol.append(f'{r["day_vol_date"]}: {r["day_vol_x"]:.1f}x 50-day avg{dl}')
        return f"""
<article class="card" data-st="{o}">
 <header><div><a href="https://www.tradingview.com/chart/?symbol=NSE:{html.escape(r['symbol'])}&interval=W" target="_blank"
  rel="noopener">{html.escape(r['symbol'])}</a><span class="nm">{html.escape(str(r.get('name', '')))[:28]}</span></div>
  <span class="mc">{f(r.get('mcap_cr'), d=0)} cr</span></header>
 <div class="tags"><span class="st st-{o}">{html.escape(r['status'])}</span>
  <span class="cc">{r['base_months']:.0f}-month base &middot; {r['tests']} tests &middot; concalls {html.escape(str(r.get('concall_status', '')))}</span></div>
 {chart(r)}
 <dl><div><dt>Price</dt><dd>{f(r['price'])} <i>{f(r['vs_res_pct'], True)} vs res.</i></dd></div>
  <div><dt>Resistance</dt><dd>{f(r['resistance'])}</dd></div>
  <div><dt>Target</dt><dd>{f(r['target'])} <i>{f(r['to_target_pct'], True)}</i></dd></div>
  <div><dt>Stop</dt><dd>{f(r['stop'])}</dd></div>
  <div><dt>Base range</dt><dd>{r['width_pct']:.0f}%</dd></div><div><dt>P/E</dt><dd>{f(r.get('pe'), d=1)}</dd></div></dl>
 <p class="vl"><b>Volume</b> {' &middot; '.join(vol)}</p>
</article>"""

    cnt = {k: sum(1 for r in fin if ORDER[r["status"]] == v) for k, v in ORDER.items()}
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f6f2;--card:#fff;--tx:#1d1d1b;--mu:#6b6a65;--bd:#e3e1da;--ac:#1f4e78;--g:#1d7a4f;--gb:#e1f3e8;
--a:#8a5a07;--ab:#fbf0d9;--b:#185fa5;--bb:#e6f1fb;--bz:rgba(31,78,120,.10);--rl:#1d7a4f;--px:#4a4944;--vb:#c9c7bf;--vs:#d85a30}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191917;--card:#232321;--tx:#ecebe6;--mu:#a3a199;--bd:#3a3936;--ac:#85b7eb;
--g:#7fd1a5;--gb:#173b2a;--a:#efc06a;--ab:#3d2b0c;--b:#85b7eb;--bb:#132c46;--bz:rgba(133,183,235,.14);--rl:#7fd1a5;
--px:#d3d1c7;--vb:#55534d;--vs:#f0997b}}}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--tx)}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mu);margin:0 0 14px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}}
.chip{{border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:999px;padding:5px 12px;cursor:pointer;font:inherit}}
.chip.on{{border-color:var(--ac);color:var(--ac)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px}}
.card header{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
.card a{{color:var(--ac);font-weight:600;text-decoration:none;font-size:16px}}.nm{{color:var(--mu);margin-left:6px;font-size:12px}}
.mc{{font-size:12px;color:var(--mu)}}.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px}}
.st,.cc{{padding:2px 9px;border-radius:6px;font-size:12px;font-weight:600}}.st-0{{background:var(--gb);color:var(--g)}}
.st-1{{background:var(--ab);color:var(--a)}}.st-2{{background:var(--bb);color:var(--b)}}.cc{{border:1px solid var(--bd);color:var(--mu);font-weight:500}}
.ch{{width:100%;height:auto;display:block;margin:6px 0}}.bz{{fill:var(--bz)}}.rl{{stroke:var(--rl);stroke-dasharray:4 3;stroke-width:1}}
.px{{fill:none;stroke:var(--px);stroke-width:1.2}}.vb{{fill:var(--vb)}}.vs{{fill:var(--vs)}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;margin:6px 0}}dl div{{display:flex;justify-content:space-between;gap:6px}}
dt{{color:var(--mu);font-size:12px}}dd{{margin:0;font-variant-numeric:tabular-nums;font-size:13px}}dd i{{color:var(--mu);font-style:normal;font-size:11px}}
.vl{{font-size:12px;color:var(--mu);border-top:1px solid var(--bd);padding-top:6px;margin:6px 0 0}}.vl b{{color:var(--tx)}}
.note{{color:var(--mu);font-size:12px;margin-top:24px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">Small &amp; micro caps leaving a {P.min_weeks}+ week base, with unusual volume near the breakout level and regular
concalls. Shaded box = the base, dashed line = resistance (closing basis), orange bar = volume spike week.</p>
<div class="chips"><button class="chip on" data-f="*">All <b>{len(fin)}</b></button>
<button class="chip" data-f="0">Breakout confirmed <b>{cnt['Breakout confirmed']}</b></button>
<button class="chip" data-f="1">Breaking out <b>{cnt['Breaking out (week not closed)']}</b></button>
<button class="chip" data-f="2">Near breakout <b>{cnt['Near breakout']}</b></button></div>
<section class="grid">{''.join(card(r) for r in fin) or '<p>No stock passed every filter.</p>'}</section>
<p class="note">Mechanical scan. Check every chart by eye. Not investment advice.</p>
</main><script>
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===b));
const v=b.dataset.f;document.querySelectorAll('.card').forEach(k=>k.style.display=(v==='*'||k.dataset.st===v)?'':'none');}});
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Long-consolidation breakout scanner")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--universe")
    ap.add_argument("--min-mcap", type=float, default=500)
    ap.add_argument("--max-mcap", type=float, default=None)
    ap.add_argument("--exclude-top", type=int, default=250)
    ap.add_argument("--min-weeks", type=int, default=26, help="shortest base in weeks (default 26)")
    ap.add_argument("--max-width", type=float, default=40, help="max base range %% (default 40)")
    ap.add_argument("--near", type=float, default=5, help="near breakout = within this %% (default 5)")
    ap.add_argument("--weekly-spike", type=float, default=3.0)
    ap.add_argument("--daily-spike", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    P.min_weeks, P.max_width, P.near = a.min_weeks, a.max_width / 100, a.near / 100
    P.wk_spike, P.day_spike = a.weekly_spike, a.daily_spike
    now = dt.datetime.now()
    today = now.date()

    cfg = es.Config(min_mcap_cr=a.min_mcap, max_mcap_cr=a.max_mcap, exclude_top=a.exclude_top)
    uni = es.build_universe(a, cfg)
    if uni.empty:
        sys.exit("Universe is empty.")
    print(f"Downloading 6y of daily prices for {len(uni)} stocks ...")
    prices = es.download_prices(uni["symbol"].tolist(), 6)
    meta = uni.set_index("symbol")[["name", "mcap_cr"]].to_dict("index")
    rows = []
    for sym, df in prices.items():
        try:
            r = analyse(sym, df, now)
        except Exception as e:
            print(f"  {sym}: {e}")
            continue
        if r:
            r.update(meta.get(sym, {}))
            rows.append(r)
    vol = [r for r in rows if r["vol_ok"]]
    print(f"Coming out of a long base: {len(rows)} stocks; with unusual volume: {len(vol)}")

    syms = sorted({r["symbol"] for r in vol})
    if syms:
        comp = ip.company_data(syms)
        try:
            listed = ip.listing_dates()
        except Exception:
            listed = {}
        for r in vol:
            c = comp.get(r["symbol"], {})
            if c.get("error"):
                r["concall_status"] = "Unknown (Screener not reachable)"
            else:
                ld = listed.get(r["symbol"], dt.date(2000, 1, 1))
                r.update(ip.concall_check(c.get("concalls"), ld, today))
            r["pe"] = c.get("Stock P/E", np.nan)
            r["profit_growth_ttm"] = (c.get("profit_growth") or {}).get("TTM", np.nan)
        # delivery on the spike day and the last 20 sessions
        try:
            days = {d.date().isoformat() for d in prices[syms[0]].index[-20:]}
            days |= {r["day_vol_date"] for r in vol if r.get("day_vol_date")}
            bhav = es.load_delivery(days, CACHE_DIR / "bhav")
            recent = sorted(d for d in bhav)[-20:]
            for r in vol:
                s = r["symbol"]
                rec = bhav.get(r.get("day_vol_date"), {}).get(s)
                if rec and np.isfinite(rec[2]):
                    r["spike_deliv_pct"] = round(float(rec[2]), 1)
                pc = [bhav[d][s][2] for d in recent if s in bhav[d] and np.isfinite(bhav[d][s][2])]
                if pc:
                    r["avg20_deliv_pct"] = round(float(np.mean(pc)), 1)
        except Exception as e:
            print(f"  delivery data skipped ({e})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = frame(rows)
    if df.empty:
        print("No stock is coming out of a long base right now.")
        return
    xlsx, page = out / f"base_breakout_{today}.xlsx", out / f"base_breakout_{today}.html"
    write_excel(df, xlsx)
    write_html(rows, page, f"Long-base breakout scan - {today}")
    fin = df[df["_v"] & df["_c"]]
    print(f"\nDone. Final list: {len(fin)} stocks.\n  Excel : {xlsx}\n  Report: {page}")
    if len(fin):
        print(fin[["symbol", "status", "base_months", "resistance", "vs_res_pct", "wk_vol_x", "day_vol_x"]]
              .head(30).to_string(index=False))


if __name__ == "__main__":
    main()
