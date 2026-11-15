#!/usr/bin/env python3
"""
Minervini Stage 2 scanner (NSE small & micro caps, mcap >= 500 crore, outside the top 250).

Timeframe: DAILY - Minervini's Trend Template is defined on the 50/150/200-day moving averages
(the 150/200-day lines are his stand-ins for the 30/40-week averages of Weinstein's stage analysis).

Trend Template (all 8 must hold = Stage 2):
  1. Price above the 150-day and 200-day MA
  2. 150-day MA above the 200-day MA
  3. 200-day MA trending up for at least 1 month (months rising reported, 4-5 is ideal)
  4. 50-day MA above the 150-day and 200-day MA
  5. Price above the 50-day MA
  6. Price at least 30% above its 52-week low
  7. Price within 25% of its 52-week high
  8. Relative Strength rating >= 70 (IBD-style: 40% 3-month + 20% each 6/9/12-month
     performance, ranked 1-99 against the whole small/micro universe)
"Stage 2 started" = the template turned true within the last 30 sessions.
"Near stage 2"   = 7 of the 8 hold (one missing).

Also checked (Think & Trade like a Champion):
  * VCP - successive pullbacks getting smaller (e.g. 22% -> 12% -> 6%) with volume drying up in
    the last one; the pivot is the high of the last contraction.
  * Code 33 - quarterly EPS growth, sales growth and operating margin all accelerating for the
    last 3 quarters (year-on-year), from Screener.in.
Kept from the other scanners: unusual volume (a day >= 5x its 50-day average in the last 20
sessions, or a week >= 3x its 20-week average in the last 4) with NSE delivery %, regular
concalls, and forward P/E compression / forward PEG for the next 3 financial years.

Run:  python stage2_scanner.py
Not investment advice.
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
from pathlib import Path

import numpy as np
import pandas as pd

import elliott_scanner as es
import flag_pole_scanner as fp
import ipo_flag_scanner as ip
import stage2_maturity as sm

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
CACHE_DIR = HERE / "cache"

CRITERIA = ["Price > 150 & 200-DMA", "150-DMA > 200-DMA", "200-DMA rising 1M+", "50-DMA > 150 & 200-DMA",
            "Price > 50-DMA", ">= 30% above 52w low", "Within 25% of 52w high", "RS rating >= 70"]


# --------------------------------------------------------------------------------------
# Trend template (vectorised over the whole universe)
# --------------------------------------------------------------------------------------
def rs_rating(C: pd.DataFrame) -> pd.DataFrame:
    perf = (0.4 * C.pct_change(63, fill_method=None) + 0.2 * C.pct_change(126, fill_method=None)
            + 0.2 * C.pct_change(189, fill_method=None) + 0.2 * C.pct_change(252, fill_method=None))
    return (perf.rank(axis=1, pct=True) * 98 + 1).round(0)


def template(C: pd.DataFrame, H: pd.DataFrame, L: pd.DataFrame, rs: pd.DataFrame, min_rs=70):
    s50 = C.rolling(50, min_periods=50).mean()
    s150 = C.rolling(150, min_periods=150).mean()
    s200 = C.rolling(200, min_periods=200).mean()
    lo52 = L.rolling(252, min_periods=200).min()
    hi52 = H.rolling(252, min_periods=200).max()
    crit = [
        (C > s150) & (C > s200),
        s150 > s200,
        s200 > s200.shift(21),
        (s50 > s150) & (s50 > s200),
        C > s50,
        C >= 1.30 * lo52,
        C >= 0.75 * hi52,
        rs >= min_rs,
    ]
    return crit, dict(s50=s50, s150=s150, s200=s200, lo52=lo52, hi52=hi52)


def stage2_run(p: np.ndarray, gap: int = 10) -> int:
    """Sessions since stage 2 began, ignoring short lapses (<= gap sessions) in the template."""
    i, start, miss = len(p) - 1, len(p) - 1, 0
    while i >= 0:
        if p[i]:
            start, miss = i, 0
        else:
            miss += 1
            if miss > gap:
                break
        i -= 1
    return len(p) - start


def months_rising(s200: pd.Series) -> int:
    k = 0
    v = s200.dropna().values
    while k < 12 and len(v) > 21 * (k + 1) and v[-1 - 21 * k] > v[-1 - 21 * (k + 1)]:
        k += 1
    return k


# --------------------------------------------------------------------------------------
# VCP
# --------------------------------------------------------------------------------------
def swings(H, L, pct):
    """Alternating swing highs/lows (index, price, 'H'/'L') with a percentage reversal."""
    piv, up, ext = [], True, 0
    for i in range(1, len(H)):
        if up:
            if H[i] >= H[ext]:
                ext = i
            elif L[i] <= H[ext] * (1 - pct):
                piv.append((ext, H[ext], "H"))
                up, ext = False, i
        else:
            if L[i] <= L[ext]:
                ext = i
            elif H[i] >= L[ext] * (1 + pct):
                piv.append((ext, L[ext], "L"))
                up, ext = True, i
    piv.append((ext, H[ext] if up else L[ext], "H" if up else "L"))
    return piv


def vcp(d: pd.DataFrame, window=130) -> dict:
    x = d.iloc[-window:]
    H, L, C, V = (x[c].values.astype(float) for c in ("High", "Low", "Close", "Volume"))
    vol50 = d["Volume"].astype(float).iloc[-50:].mean()
    top = int(np.argmax(H[:-3])) if len(H) > 10 else 0   # base starts at the highest high
    piv = [p for p in swings(H, L, 0.04) if p[0] >= top]
    if not piv or piv[0][2] != "H":
        piv = [(top, H[top], "H")] + [p for p in piv if p[0] > top]
    cons = []
    for a, b in zip(piv, piv[1:]):
        if a[2] == "H" and b[2] == "L":
            cons.append((a[0], b[0], (a[1] - b[1]) / a[1], a[1]))
    out = {"vcp": "No clear VCP", "contractions": "", "pivot": np.nan, "to_pivot_pct": np.nan,
           "dryup": np.nan, "_vcp_i": None}
    if len(cons) < 2:
        return out
    cons = cons[-4:]
    depths = [c[2] for c in cons]
    shrinking = all(depths[i + 1] <= depths[i] * 1.1 for i in range(len(depths) - 1))
    last_hi_i, last_lo_i = cons[-1][0], cons[-1][1]
    pivot = float(cons[-1][3])
    dry = float(V[last_lo_i - min(5, last_lo_i - last_hi_i):].mean() / vol50) if vol50 else np.nan
    out.update(contractions=" > ".join(f"{dd * 100:.0f}%" for dd in depths), pivot=round(pivot, 2),
               to_pivot_pct=round((pivot / C[-1] - 1) * 100, 1), dryup=round(dry, 2),
               _vcp_i=(len(d) - window + cons[0][0], len(d) - window + last_hi_i))
    if shrinking and depths[-1] <= 0.12 and depths[0] >= 0.10:
        if pivot < C[-1] <= pivot * 1.05 and (C[-6:-1] <= pivot * 1.01).any():
            out["vcp"] = "VCP - breaking out of pivot"
        elif pivot * 0.95 <= C[-1] <= pivot:
            out["vcp"] = "VCP - near pivot"
        elif C[-1] < pivot:
            out["vcp"] = "VCP forming"
        else:
            out["vcp"] = "VCP - extended past pivot"
    return out


# --------------------------------------------------------------------------------------
# Screener: quarterly results (Code 33) + everything the IPO scanner reads
# --------------------------------------------------------------------------------------
def parse_quarters(page: str) -> dict:
    i = page.find('id="quarters"')
    if i == -1:
        return {}
    sec = page[i:page.find("</section>", i)]
    heads = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h)).strip()
             for h in re.findall(r"<th[^>]*>(.*?)</th>", sec[sec.find("<thead"):sec.find("</thead>")], re.S)]
    heads = [h for h in heads if h]
    out = {"q_labels": heads}
    for key, labels in (("sales", ("Sales", "Revenue")), ("opm", ("OPM %", "Financing Margin %")),
                        ("eps", ("EPS in Rs",)), ("np", ("Net Profit",))):
        for lab in labels:
            m = re.search(r'<td class="text">\s*(?:<button[^>]*>)?\s*' + re.escape(lab) + r".*?</td>(.*?)</tr>", sec, re.S)
            if m:
                vals = [ip._num(v) for v in re.findall(r"<td[^>]*>(.*?)</td>", m.group(1), re.S)]
                out[key] = vals[-len(heads):] if heads else vals
                break
    return out


def code33(q: dict) -> dict:
    def yoy(v):
        v = [x if x is not None else np.nan for x in (v or [])]
        if len(v) < 7:
            return []
        g = []
        for k in (3, 2, 1):                       # last 3 quarters vs same quarter a year earlier
            a, b = v[-k], v[-k - 4]
            g.append((a / b - 1) * 100 if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan)
        return g
    eg, sg = yoy(q.get("eps")), yoy(q.get("sales"))
    opm = [x if x is not None else np.nan for x in (q.get("opm") or [])][-3:]
    acc = lambda g: len(g) == 3 and all(np.isfinite(g)) and g[0] < g[1] < g[2]
    out = {"eps_g_q": " > ".join(f"{x:.0f}%" for x in eg) if eg and all(np.isfinite(eg)) else "",
           "sales_g_q": " > ".join(f"{x:.0f}%" for x in sg) if sg and all(np.isfinite(sg)) else "",
           "opm_q": " > ".join(f"{x:.0f}%" for x in opm) if len(opm) == 3 and all(np.isfinite(opm)) else "",
           "eps_g_last": round(eg[-1], 1) if eg and np.isfinite(eg[-1]) else np.nan,
           "sales_g_last": round(sg[-1], 1) if sg and np.isfinite(sg[-1]) else np.nan,
           "last_quarter": (q.get("q_labels") or [""])[-1]}
    e, s = acc(eg), acc(sg)
    m = len(opm) == 3 and all(np.isfinite(opm)) and opm[0] < opm[1] < opm[2]
    out["code33"] = "Yes - EPS, sales & margin accelerating" if (e and s and m) else \
        ", ".join(n for n, f in (("EPS", e), ("Sales", s), ("Margin", m)) if f) + (" accelerating" if (e or s or m) else "") \
        if (e or s or m) else "No"
    return out


def company(symbols: list[str]) -> dict:
    f = CACHE_DIR / "stage2_screener.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    todo = [s for s in symbols if s not in cache or time.time() - cache[s].get("ts", 0) > 3 * 86400
            or cache[s].get("error")]
    if todo:
        print(f"Reading concalls, EPS history and quarterly results from Screener.in for {len(todo)} companies ...")

    def job(s):
        r = {"ts": time.time()}
        page = fp._fetch_screener(s)
        if page is None:
            r["error"] = "Screener not reachable"
        else:
            r["concalls"] = fp.parse_concalls(page)
            r.update(ip.parse_fundamentals(page))
            r["q"] = parse_quarters(page)
            if not r.get("eps") or not any(v is not None and np.isfinite(v) for v in r["eps"].values()):
                import requests
                try:
                    p2 = requests.get(f"https://www.screener.in/company/{s}/",
                                      headers={"User-Agent": es.UA["User-Agent"]}, timeout=30).text
                    r.update(ip.parse_fundamentals(p2))
                    r["q"] = parse_quarters(p2) or r["q"]
                    r["concalls"] = r["concalls"] or fp.parse_concalls(p2)
                except Exception:
                    pass
        time.sleep(1.2)
        return s, r

    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(job, s) for s in todo]):
            s, r = fut.result()
            cache[s] = r
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  companies: {done}/{len(todo)}", flush=True)
    try:
        f.write_text(json.dumps(cache, default=str))
    except Exception:
        pass
    return {s: cache.get(s, {}) for s in symbols}


# --------------------------------------------------------------------------------------
def volume_info(d: pd.DataFrame):
    v = d["Volume"].astype(float).replace(0, np.nan)
    rv = (v / v.shift(1).rolling(50, min_periods=20).mean()).values
    seg = rv[-20:]
    dx, dd = (np.nan, "")
    if np.isfinite(seg).any():
        i = len(rv) - 20 + int(np.nanargmax(seg))
        dx, dd = float(rv[i]), d.index[i].date().isoformat()
    w = fp.resample(d, "W-FRI")
    wv = w["Volume"].astype(float).replace(0, np.nan)
    wr = (wv / wv.shift(1).rolling(20, min_periods=8).mean()).values[-4:]
    wx = float(np.nanmax(wr)) if np.isfinite(wr).any() else np.nan
    c = d["Close"].values[-51:]
    vv = d["Volume"].values[-50:].astype(float)
    up = np.diff(c) > 0
    ud = vv[up].sum() / max(vv[~up].sum(), 1.0)
    return dx, dd, wx, round(float(ud), 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Minervini stage 2 scanner")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--universe")
    ap.add_argument("--min-mcap", type=float, default=500)
    ap.add_argument("--max-mcap", type=float, default=None)
    ap.add_argument("--exclude-top", type=int, default=250)
    ap.add_argument("--min-rs", type=float, default=70)
    ap.add_argument("--fresh-days", type=int, default=30, help="stage 2 'started' within this many sessions")
    ap.add_argument("--daily-spike", type=float, default=5.0)
    ap.add_argument("--weekly-spike", type=float, default=3.0)
    ap.add_argument("--check", nargs="*", default=[], help="your holdings: always report their stage and maturity")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    check = {x.upper().replace(".NS", "") for x in (a.check or [])}
    hold_file = HERE / "holdings.txt"
    if hold_file.exists():
        for line in hold_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.split("#")[0].replace(",", " ").strip()
            check |= {x.upper().replace(".NS", "") for x in line.split() if x}
    sys.stdout = Tee(OUT_DIR / "stage2_log.txt")
    today = dt.date.today()

    cfg = es.Config(min_mcap_cr=a.min_mcap, max_mcap_cr=a.max_mcap, exclude_top=a.exclude_top)
    uni = es.build_universe(a, cfg)
    if uni.empty:
        sys.exit("Universe is empty.")
    prices = es.download_prices(sorted(set(uni["symbol"]) | check), 3)
    meta = uni.set_index("symbol")[["name", "mcap_cr"]].to_dict("index")
    C = pd.DataFrame({s: d["Close"] for s, d in prices.items()}).sort_index()
    C = C[C.index.dayofweek < 5]
    H = pd.DataFrame({s: d["High"] for s, d in prices.items()}).reindex(C.index)
    L = pd.DataFrame({s: d["Low"] for s, d in prices.items()}).reindex(C.index)
    C, H, L = C.ffill(limit=3), H.ffill(limit=3), L.ffill(limit=3)
    rs = rs_rating(C)
    crit, ma = template(C, H, L, rs, a.min_rs)
    passed = crit[0].copy()
    for c in crit[1:]:
        passed &= c
    n_ok = sum(c.astype(int) for c in crit)
    print(f"Trend template computed for {C.shape[1]} stocks up to {C.index[-1].date()}")

    rows = []
    for s in C.columns:
        last = {i: bool(crit[i][s].iloc[-1]) for i in range(8)}
        k = int(n_ok[s].iloc[-1])
        if (k < 7 and s not in check) or not np.isfinite(ma["s200"][s].iloc[-1]):
            continue
        streak = stage2_run(passed[s].values) if passed[s].iloc[-1] else 0
        trigger = ""
        if 0 < streak <= a.fresh_days:
            start = len(C) - streak
            last_false = {i: (np.where(~crit[i][s].values[:start + 1])[0][-1]
                              if (~crit[i][s].values[:start + 1]).any() else -1) for i in range(8)}
            trigger = CRITERIA[max(last_false, key=last_false.get)]
        d = prices[s].loc[:C.index[-1]]
        c = float(C[s].iloc[-1])
        status = ("Stage 2 - just started" if 0 < streak <= a.fresh_days else
                  "Stage 2" if streak > a.fresh_days else
                  "Near stage 2 - missing: " + next(CRITERIA[i] for i in range(8) if not last[i]) if k == 7 else
                  f"Out of stage 2 ({k}/8 criteria)")
        dx, dd, wx, ud = volume_info(d)
        r = {"symbol": s, "name": meta.get(s, {}).get("name"), "mcap_cr": meta.get(s, {}).get("mcap_cr"),
             "price": round(c, 2), "status": status, "stage2_days": streak, "trigger": trigger,
             "stage2_since": C.index[-streak].date().isoformat() if streak else "",
             "criteria_met": k, "rs": int(rs[s].iloc[-1]) if np.isfinite(rs[s].iloc[-1]) else np.nan,
             "rs_1m_ago": int(rs[s].iloc[-22]) if np.isfinite(rs[s].iloc[-22]) else np.nan,
             "ma200_months_up": months_rising(ma["s200"][s]),
             "above_low_pct": round((c / ma["lo52"][s].iloc[-1] - 1) * 100, 1),
             "from_high_pct": round((c / ma["hi52"][s].iloc[-1] - 1) * 100, 1),
             "sma50": round(ma["s50"][s].iloc[-1], 2), "sma150": round(ma["s150"][s].iloc[-1], 2),
             "sma200": round(ma["s200"][s].iloc[-1], 2),
             "day_vol_x": round(dx, 2) if np.isfinite(dx) else np.nan, "day_vol_date": dd,
             "wk_vol_x": round(wx, 2) if np.isfinite(wx) else np.nan, "ud_ratio": ud,
             "_d": d}
        r["vol_ok"] = bool((np.isfinite(dx) and dx >= a.daily_spike) or (np.isfinite(wx) and wx >= a.weekly_spike))
        r.update(vcp(d))
        pv = passed[s].values
        j = (len(C) - streak) if streak else sm.last_run_start(pv)
        if j is not None:
            j = sm.cycle_start(pv, ma["s200"][s].values, j)
        sd = C.index[j] if j is not None else None
        r["_pre"] = sd is None
        r["stage2_began"] = sd.date().isoformat() if sd is not None else ""
        r["cycle_days"] = (len(C) - j) if j is not None else 0
        if "just" in r["status"] and r["cycle_days"] > streak + 10:
            r["status"] = "Stage 2 - resumed after a base"
        recent = bool(pv[-60:].any())
        r.update(sm.technical(d, sd, ma["s50"][s], ma["s200"][s], rs[s], k, recent, r["cycle_days"]))
        r["holding"] = "Yes" if s in check else ""
        rows.append(r)
    st2 = [r for r in rows if r["status"].startswith("Stage 2")]
    print(f"Stage 2 (all 8 criteria): {len(st2)} stocks, of which just started: "
          f"{sum(1 for r in st2 if 'just' in r['status'])}; near stage 2: {len(rows) - len(st2)}")
    cand = [r for r in rows if r["vol_ok"]]
    print(f"With unusual volume: {len(cand)}")

    syms = sorted({r["symbol"] for r in cand} | {r["symbol"] for r in rows if r["holding"]})
    comp = company(syms) if syms else {}
    guide = ip.load_guidance()
    yest = ip.yahoo_estimates([s for s in syms if s not in guide]) if syms else {}
    try:
        listed = ip.listing_dates()
    except Exception:
        listed = {}
    for r in [r for r in rows if r["symbol"] in comp]:
        c = comp.get(r["symbol"], {})
        if c.get("error"):
            r["concall_status"] = "Unknown (Screener not reachable)"
        else:
            r.update(ip.concall_check(c.get("concalls"), listed.get(r["symbol"], dt.date(2000, 1, 1)), today))
        r["pe_ttm"] = c.get("Stock P/E", np.nan)
        r.update(ip.forward_peg(r["symbol"], r["price"], c, guide, yest))
        if np.isfinite(r.get("pe_ttm") or np.nan) and np.isfinite(r.get("pe_y2", np.nan)) and r["pe_ttm"] > 0:
            r["pe_compression_pct"] = round((1 - r["pe_y2"] / r["pe_ttm"]) * 100, 1)
        r.update(code33(c.get("q") or {}))
        fpts, fsig, fo = sm.fundamental(c.get("q") or {}, r["price"], r["_start_price"], r["_start_date"])
        r.update(fo, _fpts=fpts, _fsig=fsig)
    sv = sm.sector_view(HERE)
    for r in rows:
        r["_sector"] = sv.get(r["symbol"])
        sm.label(r)
    hold = [r for r in rows if r["holding"]]
    if hold:
        print("\nYour stocks:")
        for r in hold:
            print(f"  {r['symbol']:12s} {r['status']:40s} {r['maturity']:26s} {r['maturity_signals']}")
    # delivery
    try:
        days = {x.date().isoformat() for x in C.index[-20:]} | {r["day_vol_date"] for r in cand if r["day_vol_date"]}
        bhav = es.load_delivery(days, CACHE_DIR / "bhav")
        recent = sorted(bhav)[-20:]
        for r in cand:
            s = r["symbol"]
            rec = bhav.get(r["day_vol_date"], {}).get(s)
            if rec and np.isfinite(rec[2]):
                r["spike_deliv_pct"] = round(float(rec[2]), 1)
            pc = [bhav[x][s][2] for x in recent if s in bhav[x] and np.isfinite(bhav[x][s][2])]
            if pc:
                r["avg20_deliv_pct"] = round(float(np.mean(pc)), 1)
    except Exception as e:
        print(f"  delivery data skipped ({e})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = frame(rows)
    xlsx, page = out / f"stage2_{today}.xlsx", out / f"stage2_{today}.html"
    n = write_excel(df, xlsx)
    write_html(rows, page, f"Minervini stage 2 scan - {today}")
    print(f"\nDone. Final list: {n} stocks.\n  Excel : {xlsx}\n  Report: {page}")


class Tee:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.o.flush()


# --------------------------------------------------------------------------------------
COLS = [
    ("symbol", "Symbol"), ("name", "Company"), ("mcap_cr", "Mcap (cr)"), ("price", "Price"),
    ("holding", "Your holding"), ("status", "Status"), ("maturity", "Stage 2 maturity"),
    ("maturity_score", "Maturity score"), ("maturity_signals", "Maturity signals"),
    ("stage2_since", "Stage 2 since"), ("stage2_days", "Days in stage 2"),
    ("trigger", "Last criterion to turn true"),
    ("criteria_met", "Template criteria met (of 8)"), ("rs", "RS rating"), ("rs_1m_ago", "RS 1M ago"),
    ("ma200_months_up", "200-DMA rising (months)"), ("above_low_pct", "% above 52w low"),
    ("from_high_pct", "% from 52w high"), ("vcp", "VCP"), ("contractions", "Contractions"),
    ("pivot", "Pivot"), ("to_pivot_pct", "To pivot %"), ("dryup", "Volume dry-up (x 50d)"),
    ("concall_status", "Concalls"), ("concalls_24m", "Concalls (24m)"), ("last_concall", "Last concall"),
    ("day_vol_x", "Daily vol (x 50d)"), ("day_vol_date", "Spike day"), ("spike_deliv_pct", "Spike-day delivery %"),
    ("avg20_deliv_pct", "20-day avg delivery %"), ("wk_vol_x", "Weekly vol (x 20w)"), ("ud_ratio", "Up/down volume (50d)"),
    ("code33", "Code 33"), ("eps_g_q", "EPS growth YoY, last 3Q"), ("sales_g_q", "Sales growth YoY, last 3Q"),
    ("opm_q", "OPM last 3Q"), ("last_quarter", "Latest quarter"),
    ("pe_ttm", "P/E (TTM)"), ("pe_y1", "Fwd P/E Y1"), ("pe_y2", "Fwd P/E Y2"), ("pe_y3", "Fwd P/E Y3"),
    ("pe_compression_pct", "P/E compression to Y2 %"), ("peg_y1", "Fwd PEG Y1"), ("peg_y2", "Fwd PEG Y2"),
    ("peg_y3", "Fwd PEG Y3"), ("growth_source", "Growth source"), ("peg_note", "PEG note"),
    ("base_count", "Base count"), ("ext200_pct", "% above 200-DMA"), ("climax", "Climax run"),
    ("dist_days", "Distribution days (25d)"), ("broke_50dma", "Broke 50-DMA on volume"),
    ("ma200_flat", "200-DMA flattening / chop"), ("rs_divergence", "RS divergence"),
    ("pe_expansion_x", "P/E expansion since stage 2 (x)"), ("stage2_began", "Stage 2 cycle began"), ("cycle_days", "Sessions since cycle began"),
    ("sma50", "50-DMA"), ("sma150", "150-DMA"), ("sma200", "200-DMA"),
]
MAT_ORDER = {"Early stage 2": 0, "Mid stage 2": 1, "Late stage 2": 2, "Topping - stage 3 risk": 3,
             "Stage 4 - downtrend": 3, "Pre-stage 2 (still basing)": 4}
VCP_ORDER = {"VCP - breaking out of pivot": 0, "VCP - near pivot": 1, "VCP forming": 2,
             "VCP - extended past pivot": 3, "No clear VCP": 4}


def frame(rows):
    df = pd.DataFrame([{k: r.get(k, np.nan) for k, _ in COLS} |
                       {"_v": r["vol_ok"], "_c": r.get("concall_status") == "Regular",
                        "_s2": r["status"].startswith("Stage 2"), "_fresh": "just" in r["status"],
                        "_hold": bool(r.get("holding")), "_near": r["status"].startswith("Near")} for r in rows])
    if df.empty:
        return df
    df["_vo"] = df["vcp"].map(VCP_ORDER).fillna(9)
    return df.sort_values(["_fresh", "_vo", "rs"], ascending=[False, True, False])


def write_excel(df, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    hdr, hide = dict(COLS), [c for c in df.columns if c.startswith("_")]
    fin = df[df["_s2"] & df["_v"] & df["_c"]] if not df.empty else df
    sheets = {"Final list": fin,
              "Your stocks": df[df["_hold"]],
              "Late & topping (all scanned)": df[df["maturity"].isin(["Late stage 2", "Topping - stage 3 risk"])
                                                 & (df["_s2"] | df["_near"])].sort_values("maturity_score", ascending=False),
              "Near stage 2 (vol+concalls ok)": df[df["_near"] & df["_v"] & df["_c"]],
              "Concalls not regular": df[df["_v"] & ~df["_c"]],
              "Stage 2 - no vol spike": df[df["_s2"] & ~df["_v"]],
              "Near stage 2 - no vol spike": df[df["_near"] & ~df["_v"]]}
    if not df["_hold"].any():
        sheets.pop("Your stocks")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, t in sheets.items():
            t.drop(columns=hide).rename(columns=hdr).to_excel(xw, sheet_name=name[:31], index=False)
        pd.DataFrame({"Signal": ["Base count", "% above 200-DMA", "Climax run", "Distribution",
                                 "Broke 50-DMA on volume", "200-DMA flattening / chop", "RS divergence",
                                 "EPS deceleration", "Margin contraction", "P/E expansion", "Sector rotating out",
                                 "Labels", "Your holdings"],
                      "Points / meaning": [
            "Weekly corrections of 15%+ since the stage 2 cycle began, +1. 3rd base +2, 4th or later +3 (late-stage bases fail more often).",
            "Over 70% +1, over 100% +2 - stretched far from the long-term average.",
            "+2. Largest day (6%+) or week (12%+) of the whole move within the last 60 sessions, with 25%+ in 3 weeks, or 15%+ with 8/10 up days or a 4% gap up.",
            "+1. 4+ down days of 2%+ on 1.5x volume in 25 sessions while up/down volume is below 1.2, or up/down volume below 0.8.",
            "+2. Closed below the 50-day on 1.4x average volume in the last 30 sessions and still below it.",
            "+2. 200-day rise has stalled (<0.5% in a month after >1% earlier) or price crossed it 3+ times in 60 days - only counted once the cycle is 120+ sessions old.",
            "+1. Price within 3% of its 52-week high but RS rating 10+ points below its 6-month peak.",
            "+1. EPS growth (YoY) lower in each of the last 3 quarters.",
            "+1. Operating margin lower in each of the last 3 quarters.",
            "+1. Trailing P/E now 2x or more the P/E when the stage 2 run began.",
            "+1. Stock's NSE sector is Weakening or Lagging on the long-term view of the latest sector rotation report.",
            "Early = stage 2 under 60 sessions, 1st/2nd base, at most 1 point. Mid = up to 2 points. Late = 3-4. "
            "Topping - stage 3 risk = 5+, or a 50-DMA break on volume with distribution or a flattening 200-DMA. "
            "Stage 4 - downtrend = below a falling 200-DMA. Fell out of the template after a stage 2 run adds +2.",
            "List your stocks in holdings.txt (one symbol per line) or pass --check SYMBOL ...; they are always "
            "reported in 'Your stocks', even after leaving stage 2."]}).to_excel(xw, sheet_name="Stage 2 maturity", index=False)
        pd.DataFrame({"Item": ["Timeframe", "Trend Template", "Stage 2 - just started", "Near stage 2",
                               "RS rating", "VCP", "Pivot", "Volume dry-up", "Unusual volume", "Up/down volume",
                               "Code 33", "P/E compression", "Concalls", "Caveat"], "Meaning": [
            "Daily: Minervini's template uses the 50/150/200-day moving averages.",
            "; ".join(f"{i + 1}. {c}" for i, c in enumerate(CRITERIA)) + ". All 8 = stage 2.",
            "All 8 criteria true for 30 sessions or less (the stage 2 start).",
            "7 of 8 criteria true; the missing one is named in Status.",
            "IBD-style: 40% 3-month + 20% each 6/9/12-month return, ranked 1-99 against all small/micro caps.",
            "Pullbacks inside the base get smaller (first >= 10%, last <= 12%), e.g. 24% > 13% > 6%.",
            "High of the last contraction - Minervini's buy point; 'To pivot %' is the distance from today's price.",
            "Average volume in the last contraction vs the 50-day average. Below 1 = supply drying up (good).",
            "A day >= 5x its 50-day average in the last 20 sessions, or a week >= 3x its 20-week average in the last 4.",
            "Volume on up days / volume on down days over 50 sessions. Above 1 = accumulation.",
            "EPS growth, sales growth and operating margin all accelerating for the last 3 quarters (YoY).",
            "1 - (forward P/E of year 2 / trailing P/E). Higher = earnings growth is compressing the multiple. "
            "Forward EPS: your guidance.csv > Yahoo consensus > own EPS CAGR (faded). See 'Growth source'.",
            "Screener.in: 4+ calls in 24 months and one within 7 months (scaled for recent listings).",
            "Mechanical scan - check every chart. Not investment advice."]}).to_excel(xw, sheet_name="How to read", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F4E78")
                c.alignment = Alignment(wrap_text=True, vertical="center")
            for j, c in enumerate(ws[1], 1):
                ws.column_dimensions[get_column_letter(j)].width = {
                    "Company": 28, "Status": 30, "VCP": 24, "Stage 2 maturity": 22, "Maturity signals": 70,
                    "Signal": 24, "Points / meaning": 110, "Code 33": 30, "Concalls": 20, "Growth source": 36,
                    "PEG note": 30, "Item": 22, "Meaning": 110}.get(c.value, 12)
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
    return len(fin)


def chart(r, w=340, h=150):
    d = r["_d"].iloc[-200:]
    C = d["Close"].astype(float)
    full = r["_d"]["Close"].astype(float)
    mas = {k: full.rolling(n).mean().iloc[-len(d):].values for k, n in (("m50", 50), ("m150", 150), ("m200", 200))}
    V = d["Volume"].astype(float).values
    n = len(d)
    vals = np.concatenate([C.values] + [m[np.isfinite(m)] for m in mas.values()])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad, ph, vh = 12, 100, 26
    X = lambda i: pad + (w - 2 * pad) * i / max(n - 1, 1)
    Y = lambda p: pad + ph * (hi - p) / max(hi - lo, 1e-9)
    line = lambda arr, cls: (f'<polyline class="{cls}" points="' + " ".join(
        f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(arr) if np.isfinite(v)) + '"/>')
    o = [line(mas["m200"], "m200"), line(mas["m150"], "m150"), line(mas["m50"], "m50"), line(C.values, "px")]
    if np.isfinite(r.get("pivot", np.nan)) and lo <= r["pivot"] <= hi:
        o.append(f'<line x1="{X(n * 0.55):.1f}" x2="{X(n - 1):.1f}" y1="{Y(r["pivot"]):.1f}" y2="{Y(r["pivot"]):.1f}" class="pv"/>')
    vmax = float(np.nanmax(V)) or 1.0
    by = pad + ph + 8 + vh
    hot = d.index.get_indexer([pd.Timestamp(r["day_vol_date"])])[0] if r.get("day_vol_date") else -1
    bw = max(0.6, (w - 2 * pad) / n * 0.7)
    for i in range(n):
        hh = vh * (V[i] / vmax if np.isfinite(V[i]) else 0)
        o.append(f'<rect x="{X(i) - bw / 2:.1f}" y="{by - hh:.1f}" width="{bw:.1f}" height="{hh:.1f}" '
                 f'class="{"vs" if (i == hot and r["day_vol_x"] >= 5) else "vb"}"/>')
    return f'<svg viewBox="0 0 {w} {by + 4:.0f}" class="ch" role="img" aria-label="daily chart with 50/150/200-day averages">{"".join(o)}</svg>'


def write_html(rows, path, title):
    fin = [r for r in rows if r["status"].startswith("Stage 2") and r["vol_ok"] and r.get("concall_status") == "Regular"]
    fin.sort(key=lambda r: ("just" not in r["status"], VCP_ORDER.get(r["vcp"], 9), -(r["rs"] or 0)))

    def f(v, pct=False, d=2):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:+.1f}%" if pct else f"{v:,.{d}f}"

    def card(r):
        fresh = "just" in r["status"]
        tags = [f'<span class="st {"fr" if fresh else "s2"}">{html.escape(r["status"])}</span>']
        if r["vcp"] != "No clear VCP":
            tags.append(f'<span class="tg vc">{html.escape(r["vcp"])}</span>')
        if str(r.get("code33", "")).startswith("Yes"):
            tags.append('<span class="tg c3">Code 33</span>')
        mo = MAT_ORDER.get(r.get("maturity"), 1)
        tags.insert(1, f'<span class="tg m{mo}">{html.escape(r.get("maturity", ""))}</span>')
        pe = ""
        if np.isfinite(r.get("pe_y1", np.nan)):
            lb = r.get("fy_labels") or ["Y1", "Y2", "Y3"]
            pe = (f'<div class="pe"><span>P/E TTM <b>{f(r.get("pe_ttm"), d=1)}</b></span>'
                  + "".join(f'<span>{lb[i - 1]} <b>{f(r.get(f"pe_y{i}"), d=1)}</b> <i>PEG {f(r.get(f"peg_y{i}"))}</i></span>' for i in (1, 2, 3))
                  + (f'<span>Compression <b>{r["pe_compression_pct"]:.0f}%</b></span>' if np.isfinite(r.get("pe_compression_pct", np.nan)) else "")
                  + f'</div><p class="src">{html.escape(str(r.get("growth_source", "")))}</p>')
        elif r.get("peg_note"):
            pe = f'<p class="src">P/E: {html.escape(str(r["peg_note"]))}</p>'
        dl = (f', delivery {r["spike_deliv_pct"]:.0f}% (20d avg {r.get("avg20_deliv_pct", np.nan):.0f}%)'
              if np.isfinite(r.get("spike_deliv_pct", np.nan)) else "")
        return f"""
<article class="card" data-f="{'1' if fresh else '0'}" data-v="{VCP_ORDER.get(r['vcp'], 9)}" data-m="{MAT_ORDER.get(r.get('maturity'), 1)}">
 <header><div><a href="https://www.tradingview.com/chart/?symbol=NSE:{html.escape(r['symbol'])}" target="_blank" rel="noopener">{html.escape(r['symbol'])}</a>
  <span class="nm">{html.escape(str(r.get('name', '')))[:28]}</span></div><span class="rs" title="RS rating">RS {r['rs']}</span></header>
 <div class="tags">{''.join(tags)}</div>
 <p class="dt">In stage 2 since {r['stage2_since']} ({r['stage2_days']} sessions{(', completed by: ' + html.escape(r['trigger'])) if r.get('trigger') else ''}) &middot; 200-DMA rising {r['ma200_months_up']} months &middot;
  {r['from_high_pct']:+.0f}% from 52w high, {r['above_low_pct']:.0f}% above low</p>
 {chart(r)}
 <dl><div><dt>Price</dt><dd>{f(r['price'])}</dd></div><div><dt>Mcap</dt><dd>{f(r.get('mcap_cr'), d=0)} cr</dd></div>
  <div><dt>Pivot</dt><dd>{f(r.get('pivot'))} <i>{f(r.get('to_pivot_pct'), True)}</i></dd></div>
  <div><dt>Contractions</dt><dd>{html.escape(r.get('contractions') or '-')}</dd></div>
  <div><dt>Up/down vol</dt><dd>{f(r['ud_ratio'])}</dd></div><div><dt>Concalls</dt><dd>{r.get('concalls_24m', '-')} in 24m</dd></div></dl>
 <p class="ln"><b>Volume</b> {f(r['day_vol_x'], d=1)}x on {r['day_vol_date']}{dl}; week {f(r['wk_vol_x'], d=1)}x</p>
 <p class="ln"><b>Quarters</b> EPS {html.escape(r.get('eps_g_q') or '-')} &middot; sales {html.escape(r.get('sales_g_q') or '-')} &middot; OPM {html.escape(r.get('opm_q') or '-')}</p>
 {pe}
 <p class="ln"><b>Maturity</b> base {html.escape(str(r.get('base_count', '-')))}, {f(r.get('ext200_pct'), d=0)}% above 200-DMA{(' &middot; ' + html.escape(r['maturity_signals'])) if r.get('maturity_signals') else ' &middot; no warning signs'}</p>
</article>"""

    n_f = sum(1 for r in fin if "just" in r["status"])
    n_v = sum(1 for r in fin if VCP_ORDER.get(r["vcp"], 9) <= 1)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f6f2;--card:#fff;--tx:#1d1d1b;--mu:#6b6a65;--bd:#e3e1da;--ac:#1f4e78;--g:#1d7a4f;--gb:#e1f3e8;
--b:#185fa5;--bb:#e6f1fb;--a:#8a5a07;--ab:#fbf0d9;--r:#a32d2d;--rb:#fcebeb;--px:#2c2c2a;--m50:#185fa5;--m150:#ba7517;--m200:#a32d2d;--vb:#c9c7bf;--vs:#d85a30}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191917;--card:#232321;--tx:#ecebe6;--mu:#a3a199;--bd:#3a3936;--ac:#85b7eb;
--g:#7fd1a5;--gb:#173b2a;--b:#85b7eb;--bb:#132c46;--a:#efc06a;--ab:#3d2b0c;--r:#f09595;--rb:#4a1b1b;--px:#ecebe6;--m50:#85b7eb;--m150:#efc06a;--m200:#f09595;
--vb:#55534d;--vs:#f0997b}}}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--tx)}}
main{{max-width:1260px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mu);margin:0 0 14px;max-width:920px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}}.chip{{border:1px solid var(--bd);background:var(--card);color:var(--tx);
border-radius:999px;padding:5px 12px;cursor:pointer;font:inherit}}.chip.on{{border-color:var(--ac);color:var(--ac)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px}}
.card header{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
.card a{{color:var(--ac);font-weight:600;text-decoration:none;font-size:16px}}.nm{{color:var(--mu);margin-left:6px;font-size:12px}}
.rs{{font-weight:600;font-size:12px;border:1px solid var(--bd);border-radius:6px;padding:0 7px}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px}}.st,.tg{{padding:2px 9px;border-radius:6px;font-size:12px;font-weight:600}}
.fr{{background:var(--gb);color:var(--g)}}.s2{{background:var(--bb);color:var(--b)}}.vc{{background:var(--ab);color:var(--a)}}
.c3{{border:1px solid var(--g);color:var(--g)}}.m0{{background:var(--gb);color:var(--g)}}.m1{{background:var(--bb);color:var(--b)}}
.m2{{background:var(--ab);color:var(--a)}}.m3{{background:var(--rb);color:var(--r)}}.m4{{border:1px solid var(--bd);color:var(--mu)}}.dt{{margin:2px 0 4px;color:var(--mu);font-size:12px}}
.ch{{width:100%;height:auto;display:block;margin:4px 0}}.ch polyline{{fill:none}}.px{{stroke:var(--px);stroke-width:1.2}}
.m50{{stroke:var(--m50);stroke-width:1}}.m150{{stroke:var(--m150);stroke-width:1}}.m200{{stroke:var(--m200);stroke-width:1}}
.pv{{stroke:var(--g);stroke-dasharray:4 3;stroke-width:1}}.vb{{fill:var(--vb)}}.vs{{fill:var(--vs)}}
.lg{{display:flex;gap:12px;font-size:12px;color:var(--mu);margin:0 0 12px;flex-wrap:wrap}}.lg i{{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:4px}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;margin:6px 0}}dl div{{display:flex;justify-content:space-between;gap:6px}}
dt{{color:var(--mu);font-size:12px}}dd{{margin:0;font-variant-numeric:tabular-nums;font-size:13px}}dd i{{color:var(--mu);font-style:normal;font-size:11px}}
.ln{{font-size:12px;color:var(--mu);margin:4px 0;border-top:1px solid var(--bd);padding-top:4px}}.ln b{{color:var(--tx)}}
.pe{{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;border-top:1px solid var(--bd);padding-top:4px;margin-top:4px}}
.pe span{{color:var(--mu)}}.pe b{{color:var(--tx)}}.pe i{{font-style:normal}}.src{{color:var(--mu);font-size:11px;margin:2px 0}}
.note{{color:var(--mu);font-size:12px;margin-top:24px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">Small &amp; micro caps passing all 8 points of Minervini's Trend Template (daily chart), with an unusual volume print
and regular concalls. "Just started" = the template turned true in the last 30 sessions. Sorted: fresh stage 2 first, then by VCP stage and RS rating.</p>
<div class="lg"><span><i style="background:var(--px)"></i>Price</span><span><i style="background:var(--m50)"></i>50-DMA</span>
<span><i style="background:var(--m150)"></i>150-DMA</span><span><i style="background:var(--m200)"></i>200-DMA</span>
<span><i style="background:var(--g)"></i>VCP pivot</span><span><i style="background:var(--vs);height:8px"></i>Volume spike</span></div>
<div class="chips"><button class="chip on" data-k="*">All <b>{len(fin)}</b></button>
<button class="chip" data-k="f">Stage 2 just started <b>{n_f}</b></button>
<button class="chip" data-k="v">VCP near / breaking pivot <b>{n_v}</b></button>
<button class="chip" data-k="m0">Early stage 2 <b>{sum(1 for r in fin if r.get('maturity') == 'Early stage 2')}</b></button>
<button class="chip" data-k="m2">Late / topping <b>{sum(1 for r in fin if MAT_ORDER.get(r.get('maturity'), 1) >= 2)}</b></button></div>
<section class="grid">{''.join(card(r) for r in fin) or '<p>No stock passed every filter.</p>'}</section>
<p class="note">Mechanical scan based on Minervini's published Trend Template and VCP description. Check every chart. Not investment advice.</p>
</main><script>
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===b));
const k=b.dataset.k;document.querySelectorAll('.card').forEach(c=>c.style.display=(k==='*'||(k==='f'&&c.dataset.f==='1')||(k==='v'&&+c.dataset.v<=1)||(k==='m0'&&c.dataset.m==='0')||(k==='m2'&&+c.dataset.m>=2))?'':'none');}});
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
