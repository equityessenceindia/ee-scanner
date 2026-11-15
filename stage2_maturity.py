"""
Stage 2 maturity - is the advance early, mid, late, or topping into stage 3?

Signals (Minervini / Weinstein), each adds points:
  Base count since stage 2 began         weekly corrections of 15%+; 3rd base +2, 4th or later +3
  Extension above the 200-day average    > 70% +1, > 100% +2
  Climax run                             +2  (largest day/week gain of the move within the last 60 sessions,
                                              with 25%+ in 3 weeks, 8 of 10 up days, or a 4%+ gap up)
  Distribution                           +1  (4+ down days of 2%+ on 1.5x volume in 25 sessions with up/down vol < 1.2, or up/down < 0.8)
  50-DMA broken on volume                +2  (close below the 50-day on 1.4x volume in 30 sessions, not reclaimed)
  200-DMA flattening / price chopping    +2  (stage 3 hallmark; only once the cycle is 120+ sessions old)
  RS divergence                          +1  (price near its high, RS rating 10+ points below its recent peak)
  EPS growth decelerating 3 quarters     +1
  Operating margin contracting           +1
  P/E expanded 2x+ since stage 2 began   +1
  Sector rotating out (long-term view)   +1  (from the latest sector rotation report, when available)
Label: Early (young move, few signals) / Mid (0-2) / Late (3-4) / Topping - stage 3 risk (5+, or a
50-DMA break together with distribution or a flattening 200-DMA).
"""
from __future__ import annotations

import datetime as dt
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

MONTH = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                                     "Oct", "Nov", "Dec"], 1)}


def last_run_start(passed: np.ndarray, gap=10, lookback=250):
    """Index where the most recent stage 2 run began (None if no run in the lookback)."""
    n = len(passed)
    idx = np.where(passed[max(0, n - lookback):])[0]
    if not len(idx):
        return None
    i = max(0, n - lookback) + idx[-1]
    start, miss = i, 0
    while i >= 0:
        if passed[i]:
            start, miss = i, 0
        else:
            miss += 1
            if miss > gap:
                break
        i -= 1
    return start


def cycle_start(passed: np.ndarray, s200: np.ndarray, fallback):
    """Start of the whole stage 2 cycle: first template pass after the 200-DMA last stopped rising."""
    rising = np.r_[np.full(21, False), s200[21:] > s200[:-21]]
    not_rising = np.where(~rising & np.isfinite(s200))[0]
    after = not_rising[-1] + 1 if len(not_rising) else 0
    hits = np.where(passed[after:])[0]
    if len(hits):
        return after + hits[0]
    return fallback


def _swings(H, L, pct):
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
    return piv, up, ext


def technical(d: pd.DataFrame, start_date, s50: pd.Series, s200: pd.Series, rs: pd.Series,
              k: int = 8, recent_stage2: bool = True, cycle_days: int = 999) -> dict:
    """Price/volume maturity signals. d = the stock's daily OHLCV; s50/s200/rs aligned to the universe calendar."""
    d = d.dropna(subset=["Close"])
    C, H, L, O = (d[c].values.astype(float) for c in ("Close", "High", "Low", "Open"))
    V = d["Volume"].values.astype(float)
    n = len(d)
    st = int(np.searchsorted(d.index.values, np.datetime64(start_date))) if start_date is not None else max(0, n - 120)
    st = min(max(st, 0), n - 2)
    s50v = s50.reindex(d.index).values
    s200v = s200.reindex(d.index).values
    avgv = pd.Series(V).shift(1).rolling(50, min_periods=20).mean().values
    sig, pts = [], 0
    out = {}

    # 1. base count since stage 2 began: weekly corrections of 15%+ (how bases are drawn on a weekly chart)
    w = d.iloc[st:].resample("W-FRI").agg({"High": "max", "Low": "min"}).dropna()
    piv, up, ext = _swings(w["High"].values, w["Low"].values, 0.15)
    corrections = sum(1 for a, b in zip(piv, piv[1:]) if a[2] == "H" and b[2] == "L")
    if not up:
        corrections += 1                      # currently in a 15%+ correction = a base forming
    base = 1 + corrections
    out["base_count"] = f"{base}{' (forming)' if not up else ''}"
    if base >= 4:
        pts += 3
        sig.append(f"{base}th base - late stage")
    elif base == 3:
        pts += 2
        sig.append("3rd base - late stage begins")

    # 2. extension above the 200-day
    ext200 = (C[-1] / s200v[-1] - 1) * 100 if np.isfinite(s200v[-1]) else np.nan
    out["ext200_pct"] = round(ext200, 1) if np.isfinite(ext200) else np.nan
    if np.isfinite(ext200) and ext200 > 100:
        pts += 2
        sig.append(f"{ext200:.0f}% above 200-DMA (very stretched)")
    elif np.isfinite(ext200) and ext200 > 70:
        pts += 1
        sig.append(f"{ext200:.0f}% above 200-DMA")

    # 3. climax run
    ret = np.r_[np.nan, C[1:] / C[:-1] - 1]
    since = ret[st + 1:]
    wk = pd.Series(C, index=d.index).iloc[st:].resample("W-FRI").last().pct_change().dropna()
    W = 60                                    # look back 60 sessions: catches climaxes that already topped
    climax, info = False, ""
    if len(since) > W:
        j = n - W + int(np.nanargmax(ret[-W:]))
        big_day = ret[j] >= np.nanmax(since) * 0.999 and ret[j] >= 0.06
        big_week = len(wk) > 8 and wk.iloc[-12:].max() >= wk.max() * 0.999 and wk.iloc[-12:].max() >= 0.12
        for e in range(n - W + 15, n):         # any 3-week window ending in the last 40 sessions
            g3 = C[e] / C[e - 15] - 1
            ups = int(np.sum(ret[e - 9:e + 1] > 0))
            gp = bool(np.any(O[e - 9:e + 1] / H[e - 10:e] - 1 >= 0.04))
            if (big_day or big_week) and (g3 >= 0.25 or (g3 >= 0.15 and (ups >= 8 or gp))):
                climax, info = True, f"{g3 * 100:+.0f}% in 3 weeks, {ups}/10 up days{', gap up' if gp else ''}, {n - 1 - e} sessions ago"
                break
    out["climax"] = "Yes" if climax else ""
    if climax:
        pts += 2
        sig.append(f"climax run ({info})")

    # 4. distribution
    dist = int(np.sum((ret[-25:] <= -0.02) & (V[-25:] >= 1.5 * avgv[-25:])))
    c51, v50 = C[-51:], V[-50:]
    upm = np.diff(c51) > 0
    ud = v50[upm].sum() / max(v50[~upm].sum(), 1.0)
    out["dist_days"] = dist
    if (dist >= 4 and ud < 1.2) or ud < 0.8:
        pts += 1
        sig.append(f"distribution ({dist} heavy down days in 25, up/down vol {ud:.2f})")

    # 5. 50-day broken on volume
    broke50 = False
    for i in range(max(1, n - 30), n):
        if (np.isfinite(s50v[i]) and C[i] < s50v[i] and C[i - 1] >= s50v[i - 1]
                and np.isfinite(avgv[i]) and V[i] >= 1.4 * avgv[i]):
            broke50 = True
    broke50 = broke50 and np.isfinite(s50v[-1]) and C[-1] < s50v[-1]
    out["broke_50dma"] = "Yes" if broke50 else ""
    if broke50:
        pts += 2
        sig.append("broke 50-DMA on heavy volume")

    # 6. 200-day flattening / chop
    flat = chop = False
    if cycle_days > 120 and n > 70 and np.all(np.isfinite(s200v[[-1, -22, -43, -64]])):
        ch1 = s200v[-1] / s200v[-22] - 1
        ch_prev = s200v[-43] / s200v[-64] - 1
        flat = ch1 < 0.005 and ch_prev > 0.01
        above = C[-60:] > s200v[-60:]
        chop = int(np.sum(above[1:] != above[:-1])) >= 3
    if flat or chop:
        pts += 2
        sig.append("200-DMA flattening" if flat else "price chopping around 200-DMA")
    out["ma200_flat"] = "Yes" if (flat or chop) else ""

    # 7. RS divergence
    rsv = rs.reindex(d.index).values
    if np.isfinite(rsv[-1]) and C[-1] >= 0.97 * np.nanmax(C[-252:]) and np.nanmax(rsv[-120:]) - rsv[-1] >= 10:
        pts += 1
        sig.append(f"RS divergence (RS {rsv[-1]:.0f} vs peak {np.nanmax(rsv[-120:]):.0f})")
        out["rs_divergence"] = "Yes"
    # 8. fell out of the template after a stage 2 run
    if k <= 6 and recent_stage2:
        pts += 2
        sig.append(f"fell out of the trend template ({k}/8 criteria)")
    out["_stage4"] = bool(np.isfinite(s200v[-1]) and C[-1] < s200v[-1] and n > 22 and s200v[-1] < s200v[-22])
    out["_pts"], out["_sig"] = pts, sig
    out["_start_price"] = float(C[st])
    out["_start_date"] = d.index[st].date()
    return out


def _q_end(label: str):
    try:
        m, y = label.split()
        mm = MONTH[m[:3]]
        return (pd.Timestamp(int(y), mm, 1) + pd.offsets.MonthEnd(0)).date()
    except Exception:
        return None


def fundamental(q: dict, price_now: float, start_price: float, start_date: dt.date) -> tuple[int, list, dict]:
    pts, sig, out = 0, [], {}
    eps, sales, opm, labels = (q.get(k) or [] for k in ("eps", "sales", "opm", "q_labels"))
    clean = lambda v: [x if (x is not None and np.isfinite(x)) else np.nan for x in v]
    eps, sales, opm = clean(eps), clean(sales), clean(opm)
    if len(eps) >= 7:
        g = [(eps[-k] / eps[-k - 4] - 1) if eps[-k - 4] and eps[-k - 4] > 0 and np.isfinite(eps[-k]) else np.nan
             for k in (3, 2, 1)]
        if all(np.isfinite(g)) and g[0] > g[1] > g[2]:
            pts += 1
            sig.append("EPS growth decelerating " + " > ".join(f"{x * 100:.0f}%" for x in g))
    if len(opm) >= 3 and all(np.isfinite(opm[-3:])) and opm[-3] > opm[-2] > opm[-1]:
        pts += 1
        sig.append("margins contracting " + " > ".join(f"{x:.0f}%" for x in opm[-3:]))
    # P/E expansion since stage 2 began (trailing 4-quarter EPS, results assumed public 45 days after quarter end)
    if len(eps) >= 8 and len(labels) == len(eps):
        ends = [_q_end(lb) for lb in labels]
        now_eps = np.nansum(eps[-4:])
        avail = [i for i, e in enumerate(ends) if e and e + dt.timedelta(days=45) <= start_date]
        if avail and avail[-1] >= 3:
            j = avail[-1]
            start_eps = np.nansum(eps[j - 3:j + 1])
            if now_eps > 0 and start_eps > 0:
                exp = (price_now / now_eps) / (start_price / start_eps)
                out["pe_expansion_x"] = round(exp, 2)
                if exp >= 2:
                    pts += 1
                    sig.append(f"P/E expanded {exp:.1f}x since stage 2 began")
    return pts, sig, out


def sector_view(base: Path) -> dict:
    """symbol -> long-term quadrant of its sector, from the latest sector rotation report."""
    try:
        files = sorted(glob.glob(str(base / "output" / "sector_rotation_*.xlsx")), key=os.path.getmtime)
        if not files:
            return {}
        rot = pd.read_excel(files[-1], sheet_name="Small & micro caps")
        quad = dict(zip(rot["Sector"], rot["Long-term quadrant"]))
        mem = pd.concat([pd.read_csv(base / "cache" / f) for f in ("totalmarket.csv",)
                         if (base / "cache" / f).exists()])
        return {s: (sec, quad.get(sec)) for s, sec in zip(mem["symbol"], mem["sector"])}
    except Exception:
        return {}


def label(r: dict) -> None:
    pts, sig = r.get("_pts", 0), list(r.get("_sig", []))
    if r.get("_fpts"):
        pts += r["_fpts"]
        sig += r.get("_fsig", [])
    sec = r.get("_sector")
    if sec and sec[1] in ("Weakening", "Lagging"):
        pts += 1
        sig.append(f"sector ({sec[0]}) {sec[1].lower()} long term")
    r["maturity_score"] = pts
    r["maturity_signals"] = "; ".join(sig)
    if r.get("_stage4"):
        r["maturity"] = "Stage 4 - downtrend"
        return
    if r.get("_pre"):
        r["maturity"] = "Pre-stage 2 (still basing)"
        return
    topping = pts >= 5 or (r.get("broke_50dma") == "Yes" and (r.get("ma200_flat") == "Yes" or r.get("dist_days", 0) >= 4))
    if topping:
        r["maturity"] = "Topping - stage 3 risk"
    elif pts >= 3:
        r["maturity"] = "Late stage 2"
    elif r.get("cycle_days", 999) <= 60 and str(r.get("base_count", "9"))[0] in "12" and pts <= 1:
        r["maturity"] = "Early stage 2"
    else:
        r["maturity"] = "Mid stage 2"
