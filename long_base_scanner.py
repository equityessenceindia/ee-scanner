#!/usr/bin/env python3
"""
Two scanners for the EQUITY ESSENCE app (used by app_scan.py):

accum  "Base + Accumulation" - a stock coming out of a LONG consolidation on the weekly or monthly
       chart, with at least one unusual-volume print INSIDE the base, before the breakout
       (someone accumulating while the price was still going sideways).
         Weekly : 26+ weeks (up to 5 years) of weekly closes inside a 40% range
         Monthly: 12+ months (up to 8 years) of monthly closes inside a 60% range
         Base must be flat (not a slow trend) and its top (closing basis) tested 2+ times.
         Coming out now: within 5% below the top, breaking out this bar, or a confirmed breakout
         in the last 4 weeks / 2 months that still holds and is not more than 15% / 20% extended.
         Unusual volume inside the base, on an up bar: a week >= 3x its 20-week average, a month
         >= 2.5x its 12-month average, or a day >= 5x its 50-day average.

ipo    "IPO Base Breakout" - listed in the last 3 years, came out of its first consolidation
       (IPO base) after listing, with good volume on the way out.
         Base of 6+ weeks after the listing week, weekly closes inside a 45% range, flat, top tested 2+ times.
         Coming out now: within 5% below the base top, breaking out this week, or a confirmed
         breakout in the last 4 weeks that still holds (max 20% extended).
         Good volume: a week in the last 4 >= 2.5x the base's average week, or a day in the last
         20 sessions >= 3x its 50-day average (listing-week volume is ignored).

Educational use only - not investment advice.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

import flag_pole_scanner as fp
import ipo_flag_scanner as ip


@dataclass
class Prm:
    tf: str              # "W" or "M"
    rule: str            # resample rule
    min_len: int
    max_len: int
    max_width: float
    near: float = 0.05
    max_brk: int = 4
    max_ext: float = 0.15
    touch: float = 0.03
    min_tests: int = 2
    vol_avg: int = 20     # bars in the volume average
    spike: float = 3.0    # bar volume / average


WEEKLY = Prm("W", "W-FRI", 26, 260, 0.40, max_brk=4, max_ext=0.15, vol_avg=20, spike=3.0)
MONTHLY = Prm("M", "ME", 12, 96, 0.60, near=0.06, max_brk=2, max_ext=0.20, touch=0.04, vol_avg=12, spike=2.5)
IPO = Prm("W", "W-FRI", 6, 156, 0.45, max_brk=4, max_ext=0.20, min_tests=2, vol_avg=20, spike=2.5)
DAY_SPIKE_ACCUM = 5.0
DAY_SPIKE_IPO = 3.0
IPO_MAX_YEARS = 3
ORDER = {"Breakout confirmed": 0, "Breaking out (bar not closed)": 1, "Near breakout": 2}
UNIT = {"W": ("week", "weeks"), "M": ("month", "months")}


def _clusters(idx: np.ndarray, gap=3) -> int:
    if len(idx) == 0:
        return 0
    return 1 + int(np.sum(np.diff(idx) > gap))


def find_base(bars: pd.DataFrame, complete: bool, p: Prm) -> dict | None:
    """Longest flat closing-basis base that price is leaving now (same rules as the laptop
    base_breakout_scanner, with the timeframe's own parameters)."""
    C = bars["Close"].values.astype(float)
    n = len(C)
    best = None
    for k in range(0, p.max_brk + 1):
        e = n - 1 - k                                   # last bar of the base
        if e + 1 < p.min_len:
            break
        hi = lo = C[e]
        length = 0
        for i in range(e, max(-1, e - p.max_len), -1):
            nh, nl = max(hi, C[i]), min(lo, C[i])
            if nh / nl - 1 > p.max_width:
                break
            hi, lo, length = nh, nl, length + 1
        if length < p.min_len:
            continue
        found = None
        for ln in range(length, p.min_len - 1, -1):
            s = e - ln + 1
            res, sup = C[s:e + 1].max(), C[s:e + 1].min()
            if k == 0:
                if e - 1 <= s:
                    continue
                res = C[s:e - 1].max()
                if not (res * (1 - p.near) <= C[-1] <= res):
                    continue
            else:
                b = e + 1
                if C[b] <= res or C[-1] < res * 0.97 or C[b:].min() < res * 0.95 or C[-1] > res * (1 + p.max_ext):
                    continue
            x = np.arange(ln, dtype=float)
            slope = np.cov(x, C[s:e + 1], bias=True)[0, 1] / np.var(x)
            if abs(slope * ln) > 0.6 * (res - sup):
                continue                                # trending, not a consolidation
            tests = _clusters(np.where(C[s:e - 1] >= res * (1 - p.touch))[0])
            if tests < p.min_tests:
                continue
            found = (s, ln, res, sup, tests)
            break
        if not found:
            continue
        s, ln, res, sup, tests = found
        status = ("Near breakout" if k == 0 else
                  "Breaking out (bar not closed)" if (k == 1 and not complete) else "Breakout confirmed")
        cand = dict(k=k, s=s, e=e, length=ln, res=float(res), sup=float(sup), tests=tests, status=status,
                    width=res / sup - 1, brk_i=(e + 1) if k else None)
        key = lambda c: (c["k"] > 0, c["length"])
        if best is None or key(cand) > key(best):
            best = cand
    return best


def _rel_vol(V: np.ndarray, avg: int, skip: int = 0) -> np.ndarray:
    v = pd.Series(V.astype(float)).replace(0, np.nan)
    v.iloc[:skip] = np.nan
    base = v.shift(1).rolling(avg, min_periods=4).mean().fillna(v.shift(1).expanding(min_periods=2).median())
    return (v / base).values


def _levels(b: dict) -> dict:
    h = b["res"] - b["sup"]
    return {"Resistance": round(b["res"], 2), "Target": round(b["res"] + h, 2),
            "Stop": round(b["res"] * 0.93 if b["k"] else b["sup"] + 0.5 * h, 2)}


def _d(idx, i):
    return idx[i].date().isoformat() if i is not None else ""


# ------------------------------------------------------------------------- accum
def accumulation(bars, daily, b, p: Prm):
    """Biggest unusual-volume UP bar (or up day) inside the base, before the breakout."""
    C, V = bars["Close"].values.astype(float), bars["Volume"].values
    rv = _rel_vol(V, p.vol_avg)
    s, e = b["s"], b["e"]
    best = None
    for i in range(max(s, 1), e + 1):
        if np.isfinite(rv[i]) and rv[i] >= p.spike and C[i] >= C[i - 1]:
            if best is None or rv[i] > best[0]:
                best = (float(rv[i]), i)
    bar = None if best is None else {"x": best[0], "date": _d(bars.index, best[1]),
                                     "before": e - best[1] + (1 if b["k"] else 0)}
    # daily print inside the base window
    start = bars.index[s] - pd.offsets.Day(40 if p.tf == "M" else 7)
    end = bars.index[e]
    dv = daily["Volume"].astype(float).replace(0, np.nan)
    drv = dv / dv.shift(1).rolling(50, min_periods=20).mean()
    up = daily["Close"] >= daily["Close"].shift(1)
    win = drv[(drv.index > start) & (drv.index <= end) & up]
    day = None
    if len(win) and np.isfinite(win.max()) and win.max() >= DAY_SPIKE_ACCUM:
        day = {"x": float(win.max()), "date": win.idxmax().date().isoformat()}
    return bar, day


def scan_accum(sym, daily, now) -> dict | None:
    daily = daily.dropna(subset=["High", "Low", "Close"])
    out = []
    for p in (WEEKLY, MONTHLY):
        bars = fp.resample(daily, p.rule)
        if len(bars) < p.min_len + 2:
            continue
        b = find_base(bars, ip.last_bar_complete(daily, p.tf, now), p)
        if not b:
            continue
        bar, day = accumulation(bars, daily, b, p)
        if not (bar or day):
            continue
        out.append((p, bars, b, bar, day))
    if not out:
        return None
    # prefer a real breakout, then the longer base in calendar time
    p, bars, b, bar, day = sorted(out, key=lambda t: (ORDER[t[2]["status"]],
                                                      -t[2]["length"] * (4.345 if t[0].tf == "M" else 1)))[0]
    one, many = UNIT[p.tf]
    months = b["length"] if p.tf == "M" else b["length"] / 4.345
    c = float(daily["Close"].iloc[-1])
    vol = []
    if bar:
        vol.append(f'{one} {bar["x"]:.1f}x avg ({fp_date(bar["date"])}, '
                   f'{bar["before"]} {one if bar["before"] == 1 else many} before {"breakout" if b["k"] else "now"})')
    if day:
        vol.append(f'day {day["x"]:.1f}x 50-day avg ({fp_date(day["date"])})')
    tfs = "+".join(sorted({t[0].tf for t in out}, reverse=True))
    return {
        "status": f'{b["status"]} ({"weekly" if p.tf == "W" else "monthly"})',
        "facts": [["Base", f'{months:.0f} months on the {"weekly" if p.tf == "W" else "monthly"} chart, '
                           f'{b["width"] * 100:.0f}% wide, top tested {b["tests"]}x'],
                  ["Vs resistance", f'{(c / b["res"] - 1) * 100:+.1f}%'],
                  ["Volume inside base", "; ".join(vol)],
                  ["Also on", "weekly and monthly charts"] if tfs == "W+M" else ["", ""]],
        "levels": _levels(b),
        "sort": ORDER[b["status"]] * 1000 - months,
    }


def fp_date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10]).strftime("%d %b %Y")
    except Exception:
        return str(s or "")


# ------------------------------------------------------------------------- ipo
def scan_ipo(sym, daily, now, listed: dt.date | None) -> dict | None:
    daily = daily.dropna(subset=["High", "Low", "Close"])
    first = daily.index[0].date()
    lst = listed or first
    today = daily.index[-1].date()
    if (today - lst).days > IPO_MAX_YEARS * 365 or (today - first).days > IPO_MAX_YEARS * 365 + 30:
        return None
    w = fp.resample(daily, IPO.rule)
    w = w.iloc[1:]                                       # drop the listing week
    if len(w) < IPO.min_len + 2:
        return None
    b = find_base(w, ip.last_bar_complete(daily, "W", now), IPO)
    if not b:
        return None
    V = w["Volume"].values.astype(float)
    base_avg = np.nanmean(np.where(V[b["s"]:b["e"] + 1] > 0, V[b["s"]:b["e"] + 1], np.nan))
    lo_i = max(0, len(w) - 4)
    wk = V[lo_i:] / base_avg if base_avg and np.isfinite(base_avg) else np.full(len(w) - lo_i, np.nan)
    j = int(np.nanargmax(wk)) if np.isfinite(wk).any() else None
    wk_x = float(wk[j]) if j is not None else np.nan
    dv = daily["Volume"].astype(float).replace(0, np.nan).iloc[5:]           # skip listing days
    drv = (dv / dv.shift(1).rolling(50, min_periods=15).mean()).iloc[-20:]
    day_x = float(drv.max()) if len(drv) and np.isfinite(drv.max()) else np.nan
    if not ((np.isfinite(wk_x) and wk_x >= IPO.spike) or (np.isfinite(day_x) and day_x >= DAY_SPIKE_IPO)):
        return None
    c = float(daily["Close"].iloc[-1])
    ipo_px = float(daily["Close"].iloc[0])
    vol = []
    if np.isfinite(wk_x) and wk_x >= IPO.spike:
        vol.append(f'week {wk_x:.1f}x base avg ({fp_date(_d(w.index, lo_i + j))})')
    if np.isfinite(day_x) and day_x >= DAY_SPIKE_IPO:
        vol.append(f'day {day_x:.1f}x 50-day avg ({fp_date(drv.idxmax().date().isoformat())})')
    age = (today - lst).days / 30.44
    return {
        "status": b["status"].replace("bar not closed", "week not closed"),
        "facts": [["Listed", f'{fp_date(lst.isoformat())} ({age:.0f} months ago)'
                             + ("" if listed else " - first trading day on Yahoo")],
                  ["IPO base", f'{b["length"]} weeks, {b["width"] * 100:.0f}% wide, top tested {b["tests"]}x'],
                  ["Vs resistance", f'{(c / b["res"] - 1) * 100:+.1f}%'],
                  ["From listing-day close", f'{(c / ipo_px - 1) * 100:+.0f}%'],
                  ["Volume", ", ".join(vol)]],
        "levels": _levels(b),
        "sort": ORDER[b["status"]] * 1000 + (today - lst).days / 30,
    }
