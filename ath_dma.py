#!/usr/bin/env python3
"""
First all-time high, moving-average crossovers and IPO lock-in dates for the EQUITY ESSENCE app.
Used by app_scan.py (main lists) and ipo_zone.py (IPO Zone).

first_ath   The stock went above its previous all-time high (intraday high) in the last 5 sessions
            for the first time after that high had stood for 3+ months (63+ trading days), and it
            still closes within 5% of the new high. Stocks that keep making new highs every week
            are NOT listed - only the first break after a long gap.
            Main list: stocks listed more than 5 years ago (recent IPOs are in IPO Zone, no overlap).
            Their full price history (Yahoo "max") is checked so "all-time" really means all-time.

dma_cross   Close crossed above the simple moving average in the last 3 sessions and is still above
            it, after at least 8 of the 10 closes before the cross were below it (a fresh cross,
            not a stock chopping around the average).
            Main list: 50 / 150 / 200-day. IPO Zone: 9 / 21 / 50-day.

lockins     Standard SEBI lock-in expiry dates, counted from the allotment date (taken as one
            trading day before listing): anchor investors 50% after 30 days and 50% after 90 days
            (30 days for all before April 2022), pre-IPO shareholders 6 months (1 year before
            Aug 2021), promoters' excess holding 6 months and minimum contribution (20%) 18 months
            (1 and 3 years before Aug 2021; 3 years also applies when IPO money funds capex).
            Estimates - the prospectus and NSE lock-in filings give the exact dates.

Educational use only - not investment advice.
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

ATH_GAP = 63          # trading days the previous high must have stood
ATH_RECENT = 5        # sessions in which the break must have happened
DMA_WINDOW = 3
DMA_PRIOR = 10
DMA_PRIOR_BELOW = 8


def _f(d):
    return d.strftime("%d %b %Y")


# ------------------------------------------------------------------------- first ATH
def first_ath(d: pd.DataFrame, full_high: pd.Series | None = None) -> dict | None:
    d = d.dropna(subset=["High", "Low", "Close"])
    if len(d) < ATH_GAP + ATH_RECENT + 5:
        return None
    H, C = d["High"], d["Close"]
    hist = full_high if full_high is not None and len(full_high) > len(H) else H
    hist = hist[hist.index < d.index[-ATH_RECENT]]
    if hist.empty:
        return None
    prior = float(hist.max())
    recent = H.iloc[-ATH_RECENT:]
    if recent.max() <= prior:
        return None
    first_day = recent[recent > prior].index[0]
    prior_day = hist.idxmax()
    gap = int(((H.index > prior_day) & (H.index < first_day)).sum()) if prior_day >= H.index[0] else 9999
    if gap < ATH_GAP:
        return None
    ath = float(recent.max())
    px = float(C.iloc[-1])
    if px < ath * 0.95:
        return None
    v = d["Volume"].astype(float).replace(0, np.nan)
    rv = v / v.shift(1).rolling(50, min_periods=20).mean()
    vx = rv.loc[first_day:].max()
    months = gap / 21
    return {
        "status": f"First all-time high in {months / 12:.1f} years" if months >= 12 else f"First all-time high in {months:.0f} months",
        "facts": [["Previous all-time high", f"₹{prior:,.2f} on {_f(prior_day)}"],
                  ["Crossed on", f"{_f(first_day)} · new high ₹{ath:,.2f}"],
                  ["Close vs previous high", f"{(px / prior - 1) * 100:+.1f}%"],
                  ["Volume on the break", f"{vx:.1f}x 50-day average" if np.isfinite(vx) else ""]],
        "levels": {"Previous ATH": round(prior, 2), "New ATH": round(ath, 2), "Stop": round(prior * 0.93, 2)},
        "sort": -gap,
    }


def full_highs(symbols: list[str], log=print) -> dict:
    """Full-history daily highs from Yahoo for ATH candidates (the main download is 7 years)."""
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), 40):
        chunk = [s + ".NS" for s in symbols[i:i + 40]]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period="max", interval="1d", auto_adjust=True, group_by="ticker",
                                 threads=True, progress=False)
                break
            except Exception as e:
                log(f"  max history retry ({e})")
                time.sleep(3 * (attempt + 1))
        else:
            continue
        for t in chunk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                h = sub["High"].dropna()
                if len(h):
                    out[t[:-3]] = h
            except Exception:
                pass
    return out


# ------------------------------------------------------------------------- DMA cross
def dma_cross(d: pd.DataFrame, periods=(50, 150, 200)) -> dict | None:
    d = d.dropna(subset=["Close"])
    C = d["Close"]
    v = d["Volume"].astype(float).replace(0, np.nan)
    rv = v / v.shift(1).rolling(50, min_periods=10).mean()
    tags, facts, levels = [], [], {}
    px = float(C.iloc[-1])
    best_vol = 0.0
    for p in periods:
        if len(C) < p + DMA_PRIOR + DMA_WINDOW + 1:
            continue
        ma = C.rolling(p).mean()
        if not px > ma.iloc[-1]:
            continue
        for k in range(1, DMA_WINDOW + 1):                     # cross on bar -k
            i = len(C) - k
            if C.iloc[i] > ma.iloc[i] and C.iloc[i - 1] <= ma.iloc[i - 1]:
                pre = (C.iloc[i - DMA_PRIOR:i] <= ma.iloc[i - DMA_PRIOR:i]).sum()
                if pre >= DMA_PRIOR_BELOW and (C.iloc[i:] > ma.iloc[i:]).all():
                    slope = ma.iloc[-1] / ma.iloc[-11] - 1
                    vx = rv.iloc[i]
                    tags.append(str(p))
                    facts.append([f"{p}-DMA", f"crossed {_f(C.index[i])} · now {(px / ma.iloc[-1] - 1) * 100:+.1f}% above · "
                                              f"average {'rising' if slope > 0.002 else 'falling' if slope < -0.002 else 'flat'}"
                                              + (f" · volume {vx:.1f}x" if np.isfinite(vx) else "")])
                    levels[f"{p}-DMA"] = round(float(ma.iloc[-1]), 2)
                    best_vol = max(best_vol, vx if np.isfinite(vx) else 0)
                break
    if not tags:
        return None
    pos = []
    for p in sorted(set(periods) | {50, 200}):
        if len(C) > p and str(p) not in tags:
            pos.append(f"{'above' if px > C.rolling(p).mean().iloc[-1] else 'below'} {p}-DMA")
    if pos:
        facts.append(["Other averages", ", ".join(pos)])
    return {"status": "Crossed above " + " & ".join(f"{t}-DMA" for t in tags),
            "facts": facts, "levels": dict(list(levels.items())[:3]), "tags": tags,
            "sort": -len(tags) * 100 - best_vol}


# ------------------------------------------------------------------------- lock-ins
def lockins(listed: dt.date, today: dt.date) -> list[dict]:
    allot = pd.Timestamp(listed) - pd.offsets.BDay(1)
    new_rules = listed >= dt.date(2021, 8, 13)
    anchor_split = listed >= dt.date(2022, 4, 1)
    rows = []
    if anchor_split:
        rows += [("Anchor investors (50%)", allot + pd.DateOffset(days=30)),
                 ("Anchor investors (other 50%)", allot + pd.DateOffset(days=90))]
    else:
        rows += [("Anchor investors", allot + pd.DateOffset(days=30))]
    rows += [("Pre-IPO shareholders", allot + pd.DateOffset(months=6 if new_rules else 12))]
    rows += [("Promoters - excess holding", allot + pd.DateOffset(months=6 if new_rules else 12)),
             ("Promoters - minimum 20%", allot + pd.DateOffset(months=18 if new_rules else 36))]
    out = []
    for who, when in rows:
        w = when.date()
        out.append({"who": who, "date": w.isoformat(), "past": w < today,
                    "days": (w - today).days})
    return out
