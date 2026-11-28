#!/usr/bin/env python3
"""
RSI Momentum scan for the EQUITY ESSENCE app (used by app_scan.py).

  Monthly RSI(14) > 60, weekly RSI(14) > 60 and daily RSI(14) > 50 - momentum on all three timeframes.
  RSI is Wilder's 14-period RSI on closing prices (the TradingView / Kite default). The current month
  and week count as they stand today (not yet closed), as most screeners show them.
  Like every list, only companies holding regular concalls are kept (filter applied in app_scan.py).

Educational use only - not investment advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

M_MIN, W_MIN, D_MIN = 60.0, 60.0, 50.0
N = 14


def rsi(close: pd.Series, n: int = N) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    return out.where(dn > 0, 100.0).where(up.notna())


def scan(d: pd.DataFrame, show_all: bool = False) -> dict | None:
    """Card if monthly > 60, weekly > 60, daily > 50. show_all=True returns a card for any stock
    with enough history (with "pass" set), used by the ETF view."""
    c = d["Close"].dropna()
    if len(c) < 330:                                   # need ~15 months for a monthly RSI
        return None
    rd = rsi(c)
    rw = rsi(c.resample("W-FRI").last().dropna())
    rm = rsi(c.resample("ME").last().dropna())
    if rm.dropna().empty or rw.dropna().empty:
        return None
    m, w, dd = float(rm.iloc[-1]), float(rw.iloc[-1]), float(rd.iloc[-1])
    ok = m > M_MIN and w > W_MIN and dd > D_MIN
    if not ok and not show_all:
        return None
    # how long monthly RSI has been above 60, and when the daily last crossed above 50
    above = (rm > M_MIN).values[::-1]
    months = int(np.argmin(above)) if not above.all() else len(above)
    cross = rd[(rd > D_MIN) & (rd.shift(1) <= D_MIN)]
    since = cross.index[-1] if len(cross) else None
    px = float(c.iloc[-1])
    hi52 = float(d["High"].iloc[-252:].max())
    facts = [["RSI monthly / weekly / daily", f"{m:.1f} / {w:.1f} / {dd:.1f}"],
             ["Monthly RSI above 60", f"{months} month{'s' if months != 1 else ''}" + (" (new)" if months <= 1 else "")],
             ["Daily RSI above 50 since", since.strftime("%d %b %Y") if since is not None else "over a year"],
             ["From 52-week high", f"{(px / hi52 - 1) * 100:+.1f}%"]]
    if not ok:
        miss = [f"{n} {v:.1f} (needs > {t:.0f})" for n, v, t in (("monthly", m, M_MIN), ("weekly", w, W_MIN), ("daily", dd, D_MIN)) if not v > t]
        return {"status": "Not yet: " + ", ".join(miss), "facts": facts[:1] + facts[3:], "levels": {},
                "sort": -(m + w + dd), "pass": False, "rsi": [round(m, 1), round(w, 1), round(dd, 1)]}
    status = ("Fresh: monthly RSI just crossed 60" if months <= 1 else
              "Daily RSI just turned above 50" if since is not None and since >= c.index[-5] else
              "Momentum on all three timeframes")
    return {"status": status, "facts": facts,
            "levels": {"50-DMA": round(float(c.rolling(50).mean().iloc[-1]), 2),
                       "52w high": round(hi52, 2)},
            "sort": -(m + w + dd), "pass": True, "rsi": [round(m, 1), round(w, 1), round(dd, 1)]}
