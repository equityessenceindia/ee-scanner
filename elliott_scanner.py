#!/usr/bin/env python3
"""
Elliott Wave "first impulse" scanner for NSE small & micro caps.

What it does
------------
1. Builds a universe of NSE stocks with market cap >= 500 crore, excluding the
   top 250 by market cap (AMFI-style large + mid caps), i.e. small + micro caps.
2. For every stock it finds the major low of the lookback window (default ~2 years)
   and checks that this low ended a real decline (>= 30% drawdown) or is a post-IPO low.
   Only then is the rally off that low treated as a FIRST-TIME impulse.
3. It counts swings from that low with a volatility-scaled ZigZag, applies the
   Elliott rules (wave 2 never below the start of wave 1, wave 3 must clear wave 1,
   wave 4 must not overlap wave 1, wave 3 never the shortest), merges flat/double
   wave-2 corrections (incl. expanded flats), and labels the CURRENT stage.
4. Writes an Excel file and an HTML report with a mini chart of each count.

Run:  python elliott_scanner.py
      python elliott_scanner.py --symbols NEOGEN FLUOROCHEM SFCTECH
      python elliott_scanner.py --universe screener_export.csv
This is a mechanical swing counter, not investment advice. Always check the chart.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
CACHE_DIR = HERE / "cache"


@dataclass
class Config:
    min_mcap_cr: float = 500.0
    exclude_top: int = 250          # drop the largest N by market cap (large + mid caps)
    max_mcap_cr: float | None = None
    years: int = 3                  # price history to download
    lookback_bars: int = 500        # window in which the major low is searched (~2 years)
    prior_window: int = 250         # bars before the low used to measure the prior decline
    min_prior_drop: float = 0.30    # low must be >= 30% below the prior high (or be an IPO low)
    min_w1: float = 0.15            # wave 1 must rise at least 15% from the low
    min_bars: int = 5               # need at least this many bars after the low
    zz_min: float = 0.07            # ZigZag reversal threshold bounds
    zz_max: float = 0.25
    zz_mults: tuple = (1.0, 1.4, 2.0)  # degrees tried, finest first
    vol_avg: int = 50               # sessions in the "normal volume" average
    vol_spike: float = 5.0          # unusual volume = at least 5x that average (small caps spike a lot)


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------
NSE_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}


def load_nse_list() -> pd.DataFrame:
    import requests
    last = None
    for url in NSE_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip().upper() for c in df.columns]
            df = df[df["SERIES"].astype(str).str.strip().isin(["EQ", "BE"])]
            df = df.rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "name"})
            df["symbol"] = df["symbol"].astype(str).str.strip()
            return df[["symbol", "name"]].reset_index(drop=True)
        except Exception as e:  # try the next mirror
            last = e
    raise RuntimeError(f"Could not download the NSE equity list ({last}). "
                       f"Use --universe with a CSV of symbols instead.")


def _fetch_mcap(sym: str):
    import yfinance as yf
    try:
        fi = yf.Ticker(sym + ".NS").fast_info
        mc = None
        try:
            mc = fi.market_cap
        except Exception:
            pass
        if not mc:
            try:
                mc = fi.shares * fi.last_price
            except Exception:
                mc = None
        return sym, (float(mc) / 1e7 if mc else None)   # rupees -> crore
    except Exception:
        return sym, None


def _quiet_yf():
    import logging
    for name in ("yfinance", "peewee", "urllib3"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def fetch_mcaps(symbols: list[str], workers: int = 8, label="market caps") -> dict:
    """Fetch in parallel, then retry the ones Yahoo rate-limited, slowly."""
    _quiet_yf()
    out, done = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_mcap, s) for s in symbols]
        for f in as_completed(futs):
            s, mc = f.result()
            out[s] = mc
            done += 1
            if done % 100 == 0 or done == len(symbols):
                print(f"  {label}: {done}/{len(symbols)}", flush=True)
    for attempt in range(3):
        missing = [s for s, v in out.items() if not v]
        if not missing:
            break
        print(f"  retrying {len(missing)} stocks Yahoo rate-limited (pass {attempt + 1}/3) ...",
              flush=True)
        time.sleep(30)
        for i, s in enumerate(missing, 1):
            out[s] = _fetch_mcap(s)[1]
            time.sleep(0.35)
            if i % 100 == 0:
                print(f"    {i}/{len(missing)}", flush=True)
    got = sum(1 for v in out.values() if v)
    print(f"  market caps found for {got}/{len(out)} stocks", flush=True)
    return out


def _pick_col(cols, *needles):
    for c in cols:
        lc = c.lower().strip()
        if any(n in lc for n in needles):
            return c
    return None


def build_universe(args, cfg: Config) -> pd.DataFrame:
    if args.symbols:
        df = pd.DataFrame({"symbol": [s.upper().replace(".NS", "") for s in args.symbols]})
        df["name"] = df["symbol"]
        df["mcap_cr"] = np.nan
        return df  # explicit list: no market-cap filter

    if args.universe:
        raw = pd.read_csv(args.universe)
        scol = _pick_col(raw.columns, "nse code", "nse_code", "symbol", "ticker")
        if scol is None:
            sys.exit("Universe CSV needs a column named Symbol / NSE Code / Ticker.")
        mcol = _pick_col(raw.columns, "market cap", "mcap", "mar cap")
        ncol = _pick_col(raw.columns, "name")
        df = pd.DataFrame({"symbol": raw[scol].astype(str).str.strip().str.upper()})
        df["name"] = raw[ncol].astype(str) if ncol else df["symbol"]
        df = df[df["symbol"].str.len() > 0]
        if mcol:
            df["mcap_cr"] = pd.to_numeric(raw[mcol], errors="coerce")
        else:
            print("Fetching market caps for the universe file ...")
            mc = fetch_mcaps(df["symbol"].tolist())
            df["mcap_cr"] = df["symbol"].map(mc)
    else:
        CACHE_DIR.mkdir(exist_ok=True)
        cache = CACHE_DIR / "nse_universe_mcap.csv"
        fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400
        if fresh and not args.refresh:
            print(f"Using cached universe ({cache.name}, refreshed weekly; --refresh to force).")
            df = pd.read_csv(cache)
            miss = df["mcap_cr"].isna()
            if miss.mean() > 0.05:
                print(f"{int(miss.sum())} stocks have no market cap in the cache - filling them in ...")
                mc = fetch_mcaps(df.loc[miss, "symbol"].tolist(), workers=4)
                df.loc[miss, "mcap_cr"] = df.loc[miss, "symbol"].map(mc)
                df.to_csv(cache, index=False)
        else:
            print("Downloading NSE equity list ...")
            df = load_nse_list()
            print(f"Fetching market caps for {len(df)} stocks (one-time, ~2-5 min) ...")
            mc = fetch_mcaps(df["symbol"].tolist())
            df["mcap_cr"] = df["symbol"].map(mc)
            df.to_csv(cache, index=False)

    df = df.dropna(subset=["mcap_cr"]).sort_values("mcap_cr", ascending=False).reset_index(drop=True)
    df["mcap_rank"] = np.arange(1, len(df) + 1)
    keep = (df["mcap_rank"] > cfg.exclude_top) & (df["mcap_cr"] >= cfg.min_mcap_cr)
    if cfg.max_mcap_cr:
        keep &= df["mcap_cr"] <= cfg.max_mcap_cr
    df = df[keep].reset_index(drop=True)
    print(f"Universe: {len(df)} small/micro caps (mcap >= {cfg.min_mcap_cr:.0f} cr, "
          f"excluding top {cfg.exclude_top}).")
    return df


# --------------------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------------------
def download_prices(symbols: list[str], years: int, batch: int = 80) -> dict:
    import yfinance as yf
    _quiet_yf()
    data = {}
    tickers = [s + ".NS" for s in symbols]
    for k in range(0, len(tickers), batch):
        chunk = tickers[k:k + batch]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=f"{years}y", interval="1d", auto_adjust=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print(f"  retry ({e})")
                time.sleep(3 * (attempt + 1))
        if df is None or df.empty:
            continue
        multi = isinstance(df.columns, pd.MultiIndex)
        lvl0 = set(df.columns.get_level_values(0)) if multi else set()
        for t in chunk:
            try:
                if multi:
                    if t not in lvl0:
                        continue
                    sub = df[t]
                else:
                    sub = df
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                sub = sub[(sub["High"] > 0) & (sub["Low"] > 0)]
                if len(sub) >= 20:
                    data[t[:-3]] = sub.copy()
            except Exception:
                continue
        print(f"  prices: {min(k + batch, len(tickers))}/{len(tickers)}", flush=True)
    return data


# --------------------------------------------------------------------------------------
# Wave engine
# --------------------------------------------------------------------------------------
def zigzag_from(high, low, start, pct):
    """Swing points starting at a known low. The last point is the live (unconfirmed) extreme."""
    piv = [(start, float(low[start]))]
    up, ext = True, start
    for i in range(start + 1, len(high)):
        if up:
            if high[i] >= high[ext]:
                ext = i
            elif low[i] <= high[ext] * (1 - pct):
                piv.append((ext, float(high[ext])))
                up, ext = False, i
        else:
            if low[i] <= low[ext]:
                ext = i
            elif high[i] >= low[ext] * (1 + pct):
                piv.append((ext, float(low[ext])))
                up, ext = True, i
    if ext != piv[-1][0]:
        piv.append((ext, float(high[ext] if up else low[ext])))
    return piv


def merge_correction(pts, k):
    """Fold flats / double corrections into one wave.
    pts[k] is a correction low. If the next bounce (pts[k+1]) stays within 138.2% of the
    correction so far (a B wave, possibly above the prior high = expanded flat) and the
    following low (pts[k+2]) undercuts pts[k], the correction is still running."""
    kinds = []
    while len(pts) > k + 2:
        hi_prev, lo, b, c = pts[k - 1][1], pts[k][1], pts[k + 1][1], pts[k + 2][1]
        if c < lo and b <= lo + 1.382 * (hi_prev - lo):
            kinds.append("expanded flat" if b > hi_prev else "flat / double correction")
            pts = pts[:k] + pts[k + 2:]
        else:
            break
    return pts, kinds


STAGES = {
    # code: (label, sort order, is first-cycle setup)
    "W2B": ("Wave 2 - buy zone (50-78.6%)", 1, True),
    "W3A": ("Wave 3 starting - below W1 high", 2, True),
    "W3N": ("Wave 3 - (i)/(ii) pullback", 3, True),
    "W2":  ("Wave 2 in progress", 4, True),
    "W3":  ("Wave 3 confirmed", 5, True),
    "W1":  ("Wave 1 in progress", 6, True),
    "W4":  ("Wave 4 in progress", 7, True),
    "W5A": ("Wave 5 starting - below W3 high", 8, True),
    "W5":  ("Wave 5 in progress", 9, True),
    "B":   ("Basing - no wave 1 yet", 10, False),
    "Z":   ("Impulse complete - in correction", 11, False),
    "X":   ("Ambiguous / count invalid", 12, False),
    "N":   ("Not a first wave (no prior decline)", 13, False),
    "E":   ("Too little data", 14, False),
}


def _vol(V, a, b):
    seg = V[a + 1:b + 1]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if len(seg) else np.nan


def _ratio(a, b):
    return round(a / b, 2) if (b and np.isfinite(a) and np.isfinite(b) and b > 0) else np.nan


def classify(df: pd.DataFrame, o: int, pct: float, cfg: Config) -> dict:
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    V = df["Volume"].values.astype(float)
    pts = zigzag_from(H, L, o, pct)
    c = float(C[-1])
    r = {"zz_pct": round(pct * 100, 1), "notes": []}

    def done(code, detail, labels, inval=None, trigger=None, t1=None, t2=None, **kw):
        r.update(code=code, stage=STAGES[code][0], detail=detail, pts=pts[:len(labels)],
                 labels=labels, invalidation=inval, trigger=trigger, t1=t1, t2=t2, **kw)
        return r

    if len(pts) < 2:
        return done("B", "No swing yet off the low", ["0"])
    P = lambda k: pts[k][1]
    I = lambda k: pts[k][0]
    gain1 = P(1) / P(0) - 1
    if gain1 < cfg.min_w1:
        if len(pts) == 2:
            return done("B", f"Up {gain1:.0%} from the low - wave 1 not yet {cfg.min_w1:.0%}",
                        ["0", "1?"], inval=P(0))
        return done("X", f"First swing only {gain1:.0%} at this degree", ["0"])

    # fold complex wave-2 corrections (flats, expanded flats, doubles)
    pts, kinds = merge_correction(pts, 2)
    if kinds:
        r["notes"].append("W2 = " + ", ".join(kinds))
    if len(pts) > 6 and P(3) > P(1) and P(4) > P(1):
        pts, k4 = merge_correction(pts, 4)
        if k4:
            r["notes"].append("W4 = " + ", ".join(k4))
    legs = len(pts) - 1
    w1 = P(1) - P(0)
    v1 = _vol(V, I(0), I(1))
    r["w1_gain_pct"] = round(gain1 * 100, 1)

    if legs == 1:
        return done("W1", f"Up {gain1:.0%} from the low; watch for the first pullback",
                    ["0", "1?"], inval=P(0),
                    t1=P(1) - 0.5 * w1, t2=P(1) - 0.618 * w1,
                    target_note="t1/t2 = projected W2 zone 50% / 61.8%")

    ret2 = (P(1) - P(2)) / w1
    r["w2_retrace_pct"] = round(ret2 * 100, 1)
    r["vol_w2_w1"] = _ratio(_vol(V, I(1), I(2)), v1)

    if legs == 2:
        code = "W2B" if 0.5 <= ret2 <= 0.786 else "W2"
        depth = ("shallow (<38.2%)" if ret2 < 0.382 else "38.2-50%" if ret2 < 0.5
                 else "50-78.6% (classic W2)" if ret2 <= 0.786 else "deep (>78.6%)")
        return done(code, f"Retraced {ret2:.0%} of W1 - {depth}", ["0", "1", "2?"],
                    inval=P(0), trigger=P(1), t1=P(2) + w1, t2=P(2) + 1.618 * w1,
                    target_note="trigger = W1 high; t1/t2 = W3 at 1.0x / 1.618x W1")

    v3 = _vol(V, I(2), I(3))
    r["vol_w3_w1"] = _ratio(v3, v1)
    ext3 = (P(3) - P(2)) / w1
    r["w3_ext"] = round(ext3, 2)

    if legs == 3:
        if P(3) <= P(1):
            return done("W3A", f"W2 ended at {ret2:.0%} retrace; rally is {ext3:.2f}x W1, "
                               f"needs close above W1 high", ["0", "1", "2", "3?"],
                        inval=P(2), trigger=P(1), t1=P(2) + w1, t2=P(2) + 1.618 * w1,
                        target_note="trigger = W1 high; t1/t2 = W3 at 1.0x / 1.618x W1")
        t1, t2 = P(2) + 1.618 * w1, P(2) + 2.618 * w1
        if c > t1:
            t1, t2 = t2, P(2) + 4.236 * w1
        return done("W3", f"Above W1 high; W3 = {ext3:.2f}x W1 so far", ["0", "1", "2", "3?"],
                    inval=P(2), t1=t1, t2=t2, target_note="W3 Fibonacci extensions of W1")

    # legs >= 4
    if P(3) > P(1) and P(4) > P(1):          # proper wave 4 (no overlap with W1)
        w3 = P(3) - P(2)
        ret4 = (P(3) - P(4)) / w3
        r["w4_retrace_pct"] = round(ret4 * 100, 1)
        t1, t2 = P(4) + w1, P(4) + 0.618 * (P(3) - P(0))
        if legs == 4:
            return done("W4", f"W3 = {ext3:.2f}x W1; W4 retraced {ret4:.0%} of W3",
                        ["0", "1", "2", "3", "4?"], inval=P(1), trigger=P(3),
                        t1=min(t1, t2), t2=max(t1, t2),
                        target_note="invalidation = W1 high (overlap); t1/t2 = W5 targets")
        if legs == 5:
            w5 = P(5) - P(4)
            if w3 < w1 and w3 < w5:
                return done("X", "Wave 3 would be the shortest - not a valid impulse",
                            ["0", "1", "2", "3", "4", "5?"])
            code = "W5A" if P(5) <= P(3) else "W5"
            msg = ("Rising from W4, still below W3 high" if code == "W5A"
                   else f"New high above W3; W5 = {w5 / w1:.2f}x W1")
            return done(code, msg, ["0", "1", "2", "3", "4", "5?"], inval=P(4),
                        trigger=P(3) if code == "W5A" else None,
                        t1=min(t1, t2), t2=max(t1, t2),
                        target_note="late stage - W5 targets (W5 = W1, 0.618 x W1-W3)")
        if P(5) > P(3):
            fall = (P(5) - P(6)) / (P(5) - P(0))
            return done("Z", f"5 waves done at {P(5):.1f}; corrected {fall:.0%} of the impulse",
                        ["0", "1", "2", "3", "4", "5", "A?"])
        return done("X", "Complex structure after wave 4", ["0", "1", "2", "3", "4"])

    if P(4) > P(2):                           # nested (i)-(ii) inside wave 3
        trig = max(P(1), P(3))
        if legs == 4:
            return done("W3N", "Higher low after W2 - (i)/(ii) of wave 3; break above "
                               "the (i) high starts (iii)", ["0", "1", "2", "(i)", "(ii)?"],
                        inval=P(2), trigger=trig, t1=P(2) + 1.618 * w1, t2=P(2) + 2.618 * w1,
                        target_note="invalidation = W2 low; t1/t2 = W3 at 1.618x / 2.618x W1")
        if legs == 5 and P(5) > trig:
            return done("W3", "Wave (iii) of 3 underway - above the (i) and W1 highs",
                        ["0", "1", "2", "(i)", "(ii)", "(iii)?"], inval=P(4),
                        t1=P(2) + 1.618 * w1, t2=P(2) + 2.618 * w1,
                        target_note="invalidation = (ii) low")
    return done("X", "Swings overlap - no clean impulse at this degree", ["0", "1", "2"])


def find_origin(df: pd.DataFrame, cfg: Config):
    n = len(df)
    w0 = max(0, n - cfg.lookback_bars)
    lows = df["Low"].values
    o = w0 + int(np.argmin(lows[w0:]))
    ps = max(0, o - cfg.prior_window)
    prior_high = float(df["High"].values[ps:o + 1].max())
    drop = 1 - lows[o] / prior_high if prior_high > 0 else 0.0
    ipo = (o <= 15) and (n < cfg.lookback_bars)   # history starts at/near the low
    return o, drop, ipo


def analyse(sym: str, df: pd.DataFrame, cfg: Config) -> dict:
    df = df.dropna(subset=["High", "Low", "Close"])
    base = {"symbol": sym, "price": round(float(df["Close"].iloc[-1]), 2),
            "date": df.index[-1].date().isoformat() if hasattr(df.index[-1], "date") else ""}
    if len(df) < 20:
        return {**base, "code": "E", "stage": STAGES["E"][0], "detail": f"{len(df)} bars"}
    o, drop, ipo = find_origin(df, cfg)
    base.update(origin_date=df.index[o].date().isoformat(),
                origin_price=round(float(df["Low"].iloc[o]), 2),
                prior_drop_pct=round(drop * 100, 1),
                bars_since_low=len(df) - 1 - o,
                ipo_base=bool(ipo))
    if len(df) - 1 - o < cfg.min_bars:
        return {**base, "code": "B", "stage": STAGES["B"][0], "detail": "Low is too recent",
                "_df": df, "_o": o, "pts": [], "labels": []}
    if drop < cfg.min_prior_drop and not ipo:
        return {**base, "code": "N", "stage": STAGES["N"][0],
                "detail": f"Low was only {drop:.0%} below the prior high - trend continuation"}

    rng = (df["High"] - df["Low"]) / df["Close"]
    atr_pct = float(rng.iloc[-120:].median())
    base_pct = 3.0 * atr_pct
    best = None
    for m in cfg.zz_mults:
        pct = float(np.clip(base_pct * m, cfg.zz_min, cfg.zz_max))
        res = classify(df, o, pct, cfg)
        if best is None or STAGES[res["code"]][2]:
            best = res
        if STAGES[res["code"]][2]:
            break
    best = {**base, **best, "_df": df, "_o": o}
    if ipo:
        best["notes"] = ["post-IPO base"] + best.get("notes", [])
    if STAGES[best["code"]][2]:
        best.update(volume_events(df, best.get("pts") or [], cfg))
    return finalize(best)


def _rel_volume(df: pd.DataFrame, avg: int) -> np.ndarray:
    """Volume divided by the average of the previous `avg` sessions (today excluded)."""
    v = df["Volume"].astype(float).replace(0, np.nan)
    base = v.shift(1).rolling(avg, min_periods=max(10, avg // 3)).mean()
    return (v / base).values


def _spike_in(rv, V, idx, lo, hi):
    lo, hi = max(0, lo), min(len(rv) - 1, hi)
    if hi < lo:
        return None
    seg = rv[lo:hi + 1]
    if not np.isfinite(seg).any():
        return None
    k = lo + int(np.nanargmax(seg))
    return {"ratio": round(float(rv[k]), 2), "date": idx[k].date().isoformat(),
            "volume": int(V[k]) if np.isfinite(V[k]) else None}


def volume_events(df: pd.DataFrame, pts: list, cfg: Config) -> dict:
    """Biggest relative-volume day around the end of wave 1 and the start of wave 3.

    End of W1  : the 6 sessions up to the W1 top plus 2 after it (climax / distribution).
    Start of W3: from the W2 low up to 8 sessions later, plus the day price first closes
                 above the W1 high and the day after (breakout volume)."""
    out = {}
    if len(pts) < 3:
        return out
    rv = _rel_volume(df, cfg.vol_avg)
    V = df["Volume"].astype(float).values
    C = df["Close"].values
    idx = df.index
    i1, p1 = pts[1]
    e = _spike_in(rv, V, idx, i1 - 5, i1 + 2)
    if e:
        out.update(w1_spike=e["ratio"], w1_spike_date=e["date"], w1_spike_vol=e["volume"],
                   w1_spike_flag=e["ratio"] >= cfg.vol_spike)
    if len(pts) < 4:
        return out
    i2 = pts[2][0]
    cands = []
    e = _spike_in(rv, V, idx, i2, i2 + 8)
    if e:
        cands.append(e)
    above = np.where(C[i2 + 1:] > p1)[0]
    if len(above):
        b = i2 + 1 + int(above[0])
        out["w3_breakout_date"] = idx[b].date().isoformat()
        e = _spike_in(rv, V, idx, b, b + 1)
        if e:
            cands.append(e)
    if cands:
        e = max(cands, key=lambda d: d["ratio"])
        out.update(w3_spike=e["ratio"], w3_spike_date=e["date"], w3_spike_vol=e["volume"],
                   w3_spike_flag=e["ratio"] >= cfg.vol_spike)
    return out


# --------------------------------------------------------------------------------------
# NSE delivery data (daily "sec_bhavdata_full" files: total traded qty, delivery qty, %)
# --------------------------------------------------------------------------------------
BHAV_URLS = [
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv",
    "https://archives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv",
]


def parse_bhav(text: str) -> dict:
    """symbol -> (total traded qty, delivery qty, delivery %). EQ and BE series only."""
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip().upper() for c in df.columns]
    for c in ("SYMBOL", "SERIES"):
        df[c] = df[c].astype(str).str.strip()
    df = df[df["SERIES"].isin(["EQ", "BE"])]
    ttl = pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce")
    dq = pd.to_numeric(df["DELIV_QTY"].astype(str).str.strip(), errors="coerce")
    dp = pd.to_numeric(df["DELIV_PER"].astype(str).str.strip(), errors="coerce")
    be = df["SERIES"].eq("BE")              # trade-to-trade: 100% delivery by rule
    dq = dq.where(~be, ttl)
    dp = dp.where(~be, 100.0)
    return {s: (t, q, p) for s, t, q, p in zip(df["SYMBOL"], ttl, dq, dp)}


def _bhav_one(date_iso: str, folder: Path):
    import requests
    d = dt.date.fromisoformat(date_iso)
    f = folder / f"sec_bhavdata_full_{d:%d%m%Y}.csv"
    if f.exists() and f.stat().st_size > 1000:
        return date_iso, f.read_text(encoding="utf-8", errors="ignore")
    for url in BHAV_URLS:
        try:
            r = requests.get(url.format(d=f"{d:%d%m%Y}"), headers=UA, timeout=30)
            if r.status_code == 200 and "DELIV" in r.text[:400].upper():
                f.write_text(r.text, encoding="utf-8")
                return date_iso, r.text
        except Exception:
            pass
    return date_iso, None


def load_delivery(dates: set, folder: Path, workers: int = 4) -> dict:
    """date -> {symbol: (ttl, deliv, deliv%)}. Files are cached, so later runs only fetch new days."""
    folder.mkdir(parents=True, exist_ok=True)
    out, failed, done = {}, 0, 0
    dates = sorted(dates)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_bhav_one, d, folder) for d in dates]):
            d, text = f.result()
            done += 1
            if text:
                try:
                    out[d] = parse_bhav(text)
                except Exception:
                    failed += 1
            else:
                failed += 1
            if done % 25 == 0 or done == len(dates):
                print(f"  delivery files: {done}/{len(dates)}", flush=True)
    if failed:
        print(f"  {failed} day(s) of NSE delivery data could not be downloaded", flush=True)
    return out


def _deliv_fields(rec, prefix):
    if not rec:
        return {}
    ttl, dq, dp = rec
    if not (np.isfinite(ttl) and ttl > 0):
        return {}
    o = {f"{prefix}_ttl": int(ttl)}
    if np.isfinite(dq):
        o.update({f"{prefix}_deliv": int(dq), f"{prefix}_intra": int(max(0, ttl - dq)),
                  f"{prefix}_deliv_pct": round(float(dp if np.isfinite(dp) else dq / ttl * 100), 1)})
        o[f"{prefix}_intra_pct"] = round(100 - o[f"{prefix}_deliv_pct"], 1)
    return o


def add_delivery(results: list, folder: Path, last_n: int = 20):
    """Attach delivery / intraday volume for the latest session, a 20-day average and
    the two spike days, for every stock in a first-wave stage."""
    setups = [r for r in results if STAGES.get(r.get("code"), ("", 0, False))[2] and "_df" in r]
    if not setups:
        return
    recent = list(setups[0]["_df"].index[-last_n:])
    for r in setups[:50]:                      # union of latest dates across a few stocks
        recent = sorted(set(recent) | set(r["_df"].index[-last_n:]))[-last_n:]
    recent = [d.date().isoformat() for d in recent]
    need = set(recent)
    for r in setups:
        for k in ("w1", "w3"):
            if r.get(f"{k}_spike_flag") and r.get(f"{k}_spike_date"):
                need.add(r[f"{k}_spike_date"])
    print(f"Fetching NSE delivery data for {len(need)} trading days (cached after the first run) ...")
    bhav = load_delivery(need, folder)
    if not bhav:
        print("  NSE delivery data unavailable - delivery columns left blank.")
        return
    last = max(bhav)
    for r in setups:
        s = r["symbol"]
        r.update(_deliv_fields(bhav[last].get(s), "last"))
        if "last_ttl" in r:
            r["last_deliv_date"] = last
        pcts = [bhav[d][s][2] for d in recent if d in bhav and s in bhav[d]
                and np.isfinite(bhav[d][s][2])]
        if pcts:
            r["avg20_deliv_pct"] = round(float(np.mean(pcts)), 1)
            if "last_deliv_pct" in r:
                r["deliv_vs_avg"] = round(r["last_deliv_pct"] - r["avg20_deliv_pct"], 1)
        for k, pre in (("w1_spike_date", "w1s"), ("w3_spike_date", "w3s")):
            d = r.get(k)
            if d and d in bhav:
                r.update(_deliv_fields(bhav[d].get(s), pre))


def finalize(r: dict) -> dict:
    c = r["price"]
    pts = r.get("pts") or []
    if pts:
        df = r["_df"]
        last_i, last_p = pts[-1]
        r["live_extreme"] = round(last_p, 2)
        r["live_extreme_date"] = df.index[last_i].date().isoformat()
    inv, t1 = r.get("invalidation"), r.get("t1")
    r["risk_pct"] = round((c - inv) / c * 100, 1) if inv and c > inv else np.nan
    r["to_t1_pct"] = round((t1 - c) / c * 100, 1) if t1 else np.nan
    r["to_trigger_pct"] = (round((r["trigger"] - c) / c * 100, 1) if r.get("trigger") else np.nan)
    rr = (t1 - c) / (c - inv) if (t1 and inv and c > inv and t1 > c) else np.nan
    r["reward_risk"] = round(rr, 2) if np.isfinite(rr) else np.nan
    r["score"] = score(r)
    r["notes"] = "; ".join(r.get("notes", [])) if isinstance(r.get("notes"), list) else r.get("notes", "")
    return r


def score(r: dict) -> int:
    base = {"W2B": 70, "W3A": 68, "W3N": 66, "W2": 55, "W3": 55, "W1": 40,
            "W4": 42, "W5A": 25, "W5": 20, "B": 15, "Z": 5, "X": 0, "N": 0, "E": 0}
    s = base.get(r.get("code"), 0)
    code = r.get("code")
    v21, v31 = r.get("vol_w2_w1", np.nan), r.get("vol_w3_w1", np.nan)
    if code in ("W2", "W2B", "W3A", "W3N") and np.isfinite(v21) and v21 < 0.8:
        s += 8                       # pullback on drying volume
    if code in ("W3", "W3A", "W3N") and np.isfinite(v31) and v31 > 1.2:
        s += 8                       # wave 3 on expanding volume
    if code == "W3" and r.get("w3_ext", 9) < 1.0:
        s += 8                       # early in wave 3
    rr = r.get("reward_risk", np.nan)
    if np.isfinite(rr):
        s += int(min(12, max(0, rr) * 3))
    risk = r.get("risk_pct", np.nan)
    if np.isfinite(risk) and risk > 30:
        s -= 8                       # stop too far away
    return int(max(0, min(100, s)))


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
COLS = [
    ("symbol", "Symbol"), ("name", "Company"), ("mcap_cr", "Mcap (cr)"), ("price", "Price"),
    ("stage", "Stage"), ("score", "Score"), ("detail", "What the count says"),
    ("invalidation", "Invalidation"), ("risk_pct", "Risk %"), ("trigger", "Trigger"),
    ("to_trigger_pct", "To trigger %"), ("t1", "Target 1"), ("t2", "Target 2"),
    ("to_t1_pct", "To T1 %"), ("reward_risk", "Reward/Risk"),
    ("w1_gain_pct", "W1 gain %"), ("w2_retrace_pct", "W2 retrace %"), ("w3_ext", "W3 / W1"),
    ("w4_retrace_pct", "W4 retrace %"), ("vol_w2_w1", "Vol W2/W1"), ("vol_w3_w1", "Vol W3/W1"),
    ("origin_date", "Major low date"), ("origin_price", "Major low"),
    ("prior_drop_pct", "Prior fall %"), ("bars_since_low", "Bars since low"),
    ("live_extreme", "Live swing extreme"), ("live_extreme_date", "Extreme date"),
    ("w1_spike", "W1-end vol (x avg)"), ("w1_spike_date", "W1-end spike date"),
    ("w1_spike_vol", "W1-end spike volume"), ("w1s_deliv", "W1-end delivery qty"),
    ("w1s_deliv_pct", "W1-end delivery %"), ("w1s_intra", "W1-end intraday qty"),
    ("w1s_intra_pct", "W1-end intraday %"),
    ("w3_spike", "W3-start vol (x avg)"), ("w3_spike_date", "W3-start spike date"),
    ("w3_spike_vol", "W3-start spike volume"), ("w3s_deliv", "W3-start delivery qty"),
    ("w3s_deliv_pct", "W3-start delivery %"), ("w3s_intra", "W3-start intraday qty"),
    ("w3s_intra_pct", "W3-start intraday %"), ("w3_breakout_date", "Close above W1 high on"),
    ("last_deliv_date", "Latest NSE session"), ("last_ttl", "Latest total qty"),
    ("last_deliv", "Latest delivery qty"), ("last_deliv_pct", "Latest delivery %"),
    ("last_intra", "Latest intraday qty"), ("last_intra_pct", "Latest intraday %"),
    ("avg20_deliv_pct", "20-day avg delivery %"), ("deliv_vs_avg", "Delivery % vs 20d avg"),
    ("zz_pct", "Swing size %"), ("notes", "Notes"), ("target_note", "Level notes"),
    ("date", "Data as of"),
]

SPIKE_SHEETS = {
    "W1-end volume spikes": ("w1", "w1s", "Unusual volume (>= {x}x the 50-day average) in the "
                             "last 6 sessions of wave 1 or the 2 after its top."),
    "W3-start volume spikes": ("w3", "w3s", "Unusual volume (>= {x}x the 50-day average) in the "
                               "8 sessions after the wave 2 low, or on the close above the W1 high."),
}


def spike_table(df: pd.DataFrame, key: str, pre: str) -> pd.DataFrame:
    flag = df.get(f"{key}_spike_flag")
    if flag is None:
        return df.iloc[0:0]
    t = df[flag.fillna(False).astype(bool)].copy()
    cols = ["symbol", "name", "mcap_cr", "price", "stage", "score",
            f"{key}_spike_date", f"{key}_spike", f"{key}_spike_vol",
            f"{pre}_deliv", f"{pre}_deliv_pct", f"{pre}_intra", f"{pre}_intra_pct"]
    if key == "w3":
        cols.append("w3_breakout_date")
    cols += ["last_deliv_date", "last_deliv_pct", "last_intra_pct", "avg20_deliv_pct",
             "invalidation", "trigger", "t1", "reward_risk"]
    t = t[[c for c in cols if c in t.columns]].copy()
    dp = pd.to_numeric(t.get(f"{pre}_deliv_pct"), errors="coerce")
    avg = pd.to_numeric(t.get("avg20_deliv_pct"), errors="coerce")
    backed = dp >= avg
    t.insert(9, "delivery_backed", np.where(dp.isna() | avg.isna(), "n/a",
                                            np.where(backed, "Yes", "No")))
    t["_b"] = backed.fillna(False).astype(int)
    return t.sort_values(["_b", f"{key}_spike"], ascending=[False, False]).drop(columns="_b")


def to_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {k: r.get(k, np.nan) for k, _ in COLS}
        row["w1_spike_flag"] = bool(r.get("w1_spike_flag", False))
        row["w3_spike_flag"] = bool(r.get("w3_spike_flag", False))
        row["_order"] = STAGES.get(r.get("code", "E"), ("", 99))[1]
        row["_setup"] = STAGES.get(r.get("code", "E"), ("", 99, False))[2]
        for k in ("invalidation", "trigger", "t1", "t2"):
            if isinstance(row[k], (int, float)) and np.isfinite(row[k]):
                row[k] = round(float(row[k]), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    return df.sort_values(["_setup", "score", "_order"], ascending=[False, False, True])


STAGE_FILL = {"W2B": "C6EFCE", "W3A": "C6EFCE", "W3N": "C6EFCE", "W2": "E2F0D9",
              "W3": "DDEBF7", "W1": "FFF2CC", "W4": "FCE4D6", "W5A": "F8CBAD", "W5": "F8CBAD"}


def write_excel(df: pd.DataFrame, path: Path, vol_spike: float = 2.0):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    label_to_code = {v[0]: k for k, v in STAGES.items()}
    hidden = ["_order", "_setup", "w1_spike_flag", "w3_spike_flag"]
    setups_full = df[df["_setup"]]
    setups = setups_full.drop(columns=hidden)
    allr = df.drop(columns=hidden)
    header = {k: v for k, v in COLS}
    header["delivery_backed"] = "Delivery-backed? (spike deliv % >= 20d avg)"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        setups.rename(columns=header).to_excel(xw, sheet_name="First-wave setups", index=False)
        for sheet, (key, pre, _) in SPIKE_SHEETS.items():
            spike_table(setups_full, key, pre).rename(columns=header).to_excel(
                xw, sheet_name=sheet, index=False)
        allr.rename(columns=header).to_excel(xw, sheet_name="All scanned", index=False)
        guide = pd.DataFrame({"Item": [
            "Major low", "First wave", "Swing size %", "Wave 2 - buy zone", "Wave 3 starting",
            "Wave 3 - (i)/(ii)", "Wave 3 confirmed", "Wave 4 / 5", "Invalidation", "Trigger",
            "Targets", "Vol W2/W1", "Vol W3/W1", "W1-end volume spikes", "W3-start volume spikes",
            "Vol (x avg)", "Delivery qty / %", "Intraday qty / %", "20-day avg delivery %",
            "Score", "Caveat"], "Meaning": [
            "Lowest low of the last ~2 years, which ended a fall of 30%+ (or a post-IPO low).",
            "Waves are counted only from that low, so every setup is the FIRST impulse off a bottom.",
            "ZigZag reversal size used to define a swing (scaled to the stock's volatility).",
            "Pullback of 50-78.6% of wave 1 - the classic wave 2 area.",
            "Wave 2 low is in and price is rising, but has not yet closed above the wave 1 high.",
            "Wave 3 started, pulled back as a higher low: the (ii) of 3, often before the strongest leg.",
            "Price is above the wave 1 high. W3 / W1 shows how far wave 3 has run versus wave 1.",
            "Later stages: less upside left in the first impulse.",
            "Level where the count is wrong (W2: start of W1; W3: W2 low; W4: W1 high).",
            "Level that confirms the next wave (usually the prior wave high).",
            "Fibonacci projections - see 'Level notes' column for which ones.",
            "Below 1 = wave 2 fell on lighter volume than wave 1 rose (healthy).",
            "Above 1 = wave 3 on heavier volume than wave 1 (healthy).",
            SPIKE_SHEETS["W1-end volume spikes"][2].format(x=vol_spike)
            + " High delivery here = buyers holding; high intraday = churn/distribution.",
            SPIKE_SHEETS["W3-start volume spikes"][2].format(x=vol_spike)
            + " High delivery here = genuine accumulation behind wave 3.",
            "That day's volume divided by the average of the previous 50 sessions.",
            "Shares actually taken into demat that day (NSE bhavcopy DELIV_QTY / DELIV_PER).",
            "Total traded minus delivery = shares bought and sold the same day.",
            "Average delivery % over the last 20 NSE sessions; compare the latest day against it.",
            "Ranks early setups with good structure, volume and reward/risk. Not a buy signal.",
            "Mechanical count from swing points. Confirm every chart by eye; combine with fundamentals."]})
        guide.to_excel(xw, sheet_name="How to read", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "C2" if ws.title != "How to read" else "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            widths = {"Company": 28, "Stage": 30, "What the count says": 55, "Notes": 30,
                      "Level notes": 45, "Item": 22, "Meaning": 110}
            for j, cell in enumerate(ws[1], start=1):
                if isinstance(cell.value, str) and ("qty" in cell.value or "volume" in cell.value):
                    for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                        row[0].number_format = "#,##0"
            for j, cell in enumerate(ws[1], start=1):
                ws.column_dimensions[get_column_letter(j)].width = widths.get(cell.value, 12)
            if ws.title == "How to read":
                continue
            stage_col = [c.value for c in ws[1]].index("Stage") + 1
            for row in ws.iter_rows(min_row=2):
                code = label_to_code.get(row[stage_col - 1].value)
                if code in STAGE_FILL:
                    row[stage_col - 1].fill = PatternFill("solid", fgColor=STAGE_FILL[code])
            ws.auto_filter.ref = ws.dimensions


def spark_svg(r: dict, w=320, h=120) -> str:
    df = r.get("_df")
    if df is None or not r.get("pts"):
        return ""
    o = r["_o"]
    s = max(0, o - 40)
    closes = df["Close"].values[s:]
    pts = [(i - s, p) for i, p in r["pts"]]
    lo = min(closes.min(), min(p for _, p in pts))
    hi = max(closes.max(), max(p for _, p in pts))
    pad = 18
    X = lambda i: pad + (w - 2 * pad) * i / max(1, len(closes) - 1)
    Y = lambda p: pad + (h - 2 * pad) * (hi - p) / max(1e-9, hi - lo)
    line = " ".join(f"{X(i):.1f},{Y(p):.1f}" for i, p in enumerate(closes))
    wave = " ".join(f"{X(i):.1f},{Y(p):.1f}" for i, p in pts)
    marks = []
    for k, ((i, p), lab) in enumerate(zip(pts, r["labels"])):
        is_low = (k == 0) or (k > 0 and p < pts[k - 1][1])
        ty = Y(p) + (12 if is_low else -6)
        marks.append(f'<circle cx="{X(i):.1f}" cy="{Y(p):.1f}" r="2.5" class="pv"/>'
                     f'<text x="{X(i):.1f}" y="{ty:.1f}" class="lb">{html.escape(lab)}</text>')
    lines = []
    for key, cls in (("invalidation", "inv"), ("trigger", "trg")):
        v = r.get(key)
        if v and lo <= v <= hi:
            lines.append(f'<line x1="{pad}" x2="{w - pad}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="{cls}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" class="sp" role="img" aria-label="price with wave count">'
            f'{"".join(lines)}<polyline points="{line}" class="px"/>'
            f'<polyline points="{wave}" class="wv"/>{"".join(marks)}</svg>')


def write_html(results: list[dict], path: Path, title: str):
    setups = [r for r in results if STAGES[r.get("code", "E")][2]]
    setups.sort(key=lambda r: (-r.get("score", 0), STAGES[r["code"]][1]))
    counts = {}
    for r in setups:
        counts[r["code"]] = counts.get(r["code"], 0) + 1
    chips = "".join(
        f'<button class="chip" data-c="{c}">{html.escape(STAGES[c][0])} <b>{n}</b></button>'
        for c, n in sorted(counts.items(), key=lambda kv: STAGES[kv[0]][1]))
    n_w1 = sum(1 for r in setups if r.get("w1_spike_flag"))
    n_w3 = sum(1 for r in setups if r.get("w3_spike_flag"))
    chips += (f'<button class="chip vs" data-c="v1">Volume spike at W1 end <b>{n_w1}</b></button>'
              f'<button class="chip vs" data-c="v3">Volume spike at W3 start <b>{n_w3}</b></button>')

    def f(v, pct=False):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:+.1f}%" if pct else f"{v:,.2f}"

    def qty(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        v = float(v)
        return (f"{v / 1e7:.2f} cr" if v >= 1e7 else f"{v / 1e5:.2f} L" if v >= 1e5
                else f"{v:,.0f}")

    def vol_block(r):
        rows = []
        for key, pre, lab in (("w1", "w1s", "W1 end"), ("w3", "w3s", "W3 start")):
            if r.get(f"{key}_spike") is None:
                continue
            hot = " hot" if r.get(f"{key}_spike_flag") else ""
            dl = (f"deliv {qty(r.get(f'{pre}_deliv'))} ({r[f'{pre}_deliv_pct']:.0f}%) &middot; "
                  f"intraday {qty(r.get(f'{pre}_intra'))} ({r[f'{pre}_intra_pct']:.0f}%)"
                  if r.get(f"{pre}_deliv_pct") is not None else
                  ("delivery n/a" if hot else "normal volume"))
            rows.append(f'<div class="vr{hot}"><b>{lab}</b> {r[f"{key}_spike"]:.1f}x avg on '
                        f'{r.get(f"{key}_spike_date", "")} &middot; vol {qty(r.get(f"{key}_spike_vol"))}'
                        f'<br><span>{dl}</span></div>')
        if r.get("last_deliv_pct") is not None:
            avg = r.get("avg20_deliv_pct")
            rows.append(f'<div class="vr"><b>Latest</b> {r.get("last_deliv_date", "")}: deliv '
                        f'{qty(r.get("last_deliv"))} ({r["last_deliv_pct"]:.0f}%) &middot; intraday '
                        f'{qty(r.get("last_intra"))} ({r["last_intra_pct"]:.0f}%)'
                        + (f'<br><span>20-day avg delivery {avg:.0f}%</span>' if avg is not None else "")
                        + '</div>')
        return f'<div class="vb">{"".join(rows)}</div>' if rows else ""

    cards = []
    for r in setups:
        mc = r.get("mcap_cr")
        mc_s = f"{mc:,.0f} cr" if isinstance(mc, (int, float)) and np.isfinite(mc) else ""
        tags = (" v1" if r.get("w1_spike_flag") else "") + (" v3" if r.get("w3_spike_flag") else "")
        cards.append(f"""
<article class="card" data-c="{r['code']}" data-v="{tags.strip()}">
  <header><div><a href="https://www.tradingview.com/chart/?symbol=NSE:{html.escape(r['symbol'])}"
     target="_blank" rel="noopener">{html.escape(r['symbol'])}</a>
     <span class="nm">{html.escape(str(r.get('name', '')))[:34]}</span></div>
     <span class="sc" title="score">{r.get('score', 0)}</span></header>
  <div class="st s-{r['code']}">{html.escape(r['stage'])}</div>
  <p class="dt">{html.escape(str(r.get('detail', '')))}</p>
  {spark_svg(r)}
  <dl>
    <div><dt>Price</dt><dd>{f(r['price'])}</dd></div><div><dt>Mcap</dt><dd>{mc_s}</dd></div>
    <div><dt>Invalidation</dt><dd>{f(r.get('invalidation'))} <i>{f(-r.get('risk_pct', np.nan), True)}</i></dd></div>
    <div><dt>Trigger</dt><dd>{f(r.get('trigger'))} <i>{f(r.get('to_trigger_pct', np.nan), True)}</i></dd></div>
    <div><dt>Target 1</dt><dd>{f(r.get('t1'))} <i>{f(r.get('to_t1_pct', np.nan), True)}</i></dd></div>
    <div><dt>Target 2</dt><dd>{f(r.get('t2'))}</dd></div>
    <div><dt>Vol W2/W1</dt><dd>{f(r.get('vol_w2_w1', np.nan))}</dd></div>
    <div><dt>Vol W3/W1</dt><dd>{f(r.get('vol_w3_w1', np.nan))}</dd></div>
  </dl>
  {vol_block(r)}
  <p class="ft">Low {f(r.get('origin_price'))} on {r.get('origin_date', '')}
  {'(IPO listing low)' if r.get('ipo_base') else f"after a {r.get('prior_drop_pct', 0):.0f}% fall"}
  &middot; swing {r.get('zz_pct', '')}%
  {(' &middot; ' + html.escape(r['notes'])) if r.get('notes') else ''}</p>
</article>""")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f6f2;--card:#fff;--tx:#1d1d1b;--mu:#6b6a65;--bd:#e3e1da;--ac:#1f4e78;
--g:#1d7a4f;--gb:#e1f3e8;--b:#185fa5;--bb:#e6f1fb;--a:#8a5a07;--ab:#fbf0d9;--r:#a32d2d;--rb:#fcebeb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191917;--card:#232321;--tx:#ecebe6;--mu:#a3a199;
--bd:#3a3936;--ac:#85b7eb;--g:#7fd1a5;--gb:#173b2a;--b:#85b7eb;--bb:#132c46;--a:#efc06a;--ab:#3d2b0c;--r:#f09595;--rb:#4a1b1b}}}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--tx)}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:var(--mu);margin:0 0 16px}}.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}}
.chip{{border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:999px;padding:5px 12px;cursor:pointer;font:inherit}}
.chip.on{{border-color:var(--ac);color:var(--ac)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px}}
.card header{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
.card a{{color:var(--ac);font-weight:600;text-decoration:none;font-size:16px}}.nm{{color:var(--mu);margin-left:6px;font-size:12px}}
.sc{{font-weight:600;border:1px solid var(--bd);border-radius:6px;padding:0 7px}}
.st{{display:inline-block;margin:8px 0 4px;padding:2px 9px;border-radius:6px;font-size:12px;font-weight:600}}
.s-W2B,.s-W3A,.s-W3N{{background:var(--gb);color:var(--g)}}.s-W2,.s-W3{{background:var(--bb);color:var(--b)}}
.s-W1,.s-W4{{background:var(--ab);color:var(--a)}}.s-W5,.s-W5A{{background:var(--rb);color:var(--r)}}
.dt{{margin:2px 0 6px;color:var(--mu);font-size:13px}}
.sp{{width:100%;height:auto;display:block;margin:4px 0}}.px{{fill:none;stroke:var(--mu);stroke-width:1;opacity:.55}}
.wv{{fill:none;stroke:var(--ac);stroke-width:1.6}}.pv{{fill:var(--ac)}}
.lb{{font-size:10px;fill:var(--tx);text-anchor:middle;font-weight:600}}
.inv{{stroke:var(--r);stroke-dasharray:3 3;stroke-width:.8}}.trg{{stroke:var(--g);stroke-dasharray:3 3;stroke-width:.8}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;margin:6px 0}}dl div{{display:flex;justify-content:space-between;gap:6px}}
dt{{color:var(--mu);font-size:12px}}dd{{margin:0;font-variant-numeric:tabular-nums;font-size:13px}}dd i{{color:var(--mu);font-style:normal;font-size:11px}}
.ft{{color:var(--mu);font-size:11px;margin:6px 0 0}}.note{{color:var(--mu);font-size:12px;margin-top:24px}}
.vb{{border-top:1px solid var(--bd);margin-top:6px;padding-top:6px;display:grid;gap:4px;font-size:12px}}
.vr span{{color:var(--mu)}}.vr.hot b{{color:var(--a)}}.vr.hot{{background:var(--ab);border-radius:6px;padding:2px 6px}}
.chip.vs{{border-style:dashed}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">{len(setups)} stocks in a first impulse off a major low &middot; sorted by score &middot;
red dashed = invalidation, green dashed = trigger &middot; click a symbol for the TradingView chart</p>
<div class="chips"><button class="chip on" data-c="*">All <b>{len(setups)}</b></button>{chips}</div>
<section class="grid">{''.join(cards) or '<p>No setups found.</p>'}</section>
<p class="note">Mechanical swing count, not investment advice. Elliott counts are subjective - confirm
each chart by eye and combine with fundamentals before acting.</p>
</main><script>
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{{
 document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===b));
 const c=b.dataset.c;document.querySelectorAll('.card').forEach(k=>k.style.display=(c==='*'||k.dataset.c===c||(k.dataset.v||'').split(' ').includes(c))?'':'none');}});
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Elliott Wave first-impulse scanner (NSE small/micro caps)")
    ap.add_argument("--symbols", nargs="*", help="scan only these NSE symbols (no mcap filter)")
    ap.add_argument("--universe", help="CSV with a Symbol/NSE Code column and optional Market Cap (cr)")
    ap.add_argument("--min-mcap", type=float, default=500, help="minimum market cap in crore (default 500)")
    ap.add_argument("--max-mcap", type=float, default=None, help="optional maximum market cap in crore")
    ap.add_argument("--exclude-top", type=int, default=250, help="drop the largest N by mcap (default 250)")
    ap.add_argument("--lookback", type=int, default=500, help="bars to search for the major low (default 500)")
    ap.add_argument("--min-prior-drop", type=float, default=30, help="prior fall %% that defines a bottom (default 30)")
    ap.add_argument("--refresh", action="store_true", help="re-download the universe and market caps")
    ap.add_argument("--vol-spike", type=float, default=5.0,
                    help="volume this many times the 50-day average counts as unusual (default 5)")
    ap.add_argument("--no-delivery", action="store_true", help="skip NSE delivery/intraday volumes")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder")
    a = ap.parse_args(argv)

    cfg = Config(min_mcap_cr=a.min_mcap, max_mcap_cr=a.max_mcap, exclude_top=a.exclude_top,
                 lookback_bars=a.lookback, min_prior_drop=a.min_prior_drop / 100,
                 vol_spike=a.vol_spike)
    uni = build_universe(a, cfg)
    if uni.empty:
        sys.exit("Universe is empty.")
    print(f"Downloading {cfg.years}y of daily prices for {len(uni)} stocks ...")
    prices = download_prices(uni["symbol"].tolist(), cfg.years)
    meta = uni.set_index("symbol")[["name", "mcap_cr"]].to_dict("index")

    results = []
    for sym, df in prices.items():
        try:
            r = analyse(sym, df, cfg)
        except Exception as e:
            r = {"symbol": sym, "code": "E", "stage": STAGES["E"][0], "detail": f"error: {e}"}
        r.update(meta.get(sym, {}))
        results.append(r)
    missing = sorted(set(uni["symbol"]) - set(prices))
    for sym in missing:
        results.append({"symbol": sym, **meta.get(sym, {}), "code": "E",
                        "stage": STAGES["E"][0], "detail": "no price data from Yahoo"})

    if not a.no_delivery:
        try:
            add_delivery(results, CACHE_DIR / "bhav")
        except Exception as e:
            print(f"  delivery data skipped ({e})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    frame = to_frame(results)
    xlsx = out / f"elliott_first_wave_{stamp}.xlsx"
    page = out / f"elliott_first_wave_{stamp}.html"
    write_excel(frame, xlsx, cfg.vol_spike)
    write_html(results, page, f"Elliott first-wave scan - {stamp}")
    n_set = int(frame["_setup"].sum()) if not frame.empty else 0
    n1 = int((frame["_setup"] & frame["w1_spike_flag"]).sum()) if n_set else 0
    n3 = int((frame["_setup"] & frame["w3_spike_flag"]).sum()) if n_set else 0
    print(f"\nDone. {n_set} first-wave setups out of {len(results)} stocks.")
    print(f"  Unusual volume at end of wave 1  : {n1} stocks")
    print(f"  Unusual volume at start of wave 3: {n3} stocks")
    print(f"  Excel : {xlsx}\n  Report: {page}")
    if n_set:
        cols = ["symbol", "stage", "score", "price", "invalidation", "trigger", "t1"]
        print(frame[frame["_setup"]][cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
