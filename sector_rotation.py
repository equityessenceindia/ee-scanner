#!/usr/bin/env python3
"""
EQUITY ESSENCE scanner app - Sector Rotation tab.

Two independent views, both built every evening by app_scan.py:

1. Money flow (where money is going)
   Every NSE stock in the app universe that has an NSE sector (Nifty Total Market + Microcap 250 lists)
   is grouped by sector.  For each day:   money flow = turnover x close-location value
       turnover = close x volume (Rs)
       close-location value = ((close - low) - (high - close)) / (high - low)   (+1 close at high, -1 at low)
   This is the Chaikin / accumulation-distribution idea summed over all stocks of the sector.
   Shown per sector: net flow in Rs crore for 1 week / 1 month / 3 months, flow strength
   (net flow / turnover, -100% to +100%), change in the sector's share of market turnover
   (money rotating in or out), breadth (% stocks above 50-DMA), and 12 weekly flow bars.
   It is an estimate from price and volume - not actual FII / DII / MF flow data.

2. Sector ETF charts (long-term trend)
   Weekly and monthly charts of the main sector ETFs (Nippon BeES and others) are read the
   Weinstein / relative-strength way: 30-week average and its slope, 10-month average, Mansfield
   relative strength against Nifty 50, higher highs / higher lows, distance from the 52-week high.
   Verdict: Strong uptrend (likely to continue) / Reviving (new uptrend starting) / Pullback in
   uptrend / Uptrend fading / Basing / Downtrend, plus a long-term rotation quadrant
   (Leading / Weakening / Lagging / Improving) for a rotation map.
   ETFs with short history are read on their Nifty sector index instead (from Yahoo Finance).

Educational use only - not investment advice.
"""
from __future__ import annotations

import io
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SECTOR_FILE = CACHE / "sectors.csv"
SECTOR_TTL_DAYS = 7
LISTS = ("niftytotalmarket", "niftymicrocap250", "nifty500")
LIST_URLS = ("https://www.niftyindices.com/IndexConstituent/ind_{n}_list.csv",
             "https://nsearchives.nseindia.com/content/indices/ind_{n}_list.csv",
             "https://archives.nseindia.com/content/indices/ind_{n}_list.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/csv,text/plain,*/*", "Referer": "https://www.niftyindices.com/"}

# ETF symbol, sector name, Yahoo symbol of the Nifty index it tracks (used when the ETF is too new)
ETFS = [
    ("BANKBEES", "Bank", "^NSEBANK"),
    ("PSUBNKBEES", "PSU Bank", "^CNXPSUBANK"),
    ("FINIETF", "Financial Services ex-Bank", "NIFTY_FIN_SERVICE.NS"),
    ("ITBEES", "IT", "^CNXIT"),
    ("PHARMABEES", "Pharma", "^CNXPHARMA"),
    ("HEALTHIETF", "Healthcare", None),
    ("AUTOBEES", "Auto", "^CNXAUTO"),
    ("FMCGIETF", "FMCG", "^CNXFMCG"),
    ("CONSUMBEES", "Consumption", "^CNXCONSUM"),
    ("METALIETF", "Metal", "^CNXMETAL"),
    ("OILIETF", "Oil & Gas", "^CNXENERGY"),
    ("INFRABEES", "Infrastructure", "^CNXINFRA"),
    ("CPSEETF", "CPSE (PSU companies)", "^CNXPSE"),
    ("MOREALTY", "Realty", "^CNXREALTY"),
]
BENCH = ("NIFTYBEES", "^NSEI")
MIN_WEEKS_ETF = 156           # under 3 years of ETF history -> read the index chart instead
CHART_WEEKS = 156             # weeks of chart sent to the app
FLOW_GROUPS = ("All", "Large", "Mid", "Small", "Micro")


def log(*a):
    print(*a, flush=True)


def r2(x, d=2):
    try:
        x = float(x)
    except Exception:
        return None
    return round(x, d) if math.isfinite(x) else None


# ------------------------------------------------------------------------- stock -> sector
def _get(url: str) -> str | None:
    import requests
    for attempt in range(2):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and "Symbol" in r.text[:500]:
                return r.text
        except Exception:
            pass
        time.sleep(2)
    return None


def sector_map() -> dict:
    """symbol -> NSE sector ("Industry" column of the NSE index lists), cached weekly."""
    old = {}
    if SECTOR_FILE.exists():
        try:
            df = pd.read_csv(SECTOR_FILE)
            old = dict(zip(df["symbol"], df["sector"]))
        except Exception:
            old = {}
        if old and time.time() - SECTOR_FILE.stat().st_mtime < SECTOR_TTL_DAYS * 86400:
            return old
    new = {}
    for n in LISTS:
        for u in LIST_URLS:
            txt = _get(u.format(n=n))
            if not txt:
                continue
            try:
                df = pd.read_csv(io.StringIO(txt))
                df.columns = [c.strip() for c in df.columns]
                for s, ind in zip(df["Symbol"], df["Industry"]):
                    if isinstance(s, str) and isinstance(ind, str):
                        new.setdefault(s.strip().upper(), ind.strip())
                log(f"  sector list {n}: {len(df)} stocks")
                break
            except Exception:
                continue
    if not new:
        log("  NSE sector lists not reachable - using the cached sector file" if old else
            "  NSE sector lists not reachable and no cached file - money flow skipped")
        return old
    merged = {**old, **new}
    CACHE.mkdir(exist_ok=True)
    pd.DataFrame(sorted(merged.items()), columns=["symbol", "sector"]).to_csv(SECTOR_FILE, index=False)
    return merged


# ------------------------------------------------------------------------- 1. money flow
def _frames(prices: dict, syms: list[str], days: int):
    def mat(col):
        return pd.DataFrame({s: prices[s][col] for s in syms}).sort_index()
    C = mat("Close")
    C = C[C.index.dayofweek < 5].iloc[-days:]
    H, L, V = (mat(c).reindex(C.index) for c in ("High", "Low", "Volume"))
    return C, H, L, V


def money_flow(prices: dict, caps: dict, sectors: dict) -> dict:
    syms = [s for s in prices if s in sectors and s in caps]
    if len(syms) < 50:
        return {}
    C, H, L, V = _frames(prices, syms, 260)
    T = (C * V) / 1e7                                         # turnover, Rs crore
    med = T.rolling(60, min_periods=20).median()
    T = T.where(~(T > 15 * med), 15 * med)                    # clip bad prints / block deals
    rng = (H - L).where(H > L)
    clv = (((C - L) - (H - C)) / rng).fillna(0).clip(-1, 1)
    MF = (clv * T).fillna(0)
    T = T.fillna(0)
    above50 = C > C.rolling(50, min_periods=40).mean()
    have50 = C.rolling(50, min_periods=40).mean().notna()
    ret1m = C.iloc[-1] / C.iloc[-22] - 1
    wk = MF.index.to_period("W-FRI")
    sec = pd.Series({s: sectors[s] for s in syms})
    cap = pd.Series({s: caps[s] for s in syms})
    weeks = sorted(set(wk))[-12:]

    out = {}
    for g in FLOW_GROUPS:
        members = sec.index if g == "All" else cap.index[cap == g]
        if len(members) < 30:
            continue
        tot_now = T[members].iloc[-21:].values.sum()
        mkt_mf, mkt_t = MF[members].sum(axis=1), T[members].sum(axis=1)
        mk = {k: (mkt_mf.iloc[-n:].sum() / mkt_t.iloc[-n:].sum() if mkt_t.iloc[-n:].sum() else 0)
              for k, n in (("1w", 5), ("1m", 21), ("3m", 63))}
        tot_before = T[members].iloc[-84:-21].values.sum()
        rows = []
        for name, idx in sec[members].groupby(sec[members]).groups.items():
            idx = list(idx)
            if len(idx) < 3:
                continue
            mf, tv = MF[idx].sum(axis=1), T[idx].sum(axis=1)
            n1w, n1m, n3m = mf.iloc[-5:].sum(), mf.iloc[-21:].sum(), mf.iloc[-63:].sum()
            t1w, t1m, t3m = tv.iloc[-5:].sum(), tv.iloc[-21:].sum(), tv.iloc[-63:].sum()
            if t3m <= 0:
                continue
            s1w, s1m, s3m = (n1w / t1w if t1w else 0), (n1m / t1m if t1m else 0), n3m / t3m
            share_now = t1m / tot_now if tot_now else 0
            share_before = tv.iloc[-84:-21].sum() / tot_before if tot_before else 0
            share_chg = share_now / share_before - 1 if share_before > 0 else 0
            b = above50[idx].iloc[-1][have50[idx].iloc[-1]]
            weekly = mf.groupby(wk).sum()
            rel = {"1w": s1w - mk["1w"], "1m": s1m - mk["1m"], "3m": s3m - mk["3m"]}
            label, note = flow_label(rel["1w"], rel["1m"], rel["3m"], share_chg)
            rows.append({
                "sector": name, "n": len(idx), "label": label, "note": note,
                "net": {"1w": r2(n1w, 0), "1m": r2(n1m, 0), "3m": r2(n3m, 0)},
                "str": {"1w": r2(s1w * 100, 1), "1m": r2(s1m * 100, 1), "3m": r2(s3m * 100, 1)},
                "rel": {k: r2(v * 100, 1) for k, v in rel.items()},
                "share": r2(share_now * 100, 1), "share_chg": r2(share_chg * 100, 0),
                "turnover_1m": r2(t1m, 0),
                "breadth": r2(b.mean() * 100, 0) if len(b) else None,
                "ret1m": r2(ret1m[idx].median() * 100, 1),
                "weekly": [r2(weekly.get(w, 0), 0) for w in weeks],
                "_score": rel["1m"] * 100 + 0.5 * rel["1w"] * 100 + 0.1 * (share_chg * 100),
            })
        rows.sort(key=lambda r: -r.pop("_score"))
        out[g] = {"sectors": rows, "stocks": len(members), "market": {k: r2(v * 100, 1) for k, v in mk.items()}}
    out["weeks"] = [w.end_time.date().isoformat() for w in weeks]
    return out


def flow_label(r1w, r1m, r3m, share_chg):
    """Labels use flow strength RELATIVE to the whole market (all sectors together), so a broad
    rally or sell-off does not mark every sector as inflow / outflow - only rotation shows."""
    if r1m >= 0.08 and share_chg >= 0.05:
        lab = "Strong inflow"
    elif r1m >= 0.04:
        lab = "Inflow"
    elif r1m <= -0.08 and share_chg <= -0.05:
        lab = "Strong outflow"
    elif r1m <= -0.04:
        lab = "Outflow"
    else:
        lab = "In line with market"
    if r1w >= 0.05 and r3m < -0.02:
        note = "Buying picked up this week after 3 months of lagging the market"
    elif r1w <= -0.05 and r3m > 0.02:
        note = "Selling picked up this week after 3 months of leading the market"
    elif r1m >= 0.04 and r3m > 0 and share_chg > 0.1:
        note = "Steady inflow and a rising share of market turnover"
    elif r1m <= -0.04 and r3m < 0 and share_chg < -0.1:
        note = "Steady outflow and a falling share of market turnover"
    elif share_chg >= 0.25:
        note = "Trading activity rising sharply"
    elif share_chg <= -0.25:
        note = "Trading activity drying up"
    else:
        note = ""
    return lab, note


def sector_quadrants(prices: dict, sectors: dict) -> dict:
    """Long-term quadrant of each NSE sector from an equal-weight index of its stocks versus
    an equal-weight index of all stocks (feeds the stage 2 -> 3 'sector rotating out' signal)."""
    syms = [s for s in prices if s in sectors]
    if len(syms) < 50:
        return {}
    C = pd.DataFrame({s: prices[s]["Close"] for s in syms}).sort_index()
    C = C[C.index.dayofweek < 5].iloc[-600:].ffill(limit=3)
    R = C.pct_change().clip(-0.2, 0.2)
    mkt = (1 + R.mean(axis=1).fillna(0)).cumprod()
    out = {}
    sec = pd.Series({s: sectors[s] for s in syms})
    for name, idx in sec.groupby(sec).groups.items():
        if len(idx) < 3:
            continue
        ix = (1 + R[list(idx)].mean(axis=1).fillna(0)).cumprod()
        rs = (ix / mkt).resample("W-FRI").last().dropna()
        if len(rs) < 60:
            continue
        m = (rs / rs.rolling(52).mean() - 1) * 100
        x, y = m.iloc[-1], m.iloc[-1] - m.iloc[-9]
        out[name] = quadrant(x, y)
    return out


# ------------------------------------------------------------------------- 2. sector ETFs
def download(tickers: list[str], years: int = 12) -> dict:
    import yfinance as yf
    data = {}
    for attempt in range(3):
        try:
            df = yf.download(tickers, period=f"{years}y", interval="1d", auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
            break
        except Exception as e:
            log(f"  retry ({e})")
            time.sleep(3 * (attempt + 1))
    else:
        return data
    if df is None or df.empty:
        return data
    for t in tickers:
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            sub = sub[sub["Close"] > 0]
            if len(sub) >= 100:
                data[t] = sub
        except Exception:
            continue
    return data


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = d.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Close"])


def quadrant(x, y) -> str:
    if x >= 0:
        return "Leading" if y >= 0 else "Weakening"
    return "Improving" if y >= 0 else "Lagging"


def analyse_chart(d: pd.DataFrame, bench: pd.DataFrame) -> dict | None:
    """Long-term trend read of one weekly chart against the Nifty 50 benchmark."""
    w = weekly(d)
    if len(w) < 60:
        return None
    c, lo = w["Close"], w["Low"]
    ma30, ma10 = c.rolling(30).mean(), c.rolling(10).mean()
    slope = ma30.iloc[-1] / ma30.iloc[-6] - 1
    slope13 = ma30.iloc[-13] / ma30.iloc[-18] - 1 if len(w) >= 48 else np.nan
    mo = d["Close"].resample("ME").last().dropna()
    m10 = mo.rolling(10).mean()
    mo_ok = len(mo) >= 12 and mo.iloc[-1] > m10.iloc[-1] and m10.iloc[-1] >= m10.iloc[-3]
    b = weekly(bench)["Close"].reindex(c.index).ffill()
    rs = (c / b).dropna()
    mans = (rs / rs.rolling(52, min_periods=40).mean() - 1) * 100
    mx = mans.iloc[-1]
    my = mx - mans.iloc[-9] if len(mans) > 9 else np.nan
    px = float(c.iloc[-1])
    hi52, lo52 = c.iloc[-52:].max(), lo.iloc[-52:].min()
    from_hi = (px / hi52 - 1) * 100
    from_hi_13 = (c.iloc[-13] / c.iloc[-65:-13].max() - 1) * 100 if len(c) >= 65 else 0
    above = (c > ma30).values
    streak = 0
    for v in above[::-1]:
        if not v:
            break
        streak += 1
    hh = c.iloc[-13:].max() > c.iloc[-26:-13].max()
    hl = lo.iloc[-13:].min() > lo.iloc[-26:-13].min()
    brk26 = px >= c.iloc[-27:-1].max()
    up_ma, a30, a10 = slope > 0.004, px > ma30.iloc[-1], px > ma10.iloc[-1]
    flat = abs(slope) <= 0.004

    # Weinstein stage
    if a30 and up_ma:
        stage = "Stage 2 - advancing"
    elif not a30 and slope < -0.004:
        stage = "Stage 4 - declining"
    elif flat:
        prior_up = above[-40:-10].mean() > 0.6 if len(above) >= 40 else False
        stage = "Stage 3 - topping" if prior_up and from_hi > -15 else "Stage 1 - basing"
    elif a30:
        stage = "Stage 1 to 2 - turning up"
    else:
        stage = "Stage 2 - pullback"

    was_weak = (np.isfinite(slope13) and slope13 < 0) or from_hi_13 <= -15 or streak <= 16
    rs_up = np.isfinite(my) and my > 0
    if a30 and up_ma and streak >= 20 and mo_ok and mx > 0 and from_hi > -12 and (my > -3 or not np.isfinite(my)):
        verdict, why = "Strong uptrend", "Long-term trend and relative strength both up - likely to continue"
    elif a30 and was_weak and (ma10.iloc[-1] > ma30.iloc[-1] or brk26) and rs_up and (hh or hl or brk26):
        verdict, why = "Reviving", "Turning up from a weak phase with improving relative strength"
    elif a30 and up_ma and (not rs_up or not a10) and mx > 0:
        verdict, why = "Uptrend fading", "Still above the 30-week average but momentum versus Nifty is slipping"
    elif a30 and up_ma:
        verdict, why = "Uptrend", "Above a rising 30-week average"
    elif not a30 and up_ma and mo_ok:
        verdict, why = "Pullback in uptrend", "Dipped below the 30-week average while the long-term trend is still up"
    elif stage.startswith("Stage 4"):
        verdict, why = "Downtrend", "Below a falling 30-week average - no sign of revival yet"
    elif stage.startswith("Stage 3"):
        verdict, why = "Topping", "Trend flattening after an advance - uptrend may be ending"
    else:
        verdict, why = "Basing", "Moving sideways - watch for a break above the range with relative strength"

    score = (20 * a30 + 15 * up_ma + 15 * bool(mo_ok) + 15 * (mx > 0) + 10 * rs_up + 5 * hh + 5 * hl
             + 10 * (from_hi > -10) + 5 * (ma10.iloc[-1] > ma30.iloc[-1]))
    ret = lambda n: r2((px / c.iloc[-n - 1] - 1) * 100, 1) if len(c) > n else None
    tail = [[r2(mans.iloc[-k], 1), r2(mans.iloc[-k] - mans.iloc[-k - 8], 1)]
            for k in range(6, 0, -1) if len(mans) > k + 8]
    return {
        "verdict": verdict, "why": why, "stage": stage, "score": int(score),
        "quad": quadrant(mx, my if np.isfinite(my) else 0),
        "rs": r2(mx, 1), "rs_mom": r2(my, 1), "from_hi": r2(from_hi, 1),
        "from_lo": r2((px / lo52 - 1) * 100, 1), "weeks_above": streak,
        "ma30_slope": r2(slope * 100, 2), "monthly_ok": bool(mo_ok), "hh": bool(hh), "hl": bool(hl),
        "ret": {"3m": ret(13), "6m": ret(26), "1y": ret(52), "3y": ret(156)},
        "tail": tail,
        "chart": {"d": [i.date().isoformat() for i in c.index[-CHART_WEEKS:]][::4],
                  "c": [r2(v, 2) for v in c.iloc[-CHART_WEEKS:]],
                  "ma": [r2(v, 2) for v in ma30.iloc[-CHART_WEEKS:]]},
    }


def etf_view() -> tuple[list, pd.DataFrame | None]:
    etf_t = [e[0] + ".NS" for e in ETFS] + [BENCH[0] + ".NS"]
    idx_t = [e[2] for e in ETFS if e[2]] + [BENCH[1]]
    data = download(etf_t)
    idx = download(idx_t)
    log(f"Sector ETFs: {len(data)}/{len(etf_t)} ETFs, {len(idx)}/{len(idx_t)} indices downloaded")
    bench = data.get(BENCH[0] + ".NS")
    if bench is None or len(bench) < 2000:
        bench = idx.get(BENCH[1], bench)
    if bench is None:
        return [], None
    rows = []
    for sym, name, ix in ETFS:
        d = data.get(sym + ".NS")
        src = d
        note = ""
        if (d is None or len(weekly(d)) < MIN_WEEKS_ETF) and ix in idx:
            src, note = idx[ix], f"Chart read on the {ix.strip('^').replace('.NS', '')} index (ETF history too short)"
        if src is None:
            continue
        try:
            r = analyse_chart(src, bench)
        except Exception as e:
            log(f"  {sym}: {e}")
            continue
        if not r:
            continue
        px = d["Close"] if d is not None else None
        r.update(sym=sym, name=name, note=note,
                 price=r2(px.iloc[-1]) if px is not None else None,
                 chg=r2((px.iloc[-1] / px.iloc[-2] - 1) * 100) if px is not None and len(px) > 1 else None)
        rows.append(r)
    order = {"Strong uptrend": 0, "Reviving": 1, "Uptrend": 2, "Pullback in uptrend": 3,
             "Uptrend fading": 4, "Basing": 5, "Topping": 6, "Downtrend": 7}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), -r["score"]))
    return rows, bench


VERDICT_ORDER = {"Strong uptrend": 0, "Reviving": 1, "Uptrend": 2, "Pullback in uptrend": 3,
                 "Uptrend fading": 4, "Basing": 5, "Topping": 6, "Downtrend": 7}
THEMES_FILE = HERE / "themes.csv"


def theme_view(prices: dict, bench: pd.DataFrame | None) -> list:
    """Sub-sector baskets from themes.csv: an equal-weight combined chart of the member stocks
    (each day = average % move of the members trading that day, rebased to 100), read with the same
    long-term rules as the sector ETFs."""
    if bench is None or not THEMES_FILE.exists():
        return []
    t = pd.read_csv(THEMES_FILE)
    rows = []
    for _, th in t.iterrows():
        syms = [x for x in str(th["symbols"]).split() if x in prices]
        if len(syms) < 2:
            continue
        C = pd.DataFrame({x: prices[x]["Close"] for x in syms}).sort_index()
        C = C[C.index.dayofweek < 5].iloc[-1600:]
        R = C.pct_change(fill_method=None).clip(-0.25, 0.25)
        cnt = R.notna().sum(axis=1)
        ix = (1 + R.mean(axis=1).where(cnt >= 1, 0).fillna(0)).cumprod() * 100
        ix = ix[ix.index >= C.dropna(how="all").index[0]]
        d = pd.DataFrame({"Open": ix, "High": ix, "Low": ix, "Close": ix, "Volume": 0.0})
        try:
            r = analyse_chart(d, bench)
        except Exception as e:
            log(f"  theme {th['subsector']}: {e}")
            continue
        if not r:
            continue
        mem = []
        for x in syms:
            c = prices[x]["Close"].dropna()
            ma = c.rolling(150, min_periods=60).mean()
            mem.append({"sym": x,
                        "r3m": r2((c.iloc[-1] / c.iloc[-64] - 1) * 100, 1) if len(c) > 64 else None,
                        "r1y": r2((c.iloc[-1] / c.iloc[-253] - 1) * 100, 1) if len(c) > 253 else None,
                        "up": bool(c.iloc[-1] > ma.iloc[-1]) if np.isfinite(ma.iloc[-1]) else None})
        mem.sort(key=lambda m: -(m["r3m"] if m["r3m"] is not None else -999))
        known = [m["up"] for m in mem if m["up"] is not None]
        r.update(name=th["subsector"], sector=th["sector"], members=mem, n=len(syms),
                 breadth=r2(100 * sum(known) / len(known), 0) if known else None,
                 r1m=r2((ix.iloc[-1] / ix.iloc[-22] - 1) * 100, 1) if len(ix) > 22 else None)
        r.pop("tail", None)
        rows.append(r)
    rows.sort(key=lambda r: (VERDICT_ORDER.get(r["verdict"], 9), -r["score"]))
    log(f"Sub-sector baskets: {len(rows)}")
    return rows


# ------------------------------------------------------------------------- entry point
def run(prices: dict, caps: dict) -> tuple[dict, dict]:
    """Returns (data for the app, symbol -> (sector, long-term quadrant) for stage 2 maturity)."""
    out, stock_view = {}, {}
    try:
        sectors = sector_map()
        log(f"Sectors known for {sum(1 for s in prices if s in sectors)}/{len(prices)} stocks")
        out["flow"] = money_flow(prices, caps, sectors)
        quads = sector_quadrants(prices, sectors)
        for g in FLOW_GROUPS:
            for row in out["flow"].get(g, {}).get("sectors", []):
                row["quad"] = quads.get(row["sector"])
        stock_view = {s: (sec, quads.get(sec)) for s, sec in sectors.items() if quads.get(sec)}
    except Exception as e:
        log(f"Money flow failed: {e}")
    bench = None
    try:
        out["etfs"], bench = etf_view()
    except Exception as e:
        log(f"Sector ETFs failed: {e}")
    try:
        out["themes"] = theme_view(prices, bench)
    except Exception as e:
        log(f"Sub-sector baskets failed: {e}")
    return out, stock_view
