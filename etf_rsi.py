#!/usr/bin/env python3
"""
ETF RSI view for the EQUITY ESSENCE app (used by app_scan.py).

  All ETFs listed on NSE (NSE ETF list; a built-in list if NSE does not answer), keeping the liquid
  ones: average traded value of Rs 1 crore or more per day over the last 20 sessions.
  Money-market / liquid / overnight and bond ETFs are left out (RSI means little there).
  Where several ETFs track the same index, only the most traded one is shown (others named on the card).
  Each ETF gets the same RSI rule as the stock scan: monthly RSI > 60, weekly RSI > 60, daily RSI > 50.

Educational use only - not investment advice.
"""
from __future__ import annotations

import io
import re
import time

import pandas as pd

import rsi_scanner as rsi

MIN_TURNOVER_CR = 1.0
SKIP = re.compile(r"LIQUID|OVERNIGHT|MONEY ?MARKET|GILT|G-SEC|GSEC|\bSDL\b|BOND|BHARAT BOND|DEBT|1D RATE|"
                  r"T-BILL|TBILL|CRISIL|TREASURY|CONSTANT MATURITY|CORPORATE|PSU BOND", re.I)
SKIP_SYM = re.compile(r"LIQUID|BBETF|GILT|SDL|BOND|1DRATE|CASHIETF|TBILL|LIQ", re.I)
FALLBACK = """NIFTYBEES JUNIORBEES BANKBEES PSUBNKBEES ITBEES PHARMABEES AUTOBEES INFRABEES CPSEETF MOREALTY
FMCGIETF METALIETF OILIETF HEALTHIETF FINIETF CONSUMBEES GOLDBEES SILVERBEES HNGSNGBEES MON100 MAFANG
MASPTOP50 MIDCAPETF MID150BEES HDFCSML250 MOMOMENTUM MOM30IETF ALPHA ALPL30IETF LOWVOLIETF NV20IETF
QUAL30IETF MODEFENCE MAKEINDIA PVTBANIETF SETFNIF50 SETFNN50 SETFGOLD MONIFTY500 MOVALUE EVINDIA
DIVOPPBEES SHARIABEES NEXT50IETF MOSMALL250 BSE500IETF SENSEXIETF MIDSELIETF GOLDIETF SILVERIETF
HDFCGOLD HDFCSILVER ICICIB22 BANKIETF ITIETF PHARMAIETF AUTOIETF NIFTYIETF MIDCAPIETF SMALLCAP
MOMENTUM50 MONQ50 MOHEALTH MOCAPITAL MOGSEC GROWWEV TOP100CASE CONSUMIETF COMMOIETF MNC""".split()


def etf_list(es, log=print) -> tuple[dict, str]:
    """symbol -> underlying index name."""
    import requests
    s = requests.Session()
    s.headers.update({**es.UA, "Accept": "application/json, text/plain, */*", "Referer": "https://www.nseindia.com/market-data/exchange-traded-funds-etf"})
    try:
        s.get("https://www.nseindia.com", timeout=20)
        r = s.get("https://www.nseindia.com/api/etf", timeout=25)
        if r.status_code == 200 and r.text.strip().startswith("{"):
            rows = r.json().get("data", [])
            out = {x["symbol"].strip(): str(x.get("assets") or x.get("meta", {}).get("companyName") or "").strip()
                   for x in rows if x.get("symbol")}
            if len(out) > 50:
                return out, "NSE ETF list"
    except Exception as e:
        log(f"  NSE ETF api: {e}")
    for u in ("https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv",
              "https://archives.nseindia.com/content/equities/eq_etfseclist.csv"):
        try:
            r = s.get(u, timeout=25)
            if r.status_code == 200 and "Symbol" in r.text[:300]:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                und = next((c for c in df.columns if c.lower().startswith("underlying")), None)
                out = {str(a).strip(): (str(b).strip() if und else "") for a, b in
                       zip(df["Symbol"], df[und] if und else df["Symbol"])}
                if len(out) > 50:
                    return out, "NSE ETF list"
        except Exception as e:
            log(f"  NSE ETF csv: {e}")
    return {x: "" for x in FALLBACK}, "built-in list (NSE did not answer)"


def _key(underlying: str, sym: str) -> str:
    u = re.sub(r"\b(INDEX|TRI|ETF|TOTAL RETURN|FUND)\b", "", underlying.upper())
    u = re.sub(r"[^A-Z0-9]", "", u)
    return u or sym


def build(es, download, log=print) -> dict:
    etfs, src = etf_list(es, log)
    etfs = {k: v for k, v in etfs.items() if not SKIP.search(v or "") and not SKIP_SYM.search(k)}
    log(f"ETF RSI: {len(etfs)} equity / commodity ETFs from {src}")
    data = {}
    syms = sorted(etfs)
    for i in range(0, len(syms), 80):
        data.update({t[:-3]: d for t, d in download([x + ".NS" for x in syms[i:i + 80]], 7).items()})
        time.sleep(1)
    rows = []
    for s, d in data.items():
        d = d.dropna(subset=["Close"])
        tv = float((d["Close"] * d["Volume"]).iloc[-20:].mean()) / 1e7
        if not tv >= MIN_TURNOVER_CR:
            continue
        card = rsi.scan(d, show_all=True)
        if not card:
            continue
        px = float(d["Close"].iloc[-1])
        rows.append({"sym": s, "name": etfs.get(s) or s, "price": round(px, 2),
                     "chg": round((px / float(d["Close"].iloc[-2]) - 1) * 100, 2) if len(d) > 1 else None,
                     "tv": round(tv, 1), "key": _key(etfs.get(s, ""), s), **card})
    # one ETF per index - the most traded
    best = {}
    for r in sorted(rows, key=lambda r: -r["tv"]):
        if r["key"] in best:
            best[r["key"]]["also"].append(r["sym"])
        else:
            r["also"] = []
            best[r["key"]] = r
    out = []
    for r in best.values():
        r["facts"] = [["Avg traded value", f"₹{r['tv']:,.1f} cr a day (20 days)"]] + r["facts"]
        if r["also"]:
            r["facts"].append(["Same index", ", ".join(r["also"][:6]) + (f" +{len(r['also']) - 6} more" if len(r["also"]) > 6 else "")])
        r["facts"] = [[a, b] for a, b in r["facts"] if b]
        r.pop("key"); r.pop("also")
        out.append(r)
    out.sort(key=lambda r: (not r["pass"], r.pop("sort")))
    log(f"ETF RSI: {len(rows)} liquid, {len(out)} after one-per-index, {sum(r['pass'] for r in out)} pass")
    return {"rows": out, "source": src, "liquid": len(rows), "min_cr": MIN_TURNOVER_CR}
