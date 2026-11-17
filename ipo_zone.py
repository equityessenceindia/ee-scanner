#!/usr/bin/env python3
"""
IPO Zone for the EQUITY ESSENCE app: stocks listed on NSE in the last 5 years (market cap >= Rs 500 cr),
in four sections.  Used by app_scan.py.

deals  Listing-day buyers - NSE bulk / block deals on the listing day, BUY side only. A client that
       also sold on the same day is treated as intraday and dropped. Sorted by value bought.
inst   FII + DII holding - latest quarter from the Screener shareholding table, FII + DII >= 15%,
       with the change over the last quarter and since the first quarter after listing.
vol    Volume breakout - a day in the last 5 sessions with volume >= 3x its 50-day average,
       closing up and at a new 20-day closing high (listing days ignored).
ath    All-time high - a new all-time high (since listing) made in the last 5 sessions,
       listed at least 3 months ago.

Educational use only - not investment advice.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEALS_FILE = HERE / "cache" / "listing_day_deals.json"     # same format as the laptop ipo_flag_scanner cache
YEARS = 5
INST_MIN = 15.0
VOL_X = 3.0
DEALS_PER_RUN = 500


def _num(s):
    s = re.sub(r"<[^>]+>|&nbsp;|\+", "", s or "").replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# ------------------------------------------------------------------------- shareholding
def parse_shareholding(page: str) -> dict:
    """Quarterly shareholding table from a Screener company page ->
    {"q": [quarter labels], "Promoters": [...], "FIIs": [...], "DIIs": [...], "Public": [...], ...}"""
    i = page.find('id="quarterly-shp"')
    if i == -1:
        i = page.find('id="shareholding"')
        if i == -1:
            return {}
    sec = page[i:]
    sec = sec[:sec.find("</table>")]
    head = sec[sec.find("<thead"):sec.find("</thead>")]
    qs = [re.sub(r"<[^>]+>|\s+", " ", h).strip() for h in re.findall(r"<th[^>]*>(.*?)</th>", head, re.S)]
    qs = [q for q in qs if q]
    out = {"q": qs}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", sec[sec.find("<tbody"):], re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 2:
            continue
        label = re.sub(r"<[^>]+>|&nbsp;|\+", " ", cells[0])
        label = re.sub(r"\s+", " ", label).strip()
        for key in ("Promoters", "FIIs", "DIIs", "Government", "Public"):
            if label.lower().startswith(key.lower()):
                vals = [_num(c) for c in cells[1:]]
                out[key] = [None if not np.isfinite(v) else v for v in vals][-len(qs):] if qs else vals
    return out if ("FIIs" in out or "DIIs" in out) else {}


# ------------------------------------------------------------------------- listing-day deals
def fetch_deals(items: list[tuple[str, dt.date]], es) -> tuple[dict, str]:
    """NSE bulk/block deals on each listing day (cached for good - they never change).
    Stops early if the NSE API is blocked, so a bad day costs only a minute."""
    import requests
    cache = {}
    if DEALS_FILE.exists():
        try:
            cache = json.loads(DEALS_FILE.read_text())
        except Exception:
            cache = {}
    todo = [(s, d) for s, d in items if s not in cache or not cache[s].get("ok", True)][:DEALS_PER_RUN]
    if not todo:
        return cache, ""
    print(f"IPO Zone: fetching listing-day deals for {len(todo)} IPOs ...", flush=True)
    sess = requests.Session()
    sess.headers.update({**es.UA, "Accept": "application/json, text/plain, */*",
                         "Referer": "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"})
    try:
        sess.get("https://www.nseindia.com", timeout=20)
        sess.get("https://www.nseindia.com/report-detail/display-bulk-and-block-deals", timeout=20)
    except Exception:
        pass
    ok_n = fail_n = 0
    for i, (s, d) in enumerate(todo, 1):
        rec, ds = {"bulk": [], "block": [], "ok": True}, d.strftime("%d-%m-%Y")
        for kind in ("bulk", "block"):
            url = ("https://www.nseindia.com/api/historicalOR/bulk-block-short-deals?"
                   f"optionType={kind}_deals&symbol={s}&from={ds}&to={ds}")
            got = False
            for attempt in range(2):
                try:
                    r = sess.get(url, timeout=25)
                    if r.status_code == 200 and r.text.strip().startswith("{"):
                        rec[kind] = r.json().get("data", []) or []
                        got = True
                        break
                except Exception:
                    pass
                time.sleep(1.5)
            rec["ok"] &= got
            time.sleep(0.8)
        if rec["ok"]:
            cache[s] = rec
            ok_n += 1
        else:
            fail_n += 1
            if ok_n == 0 and fail_n >= 6:
                print("  NSE deals API not reachable - will try again next run", flush=True)
                break
        if i % 25 == 0:
            print(f"  deals {i}/{len(todo)}", flush=True)
            DEALS_FILE.write_text(json.dumps(cache))
    DEALS_FILE.parent.mkdir(exist_ok=True)
    DEALS_FILE.write_text(json.dumps(cache))
    note = "" if ok_n or not fail_n else "NSE did not answer today - listing-day deals will be filled in on a later run."
    return cache, note


def buy_only(rec: dict) -> tuple[list, int]:
    """Buy deals whose client did not also sell on the listing day."""
    rows = []
    for kind in ("bulk", "block"):
        for d in rec.get(kind, []) or []:
            g = lambda *ks: next((d[k] for k in d if any(x in k.upper() for x in ks)), None)
            try:
                qty, px = float(str(g("QTY") or 0).replace(",", "")), float(str(g("WATP", "PRICE") or 0).replace(",", ""))
            except ValueError:
                continue
            rows.append({"client": str(g("CLIENT") or "").strip(), "side": str(g("BUY_SELL", "BUY") or "").upper(),
                         "qty": qty, "px": px, "kind": kind})
    norm = lambda c: re.sub(r"[^A-Z0-9]", "", c.upper())
    sellers = {norm(r["client"]) for r in rows if r["side"].startswith("S")}
    buys = {}
    for r in rows:
        if r["side"].startswith("B") and norm(r["client"]) not in sellers:
            b = buys.setdefault(norm(r["client"]), {"client": r["client"], "qty": 0.0, "val": 0.0})
            b["qty"] += r["qty"]
            b["val"] += r["qty"] * r["px"]
    return sorted(buys.values(), key=lambda b: -b["val"]), len(sellers)


# ------------------------------------------------------------------------- sections
def _fmt_date(d):
    return d.strftime("%d %b %Y")


def _cr(v):
    return f"₹{v / 1e7:,.1f} cr" if v >= 1e6 else f"₹{v / 1e5:,.1f} lakh"


def build(prices: dict, listed: dict, comp: dict, es) -> tuple[dict, str]:
    """Returns ({section: {symbol: card}}, note). card = status, facts, levels, sort (app_scan format)."""
    today = max(d.index[-1] for d in prices.values()).date()
    ipos = {}
    for s, d in prices.items():
        d = d.dropna(subset=["Close"])
        if len(d) < 5:
            continue
        first = d.index[0].date()
        lst = min(listed[s], first) if s in listed else first
        if (today - lst).days <= YEARS * 365 and (today - first).days <= YEARS * 365 + 30:
            ipos[s] = (lst, d)
    print(f"IPO Zone: {len(ipos)} stocks listed in the last {YEARS} years", flush=True)
    out = {k: {} for k in ("deals", "inst", "vol", "ath")}
    base_fact = lambda lst, d: ["Listed", f"{_fmt_date(lst)} · {(d['Close'].iloc[-1] / d['Open'].iloc[0] - 1) * 100:+.0f}% from listing-day open"]

    # 1. listing-day buyers
    deals, note = fetch_deals(sorted(((s, l) for s, (l, _) in ipos.items()), key=lambda t: t[1], reverse=True), es)
    for s, (lst, d) in ipos.items():
        rec = deals.get(s)
        if not rec or not rec.get("ok", True):
            continue
        buys, n_sellers = buy_only(rec)
        if not buys:
            continue
        tot = sum(b["val"] for b in buys)
        qty = sum(b["qty"] for b in buys)
        avg = tot / qty if qty else np.nan
        px = float(d["Close"].iloc[-1])
        facts = [base_fact(lst, d)]
        facts += [[("Buyers" if i == 0 else ""), f'{b["client"]} - {_cr(b["val"])} ({b["qty"] / 1e5:,.2f} lakh sh @ ₹{b["val"] / b["qty"]:,.1f})']
                  for i, b in enumerate(buys[:5])]
        if len(buys) > 5:
            facts.append(["", f"+{len(buys) - 5} more buyers"])
        facts.append(["Vs buyers' avg price", f"{(px / avg - 1) * 100:+.1f}%" if np.isfinite(avg) else ""])
        facts.append(["Sold same day (dropped)", f"{n_sellers} client(s)" if n_sellers else "none"])
        out["deals"][s] = {"status": f"{_cr(tot)} bought on listing day by {len(buys)} buyer{'s' if len(buys) > 1 else ''}",
                           "facts": facts, "levels": {"Buyers' avg": round(avg, 2)} if np.isfinite(avg) else {},
                           "sort": -tot}

    # 2. FII + DII holding
    for s, (lst, d) in ipos.items():
        sh = (comp.get(s) or {}).get("shp") or {}
        f, di, q = sh.get("FIIs") or [], sh.get("DIIs") or [], sh.get("q") or []
        n = max(len(f), len(di))
        if not n:
            continue
        val = lambda arr, k: (arr[k] if -len(arr) <= k < len(arr) and arr[k] is not None else 0.0) if arr else 0.0
        tot = [val(f, k) + val(di, k) for k in range(-n, 0)]
        now = tot[-1]
        if now < INST_MIN:
            continue
        prev = tot[-2] if n >= 2 else None
        first_t = tot[0]
        qlab = q[-1] if q else "latest"
        trend = ("Rising" if prev is not None and now > prev + 0.2 else
                 "Falling" if prev is not None and now < prev - 0.2 else "Steady")
        facts = [base_fact(lst, d),
                 ["FII / DII", f"{val(f, -1):.1f}% / {val(di, -1):.1f}% ({qlab})"],
                 ["Last quarter", f"{now - prev:+.1f} pts" if prev is not None else ""],
                 ["Since first quarter", f"{now - first_t:+.1f} pts ({q[-n] if len(q) >= n else ''})" if n >= 3 else ""],
                 ["Promoters", f'{val(sh.get("Promoters") or [], -1):.1f}%' if sh.get("Promoters") else ""]]
        out["inst"][s] = {"status": f"FII + DII {now:.1f}% · {trend}", "facts": facts, "levels": {}, "sort": -now}

    # 3. volume breakout, 4. all-time high
    for s, (lst, d) in ipos.items():
        d = d.dropna(subset=["High", "Low", "Close"])
        if len(d) < 30:
            continue
        c, h, v = d["Close"], d["High"], d["Volume"].astype(float).replace(0, np.nan)
        px = float(c.iloc[-1])
        dv = v.iloc[10:]                                            # ignore listing days
        rv = dv / dv.shift(1).rolling(50, min_periods=15).mean()
        hi20 = c.shift(1).rolling(20, min_periods=10).max()
        up = c > c.shift(1)
        sig = rv[(rv >= VOL_X) & up.reindex(rv.index) & (c.reindex(rv.index) >= hi20.reindex(rv.index))].iloc[-5:]
        sig = sig[sig.index >= d.index[-5]]
        if len(sig):
            day = sig.idxmax()
            k = d.index.get_loc(day)
            chg = (c.iloc[k] / c.iloc[k - 1] - 1) * 100
            out["vol"][s] = {
                "status": f"Volume {sig.max():.1f}x average on {_fmt_date(day.date())}",
                "facts": [base_fact(lst, d),
                          ["That day", f"{chg:+.1f}%, close at a new 20-day high"],
                          ["Vs all-time high", f"{(px / h.max() - 1) * 100:+.1f}%"],
                          ["Other heavy days (5 sessions)", ", ".join(f"{_fmt_date(x.date())} {y:.1f}x" for x, y in sig.items() if x != day)]],
                "levels": {"Breakout day low": round(float(d["Low"].iloc[k]), 2), "20-day high before": round(float(hi20.iloc[k]), 2)},
                "sort": -sig.max()}
        if (today - lst).days >= 90 and len(d) > 60:
            prior = h.iloc[:-5].max()
            if h.iloc[-5:].max() > prior:
                last_ath = h.iloc[:-5].idxmax().date()
                gap = (d.index[-6].date() - last_ath).days / 30.44
                out["ath"][s] = {
                    "status": "New all-time high" + (f" - first in {gap:.0f} months" if gap >= 3 else ""),
                    "facts": [base_fact(lst, d),
                              ["All-time high", f"₹{h.iloc[-5:].max():,.2f} ({_fmt_date(h.iloc[-5:].idxmax().date())})"],
                              ["Previous high", f"₹{prior:,.2f} ({_fmt_date(last_ath)})"],
                              ["Volume vs 50-day avg", f"{float(rv.iloc[-5:].max()):.1f}x (best of last 5 days)" if len(rv) and np.isfinite(rv.iloc[-5:].max()) else ""]],
                    "levels": {"Previous ATH": round(float(prior), 2)},
                    "sort": -gap}
    return out, note
