#!/usr/bin/env python3
"""
EQUITY ESSENCE scanner app - end-of-day job.

Runs the five scanners on Large (Nifty 100), Mid (rest of Nifty 500), Small (Smallcap 250) and
Micro caps (every other NSE stock with market cap >= Rs 500 cr), keeps only companies that hold
concalls regularly, builds the Concalls feed, and writes docs/data/latest.json (+ latest.js)
for the phone app.  Runs every weekday evening on GitHub Actions (see .github/workflows/eod.yml)
and can also be run by hand:   python app_scan.py            (full run)
                                python app_scan.py --quick    (first 60 stocks, for testing)

Scanners (the same code as the laptop scanners in this folder):
  flag      Flag & pole, weekly/monthly, unusual volume at the pole top or in the flag
  elliott   Elliott wave 2 (first impulse off a major low): wave 2 in progress / buy zone /
            wave 3 starting, with an unusual-volume print at the end of wave 1 or start of wave 3
  base      Long consolidation (26+ weeks) breakout, with unusual volume near the breakout
  accum     Long weekly / monthly base breakout with an unusual-volume print inside the base
  ipo       Recent IPO (last 3 years) coming out of its post-listing base on good volume
  s2early   Minervini stage 2 - just started / resumed, or early stage 2 by the maturity model
  s2late    Stage 2 maturing into stage 3 (late stage 2 / topping) with volume drying up
Every list needs "Regular" concalls (Screener.in: 4+ calls in 24 months, latest within 7 months;
scaled down for recent listings). Educational use only - not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
import base_breakout_scanner as bb
import stage2_scanner as s2
import stage2_maturity as sm
import concall_points as cp
import sector_rotation as sr
import long_base_scanner as lb
import ipo_zone as iz
import ath_dma as ad
import rsi_scanner as rsi

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs" / "data"
CACHE = HERE / "cache"
CAPS = ["Large", "Mid", "Small", "Micro"]
SCREENER_TTL_DAYS = 3          # re-read each company page at least this often
CONCALL_FEED_DAYS = 75         # Concalls tab: calls held within this many days
MAX_NEW_TRANSCRIPTS = 60       # PDFs read per run (the rest wait for the next run)
DRY_UP = 0.75                  # stage 2 -> 3: 10-day avg volume <= 75% of the 50-day avg
MIN_MCAP_CR = 500              # micro caps below this are left out (as in the laptop scanners)


def log(*a):
    print(*a, flush=True)


def clean(x):
    """JSON-safe numbers."""
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 2)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def fmt_date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10]).strftime("%d %b %Y")
    except Exception:
        return str(s or "")


def num(x, d=1, pre="", post=""):
    x = clean(x)
    if x is None:
        return None
    return f"{pre}{x:,.{d}f}{post}"


# ------------------------------------------------------------------------- universe
def screener_index(code: str) -> dict:
    out = {}
    for page in range(1, 40):
        h = fp_get(f"https://www.screener.in/company/{code}/?limit=50&page={page}")
        if not h:
            break
        found = re.findall(r'<a href="/company/([^/"]+)/(?:consolidated/)?"[^>]*>\s*([^<]+?)\s*</a>', h)
        new = {s.replace("&amp;", "&"): n.replace("&amp;", "&").replace("&#x27;", "'")
               for s, n in found if s != code and not s.isdigit()}
        new = {s: n for s, n in new.items() if s not in out}
        if not new:
            break
        out.update(new)
        time.sleep(1.0)
    return out


def fp_get(url: str) -> str | None:
    import requests
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": es.UA["User-Agent"]}, timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            return r.text if r.status_code == 200 else None
        except Exception:
            time.sleep(3)
    return None


def build_universe() -> pd.DataFrame:
    """Large / Mid / Small from index membership (Nifty 100, Nifty 500, Smallcap 250);
    Micro = every other NSE stock with market cap >= MIN_MCAP_CR (same as the laptop scanners)."""
    f = HERE / "caps_fallback.csv"
    try:
        idx = {code: screener_index(code) for code in ("CNX500", "CNX100", "SMALLCA250")}
        if len(idx["CNX500"]) < 450 or len(idx["CNX100"]) < 90:
            raise RuntimeError("index lists incomplete")
        rows = [(s, n, "Large" if s in idx["CNX100"] else "Small" if s in idx["SMALLCA250"] else "Mid")
                for s, n in idx["CNX500"].items()]
        top = pd.DataFrame(rows, columns=["symbol", "name", "cap"])
        top.to_csv(f, index=False)
        log(f"Index members from Screener: {len(top)}")
    except Exception as e:
        log(f"Screener index pages not available ({e}); using caps_fallback.csv")
        top = pd.read_csv(f)
        top = top[top["cap"] != "Micro"]
    try:
        class A:                       # arguments the laptop universe builder expects
            symbols, universe, refresh = None, None, False
        allnse = es.build_universe(A, es.Config(min_mcap_cr=MIN_MCAP_CR, exclude_top=0))
    except Exception as e:
        log(f"NSE list / market caps not available ({e}); using the cached file")
        allnse = pd.read_csv(es.CACHE_DIR / "nse_universe_mcap.csv")
        allnse = allnse[allnse["mcap_cr"] >= MIN_MCAP_CR]
    names = dict(zip(allnse["symbol"], allnse["name"]))
    top["name"] = [names.get(s, n) for s, n in zip(top["symbol"], top["name"])]
    micro = allnse[~allnse["symbol"].isin(top["symbol"])][["symbol", "name"]].assign(cap="Micro")
    df = pd.concat([top, micro], ignore_index=True).drop_duplicates("symbol")
    log("Universe: " + ", ".join(f"{c} {int((df['cap'] == c).sum())}" for c in CAPS))
    return df


# ------------------------------------------------------------------------- company data
def load_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def company_pages(symbols: list[str], force: set) -> dict:
    """Concall months + document links + quarterly table + P/E for each symbol (cached)."""
    f = CACHE / "app_screener.json"
    cache = load_json(f, {})
    now = time.time()
    todo = [s for s in symbols if s in force or s not in cache
            or now - cache[s].get("ts", 0) > SCREENER_TTL_DAYS * 86400 or cache[s].get("error")]
    log(f"Screener company pages: {len(todo)} to refresh, {len(symbols) - len(todo)} from cache")

    def job(s):
        r = {"ts": time.time()}
        page = fp._fetch_screener(s)
        if page is None:
            r["error"] = "Screener not reachable"
            return s, r
        r["months"] = fp.parse_concalls(page)
        r["docs"] = cp.concall_docs(page)[:6]
        q = s2.parse_quarters(page)
        if not q.get("sales"):
            p2 = fp_get(f"https://www.screener.in/company/{s}/")
            if p2:
                q = s2.parse_quarters(p2) or q
                r["months"] = r["months"] or fp.parse_concalls(p2)
                r["docs"] = r["docs"] or cp.concall_docs(p2)[:6]
        r["q"] = {k: (v[-8:] if isinstance(v, list) else v) for k, v in q.items()}
        fund = ip.parse_fundamentals(page)
        try:
            r["shp"] = iz.parse_shareholding(page)
        except Exception:
            r["shp"] = {}
        r["pe"] = fund.get("Stock P/E")
        r["mcap"] = fund.get("Market Cap")
        time.sleep(1.2)
        return s, r

    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(job, s) for s in todo]):
            s, r = fut.result()
            if not r.get("error") or s not in cache:
                cache[s] = r
            done += 1
            if done % 50 == 0 or done == len(todo):
                log(f"  companies {done}/{len(todo)}")
                f.write_text(json.dumps(cache, default=str))
    f.write_text(json.dumps(cache, default=str))
    return cache


def concall_status(c: dict, listed: dict, sym: str, today: dt.date) -> dict:
    if not c or c.get("error") and "months" not in c:
        return {"ok": False, "status": "Unknown (Screener not reachable)", "last": "", "n24": None}
    v = ip.concall_check(c.get("months"), listed.get(sym, dt.date(2000, 1, 1)), today)
    last = v.get("last_concall", "")
    lab = ""
    if last:
        y, m = last.split("-")
        lab = dt.date(int(y), int(m), 1).strftime("%b %Y")
    return {"ok": v["concall_status"] == "Regular", "status": v["concall_status"],
            "last": lab, "n24": clean(v.get("concalls_24m"))}


# ------------------------------------------------------------------------- scanners
def scan_flag(sym, d):
    rows = [r for r in fp.analyse(sym, d, [fp.TIMEFRAMES["W"], fp.TIMEFRAMES["M"]]) if r["vol_flag"]]
    if not rows:
        return None
    r = sorted(rows, key=lambda x: (fp.STATUS_ORDER.get(x["status"], 9), -x["quality"]))[0]
    return {
        "status": f'{r["status"]} ({r["tf"].lower()})',
        "facts": [["Pole", f'+{r["pole_gain_pct"]:.0f}% in {r["pole_bars"]} {"wks" if r["tf"] == "Weekly" else "mths"}'],
                  ["Flag", f'{r["flag_bars"]} bars, gave back {r["retrace_pct"]:.0f}%'],
                  ["Unusual volume", f'{r["vol_x"]:.1f}x avg {r["vol_where"]} ({fmt_date(r["vol_date"])})']],
        "levels": {"Breakout": r["trigger"], "Target": r["target"], "Stop": r["invalidation"]},
        "sort": fp.STATUS_ORDER.get(r["status"], 9) * 10 - r["quality"],
    }


ELLIOTT_KEEP = ("W2B", "W2", "W3A")


def scan_elliott(sym, d):
    r = es.analyse(sym, d.iloc[-756:], es.Config())
    if r.get("code") not in ELLIOTT_KEEP:
        return None
    w1, w3 = bool(r.get("w1_spike_flag")), bool(r.get("w3_spike_flag"))
    if not (w1 or w3):
        return None
    facts = [["Wave", r["stage"]],
             ["Major low", f'Rs {r.get("origin_price")} on {fmt_date(r.get("origin_date"))} '
                           f'(after a {r.get("prior_drop_pct", 0):.0f}% fall)']]
    if w1:
        facts.append(["Unusual volume", f'{r["w1_spike"]:.1f}x avg at end of wave 1 ({fmt_date(r.get("w1_spike_date"))})'])
    if w3:
        facts.append(["Unusual volume", f'{r["w3_spike"]:.1f}x avg at start of wave 3 ({fmt_date(r.get("w3_spike_date"))})'])
    if clean(r.get("reward_risk")) is not None:
        facts.append(["Reward/risk", f'{r["reward_risk"]:.1f} to target 1'])
    return {"status": r["stage"], "facts": facts,
            "levels": {"Trigger": clean(r.get("trigger")), "Target 1": clean(r.get("t1")),
                       "Stop": clean(r.get("invalidation"))},
            "sort": -r.get("score", 0)}


def scan_base(sym, d, now):
    r = bb.analyse(sym, d, now)
    if not r or not r["vol_ok"]:
        return None
    vol = []
    if clean(r["wk_vol_x"]) is not None and r["wk_vol_x"] >= bb.P.wk_spike:
        vol.append(f'week {r["wk_vol_x"]:.1f}x ({fmt_date(r["wk_vol_week"])})')
    if clean(r["day_vol_x"]) is not None and r["day_vol_x"] >= bb.P.day_spike:
        vol.append(f'day {r["day_vol_x"]:.1f}x ({fmt_date(r["day_vol_date"])})')
    return {"status": r["status"],
            "facts": [["Base", f'{r["base_months"]:.0f} months, {r["width_pct"]:.0f}% wide, '
                               f'resistance tested {r["tests"]}x'],
                      ["Vs resistance", f'{r["vs_res_pct"]:+.1f}%'],
                      ["Unusual volume", ", ".join(vol)]],
            "levels": {"Resistance": r["resistance"], "Target": r["target"], "Stop": r["stop"]},
            "sort": bb.ORDER.get(r["status"], 9) * 100 - r["base_weeks"]}


def stage2_all(prices: dict) -> dict:
    """Trend template + maturity for every stock (logic of stage2_scanner.main)."""
    C = pd.DataFrame({s: d["Close"] for s, d in prices.items()}).sort_index()
    C = C[C.index.dayofweek < 5]
    H = pd.DataFrame({s: d["High"] for s, d in prices.items()}).reindex(C.index)
    L = pd.DataFrame({s: d["Low"] for s, d in prices.items()}).reindex(C.index)
    C, H, L = C.ffill(limit=3), H.ffill(limit=3), L.ffill(limit=3)
    rs = s2.rs_rating(C)
    crit, ma = s2.template(C, H, L, rs, 70)
    passed = crit[0].copy()
    for c in crit[1:]:
        passed &= c
    n_ok = sum(c.astype(int) for c in crit)
    out = {}
    for s in C.columns:
        try:
            if not np.isfinite(ma["s200"][s].iloc[-1]):
                continue
            k = int(n_ok[s].iloc[-1])
            pv = passed[s].values
            streak = s2.stage2_run(pv) if pv[-1] else 0
            recent = bool(pv[-60:].any())
            if k < 7 and not recent:
                continue
            d = prices[s].loc[:C.index[-1]]
            j = (len(C) - streak) if streak else sm.last_run_start(pv)
            if j is not None:
                j = sm.cycle_start(pv, ma["s200"][s].values, j)
            sd = C.index[j] if j is not None else None
            r = {"streak": streak, "k": k, "rs": clean(rs[s].iloc[-1]),
                 "cycle_days": (len(C) - j) if j is not None else 0, "_pre": sd is None,
                 "since": sd.date().isoformat() if sd is not None else "",
                 "months_up": s2.months_rising(ma["s200"][s]),
                 "from_high": round((C[s].iloc[-1] / ma["hi52"][s].iloc[-1] - 1) * 100, 1)}
            r.update(sm.technical(d, sd, ma["s50"][s], ma["s200"][s], rs[s], k, recent, r["cycle_days"]))
            dx, dd, wx, ud = s2.volume_info(d)
            r.update(day_vol_x=dx, day_vol_date=dd, wk_vol_x=wx, ud=ud)
            r.update(s2.vcp(d))
            v = d["Volume"].astype(float)
            r["dry"] = float(v.iloc[-10:].mean() / v.iloc[-60:-10].mean()) if v.iloc[-60:-10].mean() > 0 else np.nan
            out[s] = r
        except Exception as e:
            log(f"  stage2 {s}: {e}")
    return out


def finish_stage2(r: dict, comp: dict, price: float, sector=None) -> dict:
    fpts, fsig, fo = sm.fundamental((comp or {}).get("q") or {}, price, r["_start_price"], r["_start_date"])
    r.update(fo, _fpts=fpts, _fsig=fsig, _sector=sector)
    sm.label(r)
    return r


def fresh(r) -> bool:
    """Template turned true in the last 30 sessions and this is the start of the cycle."""
    return 0 < r["streak"] <= 30 and r["cycle_days"] <= r["streak"] + 10


def s2_card(r, early: bool):
    if early:
        st = ("Stage 2 - just started" if fresh(r) else "Early stage 2")
    else:
        st = r["maturity"]
    facts = [["Stage 2 since", f'{fmt_date(r["since"])} ({r["cycle_days"]} sessions)' if r["since"] else "-"],
             ["Trend template", f'{r["k"]}/8, RS rating {r["rs"]:.0f}' if r["rs"] is not None else f'{r["k"]}/8'],
             ["Base count", str(r.get("base_count", "-"))],
             ["From 52w high", f'{r["from_high"]:+.1f}%']]
    if r.get("vcp") and r["vcp"] != "No clear VCP":
        facts.append(["VCP", f'{r["vcp"]} ({r.get("contractions", "")})'])
    if early:
        if clean(r["day_vol_x"]) and r["day_vol_x"] >= 2:
            facts.append(["Volume", f'{r["day_vol_x"]:.1f}x avg on {fmt_date(r["day_vol_date"])}'])
    else:
        facts.append(["Volume drying", f'last 10 days = {r["dry"] * 100:.0f}% of normal'])
        if r.get("maturity_signals"):
            facts.append(["Warning signs", r["maturity_signals"]])
    lv = {"Pivot": clean(r.get("pivot"))} if early and clean(r.get("pivot")) else {}
    return {"status": st, "facts": facts, "levels": lv,
            "sort": (r["cycle_days"] if early else -r.get("maturity_score", 0))}


# ------------------------------------------------------------------------- concall feed
def concall_feed(meta, comp, listed, today, max_new):
    store_f = CACHE / "transcript_points.json"
    store = load_json(store_f, {})
    feed, new = [], 0
    cutoff = today - dt.timedelta(days=CONCALL_FEED_DAYS)
    cands = []
    for s, c in comp.items():
        if s not in meta or not c.get("docs"):
            continue
        d0 = c["docs"][0]
        y, m = map(int, d0["month"].split("-"))
        if dt.date(y, m, 28) < cutoff:
            continue
        cands.append((d0["month"], s))
    cands.sort(reverse=True)
    for month, s in cands:
        c, d0 = comp[s], comp[s]["docs"][0]
        url = d0.get("transcript")
        pts = store.get(url) if url else None
        if url and pts is None and new < max_new:
            try:
                pts = cp.fetch_transcript_points(url)
                store[url] = pts
            except Exception as e:
                pts = {"_error": str(e)[:80]}
                store[url] = pts if "not a PDF" not in str(e) else None
            new += 1
            time.sleep(1.0)
        snap = cp.results_snapshot(c.get("q") or {})
        points = {k: v for k, v in (pts or {}).items() if not k.startswith("_")}
        cc = concall_status(c, listed, s, today)
        feed.append({"sym": s, "name": meta[s]["name"], "cap": meta[s]["cap"],
                     "call": d0["label"], "headline": cp.headline(snap, points),
                     "results": snap.get("lines", []), "quarter": snap.get("quarter", ""),
                     "points": points, "pending": url is not None and pts is None,
                     "regular": cc["ok"], "concalls": cc,
                     "links": {k: d0[k] for k in ("transcript", "ppt", "rec") if d0.get(k)},
                     "pe": clean(c.get("pe")), "mcap": clean(c.get("mcap"))})
    store_f.write_text(json.dumps(store))
    log(f"Concall feed: {len(feed)} recent calls, {new} transcripts read this run")
    return feed


# ------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 60 stocks only (test run)")
    ap.add_argument("--max-transcripts", type=int, default=MAX_NEW_TRANSCRIPTS)
    a = ap.parse_args(argv)
    CACHE.mkdir(exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)      # IST
    today = now.date()

    uni = build_universe()
    if a.quick:
        uni = uni.groupby("cap").head(15)
    meta = uni.set_index("symbol")[["name", "cap"]].to_dict("index")
    log(f"Downloading 7 years of daily prices for {len(uni)} stocks ...")
    prices = es.download_prices(uni["symbol"].tolist(), 7)
    if not prices:
        sys.exit("No prices downloaded - Yahoo Finance not reachable.")
    asof = max(d.index[-1] for d in prices.values()).date()
    log(f"Prices up to {asof} for {len(prices)} stocks")

    try:
        listed = ip.listing_dates()
    except Exception:
        listed = {}
    hits = {k: {} for k in ("flag", "elliott", "base", "accum", "ipo", "ath", "dma", "rsi", "s2early", "s2late")}
    recent = {s for s, d in prices.items() if (today - d.index[0].date()).days <= iz.YEARS * 365 + 30}
    for i, (s, d) in enumerate(prices.items(), 1):
        d = d.dropna(subset=["High", "Low", "Close"])
        for key, fn in (("flag", lambda: scan_flag(s, d)), ("elliott", lambda: scan_elliott(s, d)),
                        ("base", lambda: scan_base(s, d, now)), ("accum", lambda: lb.scan_accum(s, d, now)),
                        ("ipo", lambda: lb.scan_ipo(s, d, now, listed.get(s))),
                        ("ath", lambda: None if s in recent or (listed.get(s) and (today - listed[s]).days <= iz.YEARS * 365)
                         else ad.first_ath(d)),
                        ("dma", lambda: ad.dma_cross(d, (50, 150, 200))),
                        ("rsi", lambda: rsi.scan(d))):
            try:
                r = fn()
                if r:
                    hits[key][s] = r
            except Exception as e:
                log(f"  {key} {s}: {e}")
        if i % 100 == 0:
            log(f"  patterns {i}/{len(prices)}")
    st2 = stage2_all(prices)
    early_c = {s for s, r in st2.items() if r["streak"] > 0 and not r["_pre"] and not r.get("_stage4")}
    late_c = {s for s, r in st2.items() if not r["_pre"] and np.isfinite(r.get("dry", np.nan)) and r["dry"] <= DRY_UP}
    log("Pattern hits before concall filter: " + ", ".join(f"{k} {len(v)}" for k, v in hits.items())
        + f", stage2 candidates {len(early_c)}/{len(late_c)}")

    if hits["ath"]:                                   # check candidates against full price history
        try:
            fh = ad.full_highs(sorted(hits["ath"]), log)
            for s in list(hits["ath"]):
                if s in fh:
                    ath_card = ad.first_ath(prices[s], fh[s])
                    if ath_card:
                        hits["ath"][s] = ath_card
                    else:
                        hits["ath"].pop(s)
            log(f"First ATH: {len(hits['ath'])} confirmed on full history ({len(fh)} checked)")
        except Exception as e:
            log(f"Full-history ATH check failed: {e}")
    cand = set().union(*[set(v) for v in hits.values()]) | early_c | late_c
    old = load_json(CACHE / "app_screener.json", {})
    cand |= {s for s in recent if "shp" not in old.get(s, {})}          # IPOs need shareholding once
    comp = company_pages(sorted(meta), force=cand)

    log("Sector rotation ...")
    sectors, sector_of = sr.run(prices, {s: m["cap"] for s, m in meta.items()})
    for s in early_c | late_c:
        finish_stage2(st2[s], comp.get(s), float(prices[s]["Close"].iloc[-1]), sector_of.get(s))
    for s in early_c:
        r = st2[s]
        if fresh(r) or r.get("maturity") == "Early stage 2":
            hits["s2early"][s] = s2_card(r, True)
    for s in late_c:
        r = st2[s]
        if r.get("maturity") in ("Late stage 2", "Topping - stage 3 risk"):
            hits["s2late"][s] = s2_card(r, False)

    def make_item(s, card, cc):
        c = comp.get(s, {})
        snap = cp.results_snapshot(c.get("q") or {})
        px = float(prices[s]["Close"].iloc[-1])
        return {
            "sym": s, "name": meta[s]["name"], "price": round(px, 2),
            "chg": clean((px / float(prices[s]["Close"].iloc[-2]) - 1) * 100) if len(prices[s]) > 1 else None,
            "mcap": clean(c.get("mcap")), "pe": clean(c.get("pe")),
            "status": card["status"], "facts": [[a, b] for a, b in card["facts"] if b],
            "levels": {k: clean(v) for k, v in card["levels"].items() if clean(v) is not None},
            "concalls": cc, "quarter": snap.get("quarter", ""), "results": snap.get("lines", [])[:3],
            **{k: card[k] for k in ("tags", "lockin", "lockin_note") if k in card},
            "_sort": card["sort"]}

    ipo_zone = {}
    try:
        zone, note = iz.build({s: prices[s] for s in recent}, listed, comp, es)
        ipo_zone = {"n": len(recent), "note": note, "sections": {}}
        for k, items in zone.items():
            rows = []
            for s, card in items.items():
                it = make_item(s, card, concall_status(comp.get(s), listed, s, today))
                it["cap"] = meta[s]["cap"]
                rows.append(it)
            rows.sort(key=lambda x: x.pop("_sort"))
            ipo_zone["sections"][k] = rows
        log("IPO Zone: " + ", ".join(f"{k} {len(v)}" for k, v in ipo_zone["sections"].items()))
    except Exception as e:
        log(f"IPO Zone failed: {e}")

    caps = {c: {k: [] for k in list(hits) + ["concalls"]} for c in CAPS}
    dropped = {k: 0 for k in hits}
    for key, items in hits.items():
        for s, card in items.items():
            cc = concall_status(comp.get(s), listed, s, today)
            if not cc["ok"]:
                dropped[key] += 1
                continue
            caps[meta[s]["cap"]][key].append(make_item(s, card, cc))
    for c in CAPS:
        for key in hits:
            caps[c][key].sort(key=lambda x: x.pop("_sort"))
    for item in concall_feed(meta, comp, listed, today, a.max_transcripts):
        caps[item["cap"]]["concalls"].append(item)

    data = {"asof": asof.isoformat(), "generated": now.strftime("%Y-%m-%d %H:%M IST"),
            "universe": {c: int((uni["cap"] == c).sum()) for c in CAPS},
            "dropped_no_concalls": dropped, "caps": caps, "sectors": sectors, "ipo_zone": ipo_zone}
    txt = json.dumps(data, ensure_ascii=False, default=clean, separators=(",", ":"))
    (DOCS / "latest.json").write_text(txt, encoding="utf-8")
    (DOCS / "latest.js").write_text("window.SCAN_DATA=" + txt + ";", encoding="utf-8")
    hist = DOCS / "history"
    hist.mkdir(exist_ok=True)
    (hist / f"{asof.isoformat()}.json").write_text(txt, encoding="utf-8")
    log("Done: " + "; ".join(f"{c}: " + ", ".join(f"{k} {len(v)}" for k, v in caps[c].items()) for c in CAPS))


if __name__ == "__main__":
    main()
