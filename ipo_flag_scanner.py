#!/usr/bin/env python3
"""
Recent-IPO flag & pole scanner (NSE small & micro caps listed in the last 5 years).

A stock makes the final list when:
  1. Listed on NSE within the last 5 years (NSE "DATE OF LISTING"), market cap >= 500 crore,
     outside the top 250 by market cap (small + micro caps).
  2. Weekly or monthly chart shows a bull flag that is either still forming (in flag / near
     breakout) or has a CONFIRMED breakout (a completed bar closed above the flag and price is
     still holding above the breakout level).
  3. Unusual volume BOTH near the pole (pole bars incl. the top) AND inside the flag
     (weekly >= 3x the 20-week average, monthly >= 2.5x the 12-month average; the listing
     bar itself is ignored because IPO-day volume is always huge).
  4. Regular concalls (Screener.in), with the bar scaled to how long the company has been listed.

For every candidate it also reports:
  - Forward P/E and forward PEG for the next 3 financial years. EPS growth comes from, in order:
    your own numbers in guidance.csv (e.g. from concall guidance) > Yahoo analyst consensus >
    the company's own recent EPS growth, faded 15% a year. The source is shown for each stock.
  - NSE bulk and block deals on the listing day.

Run:  python ipo_flag_scanner.py
Not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import io
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

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
CACHE_DIR = HERE / "cache"
GUIDANCE_FILE = HERE / "guidance.csv"

MAX_BRK = {"W": 4, "M": 2}           # how many bars back a confirmed breakout may be
LISTING_BARS = {"W": 1, "M": 1}      # bars ignored for volume (listing week / month)


# --------------------------------------------------------------------------------------
# Listing dates
# --------------------------------------------------------------------------------------
def listing_dates(refresh=False) -> dict:
    import requests
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / "nse_listing_dates.csv"
    if f.exists() and not refresh and time.time() - f.stat().st_mtime < 86400:
        df = pd.read_csv(f)
    else:
        last = None
        for url in es.NSE_URLS:
            try:
                r = requests.get(url, headers=es.UA, timeout=30)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip().upper() for c in df.columns]
                df = df[["SYMBOL", "DATE OF LISTING"]].rename(
                    columns={"SYMBOL": "symbol", "DATE OF LISTING": "listed"})
                df["symbol"] = df["symbol"].astype(str).str.strip()
                df.to_csv(f, index=False)
                break
            except Exception as e:
                last = e
        else:
            if f.exists():
                df = pd.read_csv(f)
            else:
                raise RuntimeError(f"Could not download NSE listing dates ({last})")
    d = pd.to_datetime(df["listed"], format="%d-%b-%Y", errors="coerce")
    return {s: x.date() for s, x in zip(df["symbol"], d) if pd.notna(x)}


# --------------------------------------------------------------------------------------
# Pattern detection (flag still forming, or breakout confirmed)
# --------------------------------------------------------------------------------------
def last_bar_complete(daily: pd.DataFrame, tf: str, now: dt.datetime) -> bool:
    d = daily.index[-1].date()
    closed = d < now.date() or now.time() >= dt.time(15, 35)
    if not closed:
        return False
    nxt = (pd.Timestamp(d) + pd.offsets.BDay(1)).date()
    if tf == "W":
        return nxt.isocalendar()[1] != d.isocalendar()[1]
    return nxt.month != d.month


def _rel_vol(V: np.ndarray, avg: int, skip: int) -> np.ndarray:
    v = pd.Series(V.astype(float)).replace(0, np.nan)
    v.iloc[:skip] = np.nan                                # listing bar(s)
    base = v.shift(1).rolling(avg, min_periods=4).mean()
    fallback = v.shift(1).expanding(min_periods=2).median()
    base = base.fillna(fallback)
    return (v / base).values


def _spike(rv, lo, hi):
    lo, hi = max(lo, 0), min(hi, len(rv) - 1)
    if hi < lo:
        return np.nan, None
    seg = rv[lo:hi + 1]
    if not np.isfinite(seg).any():
        return np.nan, None
    k = lo + int(np.nanargmax(seg))
    return float(rv[k]), k


def find_flag(bars: pd.DataFrame, tf: fp.TF, complete: bool) -> dict | None:
    H, L, C = bars["High"].values, bars["Low"].values, bars["Close"].values
    n = len(bars)
    if n < tf.pole_min + tf.flag_min + 2:
        return None
    rv = _rel_vol(bars["Volume"].values, tf.vol_avg, LISTING_BARS[tf.code])
    best = None
    for brk in range(0, MAX_BRK[tf.code] + 1):
        e = n - 1 - brk
        for t in range(max(tf.pole_min, e - tf.flag_max), e - tf.flag_min + 1):
            top = H[t]
            fl = slice(t + 1, e + 1)
            if H[fl].max() > top * 1.02:
                continue
            s_lo, s_hi = max(0, t - tf.pole_max), t - tf.pole_min
            if s_hi < s_lo:
                continue
            win = L[s_lo:s_hi + 1]
            s = s_lo + int(np.where(win <= win.min() * 1.03)[0][-1])
            base = L[s]
            if base <= 0 or L[s:t + 1].min() < base * 0.97 or H[s:t + 1].max() > top:
                continue
            gain = top / base - 1
            if gain < tf.pole_gain:
                continue
            height, pole_len, flag_len = top - base, t - s, e - t
            if flag_len > max(4 * pole_len, tf.flag_min + 2):
                continue
            if C[t] < base + 0.6 * height and C[min(t + 1, e)] < base + 0.6 * height:
                continue
            f_low = L[fl].min()
            retr = (top - f_low) / height
            if retr > tf.max_retrace:
                continue
            slope, _ = fp._line(C[fl])
            if slope * flag_len > 0.15 * height:
                continue
            hb, ha = fp._line(np.r_[top, H[fl]])
            upper = lambda i: ha + hb * (i - t)
            f_high = H[fl].max()
            if brk == 0:
                if C[-1] > upper(n - 1) and C[-1] > f_high:
                    continue
                trig = min(max(upper(n), float(H[fl][-3:].max()) if flag_len >= 3 else top), top)
                status = "Near breakout" if C[-1] >= trig * 0.95 else "In flag"
            else:
                b0 = e + 1
                trig = float(max(upper(b0), f_high))
                if not C[b0] > trig:
                    continue                      # first bar after the flag must close above
                if L[b0:].min() < f_low or C[-1] < trig * 0.97:
                    continue                      # breakout failed
                if brk == 1 and not complete:
                    status = "Breakout - bar not closed yet"
                else:
                    status = "Breakout confirmed"
            pv, pk = _spike(rv, s + 1, t)         # near the pole (pole bars incl. top)
            fv, fk = _spike(rv, t + 1, e)         # inside the flag
            q = gain * (1 - retr) * (1.15 if "Breakout" in status else 1.0)
            cand = dict(status=status, s=s, t=t, e=e, brk=brk, base=float(base), top=float(top),
                        gain=gain, pole_len=pole_len, flag_len=flag_len, flag_low=float(f_low),
                        flag_high=float(f_high), retrace=retr, trigger=float(trig), ha=ha, hb=hb,
                        pole_vol=pv, pole_vol_i=pk, flag_vol=fv, flag_vol_i=fk, quality=q,
                        brk_bar=(e + 1) if brk else None)
            if best is None or q > best["quality"]:
                best = cand
    return best


def analyse(sym, daily, tfs, now) -> list[dict]:
    daily = daily.dropna(subset=["High", "Low", "Close"])
    rows = []
    for tf in tfs:
        bars = fp.resample(daily, tf.rule)
        f = find_flag(bars, tf, last_bar_complete(daily, tf.code, now))
        if not f:
            continue
        idx = bars.index
        d = lambda i: idx[i].date().isoformat() if i is not None else ""
        c = float(daily["Close"].iloc[-1])
        height = f["top"] - f["base"]
        target = f["trigger"] + height
        pole_ok = np.isfinite(f["pole_vol"]) and f["pole_vol"] >= tf.vol_spike
        flag_ok = np.isfinite(f["flag_vol"]) and f["flag_vol"] >= tf.vol_spike
        rows.append({
            "symbol": sym, "tf": tf.name, "status": f["status"], "price": round(c, 2),
            "pole_start_date": d(f["s"]), "pole_start": round(f["base"], 2),
            "pole_top_date": d(f["t"]), "pole_top": round(f["top"], 2),
            "pole_gain_pct": round(f["gain"] * 100, 1), "pole_bars": f["pole_len"],
            "flag_bars": f["flag_len"], "flag_low": round(f["flag_low"], 2),
            "retrace_pct": round(f["retrace"] * 100, 1), "trigger": round(f["trigger"], 2),
            "breakout_bar": d(f["brk_bar"]),
            "to_trigger_pct": round((f["trigger"] / c - 1) * 100, 1),
            "target": round(target, 2), "to_target_pct": round((target / c - 1) * 100, 1),
            "invalidation": round(f["flag_low"], 2), "risk_pct": round((c / f["flag_low"] - 1) * 100, 1),
            "pole_vol_x": round(f["pole_vol"], 2) if np.isfinite(f["pole_vol"]) else np.nan,
            "pole_vol_date": d(f["pole_vol_i"]),
            "flag_vol_x": round(f["flag_vol"], 2) if np.isfinite(f["flag_vol"]) else np.nan,
            "flag_vol_date": d(f["flag_vol_i"]),
            "pole_vol_ok": bool(pole_ok), "flag_vol_ok": bool(flag_ok),
            "quality": round(f["quality"], 3), "_bars": bars, "_f": f, "_tf": tf,
        })
    return rows


# --------------------------------------------------------------------------------------
# Screener.in: concalls + EPS history
# --------------------------------------------------------------------------------------
def _num(s):
    s = re.sub(r"<[^>]+>", "", s or "").replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_fundamentals(page: str) -> dict:
    out = {}
    top = page[page.find('id="top-ratios"'):]
    top = top[:top.find("</ul>")]
    for name, val in re.findall(r'<span class="name">\s*(.*?)\s*</span>\s*<span class="nowrap value">(.*?)</span>\s*</li>',
                                top, re.S):
        nums = re.findall(r'<span class="number">([^<]*)</span>', val)
        if nums:
            out[re.sub(r"\s+", " ", name).strip()] = _num(nums[0])
    i = page.find('id="profit-loss"')
    if i != -1:
        sec = page[i:]
        sec = sec[:sec.find("</section>")]
        heads = [re.sub(r"\s+", " ", h).strip()
                 for h in re.findall(r"<th[^>]*>(.*?)</th>", sec[sec.find("<thead"):sec.find("</thead>")], re.S)]
        heads = [h for h in heads if h]
        m = re.search(r'<td class="text">\s*EPS in Rs\s*</td>(.*?)</tr>', sec, re.S)
        if m:
            vals = [_num(v) for v in re.findall(r"<td[^>]*>(.*?)</td>", m.group(1), re.S)]
            out["eps"] = {h: v for h, v in zip(heads[-len(vals):], vals)}
        for title, key in (("Compounded Profit Growth", "profit_growth"),
                           ("Compounded Sales Growth", "sales_growth")):
            j = sec.find(title)
            if j != -1:
                tbl = sec[j:sec.find("</table>", j)]
                out[key] = {k.strip(": ").strip(): _num(v) for k, v in
                            re.findall(r"<td>(.*?)</td>\s*<td>(.*?)</td>", tbl, re.S)}
    return out


def fetch_company(sym: str) -> dict:
    import requests
    res = {"ts": time.time()}
    page = fp._fetch_screener(sym)
    if page is None:
        res["error"] = "Screener not reachable"
        return res
    res["concalls"] = fp.parse_concalls(page)
    fund = parse_fundamentals(page)
    eps = fund.get("eps") or {}
    if not any(np.isfinite(v) for v in eps.values()):      # consolidated empty -> standalone
        try:
            r = requests.get(f"https://www.screener.in/company/{sym}/",
                             headers={"User-Agent": es.UA["User-Agent"]}, timeout=30)
            if r.status_code == 200:
                f2 = parse_fundamentals(r.text)
                if f2.get("eps"):
                    fund = f2
                    res["concalls"] = res["concalls"] or fp.parse_concalls(r.text)
        except Exception:
            pass
    res.update({k: v for k, v in fund.items()})
    return res


def company_data(symbols: list[str]) -> dict:
    f = CACHE_DIR / "ipo_screener.json"
    cache = {}
    if f.exists():
        try:
            cache = json.loads(f.read_text())
        except Exception:
            cache = {}
    todo = [s for s in symbols if s not in cache or time.time() - cache[s].get("ts", 0) > 3 * 86400
            or cache[s].get("error")]
    if todo:
        print(f"Reading concalls and EPS history from Screener.in for {len(todo)} companies ...")

    def job(s):
        r = fetch_company(s)
        time.sleep(1.2)
        return s, r

    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(job, s) for s in todo]):
            s, r = fut.result()
            cache[s] = r
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"  companies: {done}/{len(todo)}", flush=True)
    try:
        f.write_text(json.dumps(cache, default=str))
    except Exception:
        pass
    return {s: cache.get(s, {}) for s in symbols}


def concall_check(months, listed: dt.date, today: dt.date) -> dict:
    age = (today.year - listed.year) * 12 + today.month - listed.month
    possible = max(0, min(8, age // 3))
    need = max(1, min(4, math.ceil(0.6 * possible)))
    if months is None:
        return {"concall_status": "No concall section", "concalls_24m": 0, "last_concall": "",
                "concalls_needed": need}
    v = fp.concall_verdict(months, today, min_calls=need)
    if possible < 1 and not months:
        v["concall_status"] = "Too new to judge"
    v["concalls_needed"] = need
    return v


# --------------------------------------------------------------------------------------
# Forward PEG
# --------------------------------------------------------------------------------------
def load_guidance() -> dict:
    if not GUIDANCE_FILE.exists():
        GUIDANCE_FILE.write_text(
            "symbol,eps_growth_y1_pct,eps_growth_y2_pct,eps_growth_y3_pct,note\n"
            "# Example (remove the # to use): PREMIERENE,30,25,20,FY27 guidance from Q2 concall\n",
            encoding="utf-8")
        return {}
    try:
        g = pd.read_csv(GUIDANCE_FILE, comment="#")
    except Exception:
        return {}
    out = {}
    for _, r in g.iterrows():
        vals = [r.get(f"eps_growth_y{i}_pct") for i in (1, 2, 3)]
        if pd.notna(vals[0]):
            vals = [float(v) / 100 if pd.notna(v) else float(vals[0]) / 100 for v in vals]
            out[str(r["symbol"]).strip().upper()] = vals
    return out


def yahoo_estimates(symbols: list[str]) -> dict:
    import yfinance as yf
    es._quiet_yf()
    out = {}

    def job(s):
        try:
            e = yf.Ticker(s + ".NS").earnings_estimate
            if e is None or e.empty:
                return s, None
            y0 = e.loc["0y"] if "0y" in e.index else None
            y1 = e.loc["+1y"] if "+1y" in e.index else None
            if y0 is None or not np.isfinite(y0.get("avg", np.nan)):
                return s, None
            return s, {"eps0": float(y0["avg"]), "eps1": float(y1["avg"]) if y1 is not None and np.isfinite(y1.get("avg", np.nan)) else None,
                       "base": float(y0.get("yearAgoEps", np.nan)),
                       "analysts": int(y0.get("numberOfAnalysts", 0) or 0)}
        except Exception:
            return s, None

    with ThreadPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(job, s) for s in symbols]):
            s, v = f.result()
            if v:
                out[s] = v
    return out


def forward_peg(sym: str, price: float, comp: dict, guide: dict, yest: dict) -> dict:
    eps = comp.get("eps") or {}
    annual = [(k, v) for k, v in eps.items() if re.match(r"[A-Z][a-z]{2} \d{4}$", k) and v is not None
              and np.isfinite(v)]
    if not annual:
        return {"peg_note": "No EPS history on Screener"}
    base_lbl, base_eps = annual[-1]
    fy = int(base_lbl[-4:])
    labels = [f"FY{(fy + i) % 100:02d}E" for i in (1, 2, 3)]
    out = {"base_fy": f"FY{fy % 100:02d}", "base_eps": round(base_eps, 2),
           "ttm_eps": eps.get("TTM"), "fy_labels": labels}
    if base_eps <= 0:
        out["peg_note"] = "Loss-making in the base year - PEG not meaningful"
        return out
    g = None
    if sym in guide:
        g, src = guide[sym], "Your guidance.csv"
    elif sym in yest and yest[sym].get("eps0"):
        y = yest[sym]
        g1 = y["eps0"] / base_eps - 1
        g2 = (y["eps1"] / y["eps0"] - 1) if y.get("eps1") else g1 * 0.85
        g3 = max(0.0, g2 * 0.85)
        g, src = [g1, g2, g3], f"Yahoo consensus ({y['analysts']} analysts) + faded"
    else:
        vals = [v for _, v in annual[-4:]]
        hist = None
        if len(vals) >= 2 and vals[0] > 0:
            hist = (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1
        pg = comp.get("profit_growth") or {}
        if hist is None or not np.isfinite(hist):
            hist = next((pg.get(k) / 100 for k in ("3 Years", "TTM") if pg.get(k) is not None
                         and np.isfinite(pg.get(k))), None)
        if hist is None:
            out["peg_note"] = "Not enough EPS history to project growth"
            return out
        hist = float(np.clip(hist, 0.0, 0.50))
        g, src = [hist, hist * 0.85, hist * 0.7], f"Own EPS CAGR {hist:.0%} (last {len(vals) - 1}y), faded 15%/yr, capped 50%"
    e = base_eps
    for i, gi in enumerate(g, 1):
        e = e * (1 + gi)
        pe = price / e if e > 0 else np.nan
        out[f"eps_y{i}"] = round(e, 2)
        out[f"growth_y{i}"] = round(gi * 100, 1)
        out[f"pe_y{i}"] = round(pe, 1) if np.isfinite(pe) else np.nan
        out[f"peg_y{i}"] = round(pe / (gi * 100), 2) if gi > 0 and np.isfinite(pe) else np.nan
    cagr = (out["eps_y3"] / base_eps) ** (1 / 3) - 1 if out.get("eps_y3", 0) > 0 else np.nan
    out["eps_cagr_3y"] = round(cagr * 100, 1) if np.isfinite(cagr) else np.nan
    out["peg_3y"] = (round(out["pe_y1"] / (cagr * 100), 2)
                     if np.isfinite(cagr) and cagr > 0 and np.isfinite(out.get("pe_y1", np.nan)) else np.nan)
    out["growth_source"] = src
    if not any(np.isfinite(out.get(f"peg_y{i}", np.nan)) for i in (1, 2, 3)):
        out["peg_note"] = "Projected EPS growth is zero or negative - PEG not meaningful; add guidance.csv"
    return out


# --------------------------------------------------------------------------------------
# Listing-day bulk / block deals (NSE)
# --------------------------------------------------------------------------------------
def listing_deals(items: list[tuple[str, dt.date]]) -> dict:
    import requests
    f = CACHE_DIR / "listing_day_deals.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    todo = [(s, d) for s, d in items if s not in cache]
    if todo:
        print(f"Fetching NSE bulk/block deals on the listing day for {len(todo)} stocks ...")
        sess = requests.Session()
        sess.headers.update({**es.UA, "Accept": "application/json, text/plain, */*",
                             "Referer": "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"})
        try:
            sess.get("https://www.nseindia.com", timeout=20)
            sess.get("https://www.nseindia.com/report-detail/display-bulk-and-block-deals", timeout=20)
        except Exception:
            pass
        for i, (s, d) in enumerate(todo, 1):
            rec = {"bulk": [], "block": [], "ok": True}
            ds = d.strftime("%d-%m-%Y")
            for kind in ("bulk", "block"):
                url = ("https://www.nseindia.com/api/historicalOR/bulk-block-short-deals?"
                       f"optionType={kind}_deals&symbol={s}&from={ds}&to={ds}")
                for attempt in range(2):
                    try:
                        r = sess.get(url, timeout=30)
                        if r.status_code == 200 and r.text.strip().startswith("{"):
                            rec[kind] = r.json().get("data", []) or []
                            break
                        if attempt == 1:
                            rec["ok"] = False
                    except Exception:
                        if attempt == 1:
                            rec["ok"] = False
                    time.sleep(2)
                time.sleep(1.0)
            if rec["ok"]:
                cache[s] = rec
            else:
                cache.setdefault("_failed", {})[s] = True
            if i % 10 == 0 or i == len(todo):
                print(f"  deals: {i}/{len(todo)}", flush=True)
        failed = cache.pop("_failed", {})
        try:
            f.write_text(json.dumps(cache))
        except Exception:
            pass
        for s in failed:
            cache[s] = {"ok": False}
    return cache


def _deal_fields(rec: dict) -> dict:
    if not rec or not rec.get("ok", True):
        return {"listing_deals": "Not available (NSE did not respond)"}
    rows = []
    for kind in ("bulk", "block"):
        for d in rec.get(kind, []):
            g = lambda *ks: next((d[k] for k in d if any(x in k for x in ks)), None)
            rows.append({"kind": kind.title(), "client": g("CLIENT"), "side": g("BUY_SELL"),
                         "qty": g("QTY"), "price": g("WATP", "PRICE")})
    if not rows:
        return {"listing_deals": "No bulk/block deals", "_deals": []}
    buys = [r for r in rows if str(r["side"]).upper().startswith("B")]
    sells = [r for r in rows if str(r["side"]).upper().startswith("S")]
    fmt = lambda r: (f"{r['client']} {r['side']} {float(r['qty'] or 0) / 1e5:.2f}L"
                     f" @ {float(r['price'] or 0):,.2f}")
    return {"listing_deals": f"{len(rows)} deal(s): {len(buys)} buy, {len(sells)} sell",
            "listing_buyers": "; ".join(fmt(r) for r in buys)[:900],
            "listing_sellers": "; ".join(fmt(r) for r in sells)[:900],
            "_deals": rows}


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
COLS = [
    ("symbol", "Symbol"), ("name", "Company"), ("listed", "Listed on"), ("months_listed", "Months listed"),
    ("mcap_cr", "Mcap (cr)"), ("price", "Price"), ("tf", "Timeframe"), ("status", "Status"),
    ("breakout_bar", "Breakout bar"), ("concall_status", "Concalls"), ("concalls_24m", "Concalls (24m)"),
    ("concalls_needed", "Needed"), ("last_concall", "Last concall"),
    ("pole_vol_x", "Pole volume (x avg)"), ("pole_vol_date", "Pole spike bar"),
    ("flag_vol_x", "Flag volume (x avg)"), ("flag_vol_date", "Flag spike bar"),
    ("trigger", "Breakout level"), ("to_trigger_pct", "To breakout %"),
    ("target", "Measured target"), ("to_target_pct", "To target %"),
    ("invalidation", "Flag low (stop)"), ("risk_pct", "Risk %"),
    ("pole_gain_pct", "Pole gain %"), ("pole_bars", "Pole bars"), ("pole_start_date", "Pole start"),
    ("pole_top_date", "Pole top"), ("flag_bars", "Flag bars"), ("retrace_pct", "Flag retrace %"),
    ("base_fy", "Base FY"), ("base_eps", "Base EPS"),
    ("eps_y1", "EPS Y1"), ("growth_y1", "Growth Y1 %"), ("pe_y1", "Fwd P/E Y1"), ("peg_y1", "Fwd PEG Y1"),
    ("eps_y2", "EPS Y2"), ("growth_y2", "Growth Y2 %"), ("pe_y2", "Fwd P/E Y2"), ("peg_y2", "Fwd PEG Y2"),
    ("eps_y3", "EPS Y3"), ("growth_y3", "Growth Y3 %"), ("pe_y3", "Fwd P/E Y3"), ("peg_y3", "Fwd PEG Y3"),
    ("eps_cagr_3y", "EPS CAGR 3y %"), ("peg_3y", "PEG on 3y CAGR"), ("growth_source", "Growth source"),
    ("peg_note", "PEG note"), ("listing_deals", "Listing-day bulk/block deals"),
    ("listing_buyers", "Listing-day buyers"), ("listing_sellers", "Listing-day sellers"),
    ("quality", "Pattern quality"),
]
ORDER = {"Breakout confirmed": 0, "Near breakout": 1, "In flag": 2, "Breakout - bar not closed yet": 3}


def frame(rows):
    df = pd.DataFrame([{k: r.get(k, np.nan) for k, _ in COLS} |
                       {"_both": r["pole_vol_ok"] and r["flag_vol_ok"],
                        "_one": r["pole_vol_ok"] or r["flag_vol_ok"],
                        "_cc": r.get("concall_status") == "Regular",
                        "_ok_status": r["status"] in ("Breakout confirmed", "Near breakout", "In flag")}
                       for r in rows])
    if df.empty:
        return df
    df["_o"] = df["status"].map(ORDER).fillna(9)
    return df.sort_values(["_o", "peg_y1"], ascending=[True, True], na_position="last")


def write_excel(df, rows, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    hdr = dict(COLS)
    hide = [c for c in df.columns if c.startswith("_")]
    final = df[df["_both"] & df["_cc"] & df["_ok_status"]]
    sheets = {
        "Final list": final,
        "Breakout pending close": df[df["_both"] & df["_cc"] & ~df["_ok_status"]],
        "One volume spike only": df[df["_one"] & ~df["_both"] & df["_cc"]],
        "Concalls not regular": df[df["_one"] & ~df["_cc"]],
    }
    deal_rows = []
    for r in rows:
        if r["symbol"] in set(df[df["_one"]]["symbol"]):
            for d in r.get("_deals", []) or []:
                deal_rows.append({"Symbol": r["symbol"], "Listed on": r.get("listed"), "Type": d["kind"],
                                  "Client": d["client"], "Buy/Sell": d["side"], "Quantity": d["qty"],
                                  "Price": d["price"]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, t in sheets.items():
            t.drop(columns=hide).rename(columns=hdr).to_excel(xw, sheet_name=name, index=False)
        pd.DataFrame(deal_rows or [{"Symbol": "No listing-day deals among the candidates"}]).drop_duplicates() \
            .to_excel(xw, sheet_name="Listing-day deals", index=False)
        pd.DataFrame({"Item": [
            "Universe", "Status", "Pole / flag volume", "Concalls", "Forward PEG",
            "Growth source", "guidance.csv", "Listing-day deals", "Caveat"], "Meaning": [
            "NSE stocks listed in the last 5 years (NSE listing date; SME migrations count from their "
            "main-board date), mcap >= 500 cr, outside the top 250.",
            "In flag / Near breakout (within 5% of the flag top line) / Breakout confirmed (a completed "
            "bar closed above the flag within the last 4 weeks or 2 months, and price still holds above "
            "the breakout level). 'Bar not closed yet' = breaking out in the current, unfinished bar.",
            "Largest bar volume among the pole bars, and inside the flag, versus the average of the previous "
            "20 weeks (weekly, needs 3x) or 12 months (monthly, needs 2.5x). The listing bar is ignored.",
            "Screener.in concall list. 'Needed' scales with time since listing: 60% of the quarters "
            "since listing, max 4 in 24 months; latest call within 7 months.",
            "Fwd P/E = today's price / projected EPS of that year. PEG = Fwd P/E / EPS growth % of that "
            "year. Under 1 = cheap for its growth; above 2 = expensive. PEG on 3y CAGR uses Y1 P/E and "
            "the 3-year EPS CAGR.",
            "Your guidance.csv first, then Yahoo analyst consensus (current and next year, third year "
            "faded), otherwise the company's own recent EPS CAGR (capped at 50%, faded 15% a year).",
            "Put your concall-based EPS growth (in %) for the next 3 years per symbol; the scanner uses "
            "it instead of the other sources.",
            "NSE bulk and block deals on the NSE listing date (client, buy/sell, quantity, price).",
            "Mechanical pattern scan; projections are estimates. Not investment advice."]}
        ).to_excel(xw, sheet_name="How to read", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for j, cell in enumerate(ws[1], start=1):
                ws.column_dimensions[get_column_letter(j)].width = {
                    "Company": 28, "Status": 20, "Concalls": 22, "Growth source": 40, "PEG note": 30,
                    "Listing-day bulk/block deals": 26, "Listing-day buyers": 60, "Listing-day sellers": 60,
                    "Client": 45, "Item": 18, "Meaning": 110}.get(cell.value, 12)
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
    return len(final)


def chart(r, w=330, h=150):
    bars, f = r["_bars"], r["_f"]
    start = max(0, min(f["s"] - 6, len(bars) - 60))
    b = bars.iloc[start:]
    n = len(b)
    H, L, C, V = b["High"].values, b["Low"].values, b["Close"].values, b["Volume"].values.astype(float)
    lo, hi = float(L.min()), float(H.max())
    pad, ph, vh = 14, 98, 30
    X = lambda i: pad + (w - 2 * pad) * (i + 0.5) / n
    Y = lambda p: pad + ph * (hi - p) / max(hi - lo, 1e-9)
    bw = max(1.0, (w - 2 * pad) / n * 0.6)
    s, t, e = f["s"] - start, f["t"] - start, f["e"] - start
    p = [f'<rect x="{X(t):.1f}" y="{Y(f["flag_high"]):.1f}" width="{X(e) - X(t):.1f}" '
         f'height="{Y(f["flag_low"]) - Y(f["flag_high"]):.1f}" class="fz"/>']
    for i in range(n):
        c = "pc" if s <= i <= t else ("bo" if f.get("brk_bar") is not None and i >= f["brk_bar"] - start else "cd")
        p.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(H[i]):.1f}" y2="{Y(L[i]):.1f}" class="{c}"/>'
                 f'<line x1="{X(i) - bw / 2:.1f}" x2="{X(i) + bw / 2:.1f}" y1="{Y(C[i]):.1f}" y2="{Y(C[i]):.1f}" class="{c}"/>')
    y0, y1 = min(max(f["ha"], lo), hi), min(max(r["trigger"], lo), hi)
    p.append(f'<line x1="{X(t):.1f}" x2="{X(n - 1):.1f}" y1="{Y(y0):.1f}" y2="{Y(y1):.1f}" class="tg"/>')
    vmax = float(np.nanmax(V)) or 1.0
    by = pad + ph + 8 + vh
    hot = {k - start for k, ok in ((f["pole_vol_i"], r["pole_vol_ok"]), (f["flag_vol_i"], r["flag_vol_ok"]))
           if k is not None and ok}
    for i in range(n):
        hh = vh * min(1.0, V[i] / vmax) if np.isfinite(V[i]) else 0
        p.append(f'<rect x="{X(i) - bw / 2:.1f}" y="{by - hh:.1f}" width="{bw:.1f}" height="{hh:.1f}" '
                 f'class="{"vs" if i in hot else "vb"}"/>')
    return (f'<svg viewBox="0 0 {w} {by + 4:.0f}" class="ch" role="img" '
            f'aria-label="{r["tf"]} chart: pole, flag and volume">{"".join(p)}</svg>')


def write_html(rows, path, title):
    keep = lambda r: r["pole_vol_ok"] and r["flag_vol_ok"] and r.get("concall_status") == "Regular"
    final = sorted([r for r in rows if keep(r) and r["status"] in ORDER and ORDER[r["status"]] < 3],
                   key=lambda r: (ORDER[r["status"]], r.get("peg_y1") if np.isfinite(r.get("peg_y1", np.nan)) else 99))
    pending = [r for r in rows if keep(r) and r["status"].startswith("Breakout - bar")]

    def f(v, pct=False, d=2):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:+.1f}%" if pct else f"{v:,.{d}f}"

    def card(r):
        peg = ""
        if r.get("eps_y1") is not None and np.isfinite(r.get("eps_y1", np.nan)):
            lb = r.get("fy_labels") or ["Y1", "Y2", "Y3"]
            cells = "".join(f'<tr><td>{lb[i - 1]}</td><td>{f(r.get(f"eps_y{i}"))}</td>'
                            f'<td>{f(r.get(f"growth_y{i}"), d=1)}%</td><td>{f(r.get(f"pe_y{i}"), d=1)}</td>'
                            f'<td class="pg">{f(r.get(f"peg_y{i}"))}</td></tr>' for i in (1, 2, 3))
            peg = (f'<table class="peg"><tr><th></th><th>EPS</th><th>Growth</th><th>Fwd P/E</th><th>PEG</th></tr>'
                   f'{cells}</table><p class="src">{html.escape(str(r.get("growth_source", "")))}'
                   f' &middot; base {r.get("base_fy", "")} EPS {f(r.get("base_eps"))}</p>')
        elif r.get("peg_note"):
            peg = f'<p class="src">PEG: {html.escape(str(r["peg_note"]))}</p>'
        deals = html.escape(str(r.get("listing_deals", "-")))
        if r.get("listing_buyers"):
            deals += f'<br><span>Buy: {html.escape(r["listing_buyers"][:220])}</span>'
        if r.get("listing_sellers"):
            deals += f'<br><span>Sell: {html.escape(r["listing_sellers"][:220])}</span>'
        st = ORDER.get(r["status"], 9)
        return f"""
<article class="card" data-tf="{r['tf']}" data-st="{st}">
 <header><div><a href="https://www.tradingview.com/chart/?symbol=NSE:{html.escape(r['symbol'])}&interval={'W' if r['tf'] == 'Weekly' else 'M'}"
  target="_blank" rel="noopener">{html.escape(r['symbol'])}</a><span class="nm">{html.escape(str(r.get('name', '')))[:28]}</span></div>
  <span class="tf">{r['tf']}</span></header>
 <div class="tags"><span class="st st-{st}">{html.escape(r['status'])}</span>
  <span class="cc">Listed {r.get('listed', '')} &middot; concalls {html.escape(str(r.get('concall_status', '')))}</span></div>
 <p class="dt">Pole +{r['pole_gain_pct']:.0f}% in {r['pole_bars']} bars, flag {r['flag_bars']} bars.
  Volume {f(r['pole_vol_x'], d=1)}x near the pole, {f(r['flag_vol_x'], d=1)}x in the flag.</p>
 {chart(r)}
 <dl><div><dt>Price</dt><dd>{f(r['price'])}</dd></div><div><dt>Mcap</dt><dd>{f(r.get('mcap_cr'), d=0)} cr</dd></div>
  <div><dt>Breakout</dt><dd>{f(r['trigger'])} <i>{f(r['to_trigger_pct'], True)}</i></dd></div>
  <div><dt>Target</dt><dd>{f(r['target'])} <i>{f(r['to_target_pct'], True)}</i></dd></div>
  <div><dt>Flag low</dt><dd>{f(r['invalidation'])} <i>{f(-r['risk_pct'], True)}</i></dd></div></dl>
 {peg}
 <div class="dl"><b>Listing day</b> {deals}</div>
</article>"""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f6f2;--card:#fff;--tx:#1d1d1b;--mu:#6b6a65;--bd:#e3e1da;--ac:#1f4e78;--g:#1d7a4f;--gb:#e1f3e8;
--a:#8a5a07;--ab:#fbf0d9;--b:#185fa5;--bb:#e6f1fb;--pole:#1d7a4f;--fz:rgba(31,78,120,.10);--vb:#c9c7bf;--vs:#d85a30;--bo:#185fa5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191917;--card:#232321;--tx:#ecebe6;--mu:#a3a199;--bd:#3a3936;--ac:#85b7eb;
--g:#7fd1a5;--gb:#173b2a;--a:#efc06a;--ab:#3d2b0c;--b:#85b7eb;--bb:#132c46;--pole:#7fd1a5;--fz:rgba(133,183,235,.14);
--vb:#55534d;--vs:#f0997b;--bo:#85b7eb}}}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--tx)}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:28px 0 10px}}
.sub{{color:var(--mu);margin:0 0 14px}}.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}}
.chip{{border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:999px;padding:5px 12px;cursor:pointer;font:inherit}}
.chip.on{{border-color:var(--ac);color:var(--ac)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px}}
.card header{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
.card a{{color:var(--ac);font-weight:600;text-decoration:none;font-size:16px}}.nm{{color:var(--mu);margin-left:6px;font-size:12px}}
.tf{{font-size:12px;border:1px solid var(--bd);border-radius:6px;padding:0 7px}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px}}.st,.cc{{padding:2px 9px;border-radius:6px;font-size:12px;font-weight:600}}
.st-0{{background:var(--gb);color:var(--g)}}.st-1{{background:var(--ab);color:var(--a)}}.st-2,.st-3{{background:var(--bb);color:var(--b)}}
.cc{{border:1px solid var(--bd);color:var(--mu);font-weight:500}}.dt{{margin:2px 0 6px;color:var(--mu);font-size:13px}}
.ch{{width:100%;height:auto;display:block;margin:4px 0}}.cd{{stroke:var(--mu);stroke-width:1}}.pc{{stroke:var(--pole);stroke-width:1.4}}
.bo{{stroke:var(--bo);stroke-width:1.4}}.fz{{fill:var(--fz)}}.tg{{stroke:var(--g);stroke-dasharray:3 3;stroke-width:.9}}
.vb{{fill:var(--vb)}}.vs{{fill:var(--vs)}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;margin:6px 0}}dl div{{display:flex;justify-content:space-between;gap:6px}}
dt{{color:var(--mu);font-size:12px}}dd{{margin:0;font-variant-numeric:tabular-nums;font-size:13px}}dd i{{color:var(--mu);font-style:normal;font-size:11px}}
.peg{{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px;font-variant-numeric:tabular-nums}}
.peg th{{color:var(--mu);font-weight:500;text-align:right}}.peg td{{text-align:right;padding:1px 0;border-top:1px solid var(--bd)}}
.peg td:first-child,.peg th:first-child{{text-align:left}}.pg{{font-weight:600}}
.src{{color:var(--mu);font-size:11px;margin:4px 0}}.dl{{font-size:12px;border-top:1px solid var(--bd);margin-top:6px;padding-top:6px}}
.dl span{{color:var(--mu)}}.note{{color:var(--mu);font-size:12px;margin-top:24px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">Listed in the last 5 years &middot; bull flag on the weekly or monthly chart &middot; unusual volume near the pole AND
in the flag &middot; regular concalls. Sorted by status, then forward PEG. Green = pole, shaded = flag, blue = breakout bars, orange bars = volume spikes.</p>
<div class="chips"><button class="chip on" data-f="*">All <b>{len(final)}</b></button>
<button class="chip" data-f="0">Breakout confirmed <b>{sum(1 for r in final if ORDER[r['status']] == 0)}</b></button>
<button class="chip" data-f="1">Near breakout <b>{sum(1 for r in final if ORDER[r['status']] == 1)}</b></button>
<button class="chip" data-f="2">In flag <b>{sum(1 for r in final if ORDER[r['status']] == 2)}</b></button></div>
<section class="grid" id="final">{''.join(card(r) for r in final) or '<p>No stock passed every filter today.</p>'}</section>
<h2>Breaking out in the current bar - not confirmed until it closes ({len(pending)})</h2>
<section class="grid">{''.join(card(r) for r in pending) or '<p>None.</p>'}</section>
<p class="note">Mechanical scan; forward EPS figures are projections. Not investment advice.</p>
</main><script>
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===b));
const v=b.dataset.f;document.querySelectorAll('#final .card').forEach(k=>k.style.display=(v==='*'||k.dataset.st===v)?'':'none');}});
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Recent-IPO flag & pole scanner")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--universe")
    ap.add_argument("--years", type=float, default=5, help="listed within this many years (default 5)")
    ap.add_argument("--min-mcap", type=float, default=500)
    ap.add_argument("--max-mcap", type=float, default=None)
    ap.add_argument("--exclude-top", type=int, default=250)
    ap.add_argument("--timeframes", default="W,M")
    ap.add_argument("--weekly-spike", type=float, default=3.0)
    ap.add_argument("--monthly-spike", type=float, default=2.5)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)

    fp.TIMEFRAMES["W"].vol_spike = a.weekly_spike
    fp.TIMEFRAMES["M"].vol_spike = a.monthly_spike
    tfs = [fp.TIMEFRAMES[x.strip().upper()] for x in a.timeframes.split(",") if x.strip()]
    now = dt.datetime.now()
    today = now.date()

    cfg = es.Config(min_mcap_cr=a.min_mcap, max_mcap_cr=a.max_mcap, exclude_top=a.exclude_top)
    uni = es.build_universe(a, cfg)
    listed = listing_dates(a.refresh)
    cutoff = today - dt.timedelta(days=int(365.25 * a.years))
    uni["listed"] = uni["symbol"].map(listed)
    if not a.symbols:
        keep = uni["listed"].apply(lambda d: isinstance(d, dt.date) and d >= cutoff)
        uni = uni[keep].reset_index(drop=True)
    print(f"Recently listed small/micro caps (since {cutoff:%d %b %Y}): {len(uni)}")
    if uni.empty:
        sys.exit("No stocks in the universe.")
    prices = es.download_prices(uni["symbol"].tolist(), int(math.ceil(a.years)) + 1)
    meta = uni.set_index("symbol")[["name", "mcap_cr", "listed"]].to_dict("index")

    rows = []
    for sym, df in prices.items():
        try:
            for r in analyse(sym, df, tfs, now):
                m = meta.get(sym, {})
                r.update(name=m.get("name"), mcap_cr=m.get("mcap_cr"))
                ld = m.get("listed")
                if isinstance(ld, dt.date):
                    r["listed"] = ld.isoformat()
                    r["months_listed"] = (today.year - ld.year) * 12 + today.month - ld.month
                rows.append(r)
        except Exception as e:
            print(f"  {sym}: {e}")
    cands = [r for r in rows if r["pole_vol_ok"] or r["flag_vol_ok"]]
    print(f"Live flags: {len(rows)}; with a volume spike near the pole or in the flag: {len(cands)}; "
          f"with both: {sum(1 for r in rows if r['pole_vol_ok'] and r['flag_vol_ok'])}")

    syms = sorted({r["symbol"] for r in cands})
    comp = company_data(syms) if syms else {}
    guide = load_guidance()
    yest = yahoo_estimates([s for s in syms if s not in guide]) if syms else {}
    for r in cands:
        c = comp.get(r["symbol"], {})
        ld = meta.get(r["symbol"], {}).get("listed")
        if c.get("error"):
            r.update(concall_status="Unknown (Screener not reachable)")
        elif isinstance(ld, dt.date):
            r.update(concall_check(c.get("concalls"), ld, today))
        r.update(forward_peg(r["symbol"], r["price"], c, guide, yest))

    items = [(r["symbol"], meta[r["symbol"]]["listed"]) for r in cands
             if isinstance(meta.get(r["symbol"], {}).get("listed"), dt.date)]
    try:
        deals = listing_deals(sorted(set(items))) if items else {}
    except Exception as e:
        print(f"  listing-day deals skipped ({e})")
        deals = {}
    for r in cands:
        r.update(_deal_fields(deals.get(r["symbol"])))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = frame(rows)
    if df.empty:
        print("No live flags found.")
        return
    xlsx, page = out / f"ipo_flag_pole_{today}.xlsx", out / f"ipo_flag_pole_{today}.html"
    n = write_excel(df, rows, xlsx)
    write_html(rows, page, f"Recent IPOs: flag & pole scan - {today}")
    print(f"\nDone. Final list: {n} stocks.\n  Excel : {xlsx}\n  Report: {page}")
    fin = df[df["_both"] & df["_cc"] & df["_ok_status"]]
    if len(fin):
        print(fin[["symbol", "tf", "status", "trigger", "peg_y1", "peg_y2", "peg_y3", "listing_deals"]]
              .head(25).to_string(index=False))


if __name__ == "__main__":
    main()
