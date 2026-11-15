"""Synthetic checks for the flag & pole scanner. Run: python test_flag_pole.py"""
import datetime as dt

import numpy as np
import pandas as pd

import flag_pole_scanner as fp


def daily(anchors, vol_spikes=(), seed=3, noise=0.004):
    """anchors: (business-day index, price). Daily OHLCV with small noise."""
    rng = np.random.default_rng(seed)
    b, p = zip(*anchors)
    x = np.arange(b[-1] + 1)
    c = np.interp(x, b, p) * (1 + rng.normal(0, noise, len(x)))
    v = 1e5 * (1 + 0.3 * rng.random(len(x)))
    for i, m in vol_spikes:
        v[i] *= m
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": v},
                        index=pd.bdate_range("2021-01-04", periods=len(x)))


W = fp.TIMEFRAMES["W"]
cases = {
    # 2y base, pole +60% in ~8 weeks, flag drifting down ~25% of pole for 10 weeks, spike at top
    "weekly flag + spike": (daily([(0, 100), (500, 100), (540, 160), (590, 145)],
                                  vol_spikes=[(538, 12)]), "In flag/Near", True),
    "weekly flag no spike": (daily([(0, 100), (500, 100), (540, 160), (590, 145)]),
                             "In flag/Near", False),
    "breakout this week": (daily([(0, 100), (500, 100), (540, 160), (585, 147), (590, 166)],
                                 vol_spikes=[(560, 15)]), "Breaking", True),
    "deep pullback (no flag)": (daily([(0, 100), (500, 100), (540, 160), (590, 110)]), None, None),
    "slow rise (no pole)": (daily([(0, 100), (590, 160)]), None, None),
}
ok = 0
for name, (df, want, vol) in cases.items():
    rows = [r for r in fp.analyse("T", df, [W])]
    got = rows[0]["status"] if rows else None
    good = (want is None and got is None) or (got is not None and want is not None and
                                              any(w in got for w in want.split("/")))
    if good and rows and vol is not None:
        good = rows[0]["vol_flag"] == vol
    ok += good
    extra = (f"pole +{rows[0]['pole_gain_pct']}% {rows[0]['pole_bars']}w, flag {rows[0]['flag_bars']}w "
             f"retr {rows[0]['retrace_pct']}%, vol {rows[0]['vol_x']}x {rows[0]['vol_where']}, "
             f"trig {rows[0]['trigger']} tgt {rows[0]['target']}" if rows else "")
    print(f"{'ok ' if good else 'BAD'} {name:26s} got {got} {extra}")

# concall parsing
page = """<section id="documents"><div class="documents concalls flex-column">
<h3 class="margin-bottom-8">Concalls</h3><ul class="list-links">
<li><div class="ink-600">Nov 2026</div><a>Transcript</a></li>
<li><div class="ink-600">Aug 2026</div><a>Transcript</a><a>PPT</a></li>
<li><div class="ink-600">May 2026</div></li><li><div class="ink-600">Feb 2026</div></li>
<li><div class="ink-600">Nov 2025</div></li><li><div class="ink-600">Aug 2023</div></li>
</ul></div></section><div>Mar 2019 other text</div>"""
m = fp.parse_concalls(page)
v = fp.concall_verdict(m, dt.date(2026, 11, 12))
print(m, v)
ok += v["concall_status"] == "Regular" and v["concalls_24m"] == 5
v2 = fp.concall_verdict(["2025-02", "2024-11", "2024-08", "2024-05"], dt.date(2026, 11, 12))
v3 = fp.concall_verdict(None, dt.date(2026, 11, 12))
print(v2, v3)
ok += v2["concall_status"].startswith("Stopped") and v3["concall_status"] == "No concall section"
print(f"\n{ok}/{len(cases) + 2} checks passed")
