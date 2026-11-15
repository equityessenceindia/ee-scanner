"""Synthetic checks for the long-base breakout scanner. Run: python test_base_breakout.py"""
import datetime as dt

import numpy as np
import pandas as pd

import base_breakout_scanner as bb


def make(path_weeks, spike_week=None, seed=7):
    """path_weeks: weekly close anchors (week index, price); daily bars are interpolated."""
    rng = np.random.default_rng(seed)
    wk, px = zip(*path_weeks)
    days = np.arange(wk[-1] * 5 + 5)
    c = np.interp(days / 5.0, wk, px) * (1 + rng.normal(0, 0.006, len(days)))
    v = 1e5 * (1 + 0.3 * rng.random(len(days)))
    if spike_week is not None:
        v[spike_week * 5:(spike_week * 5) + 3] *= 8
    idx = pd.bdate_range(end="2026-11-13", periods=len(days))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": v}, index=idx)


# a 60-week base oscillating 100-124 with three tests of the top
base = [(0, 150), (20, 110), (30, 123), (40, 102), (50, 124), (60, 104), (70, 123), (79, 108)]
now = dt.datetime(2026, 11, 14, 10, 0)          # Saturday - last week complete
cases = {
    "near breakout": (make(base + [(84, 121)], spike_week=83), "Near breakout", True),
    "breakout confirmed": (make(base + [(82, 120), (83, 133), (84, 131)], spike_week=83), "Breakout confirmed", True),
    "breakout, no volume": (make(base + [(82, 120), (83, 133), (84, 131)]), "Breakout confirmed", False),
    "too extended": (make(base + [(82, 120), (83, 150), (84, 160)], spike_week=83), None, None),
    "steady uptrend": (make([(0, 50), (84, 160)]), None, None),
    "mid-base (not near)": (make(base + [(84, 108)], spike_week=83), None, None),
}
ok = 0
for name, (df, want, vol) in cases.items():
    r = bb.analyse("T", df, now)
    got = r["status"] if r else None
    good = got == want and (vol is None or r["vol_ok"] == vol)
    ok += good
    x = r or {}
    print(f"{'ok ' if good else 'BAD'} {name:22s} {got} | base {x.get('base_months')}m range {x.get('width_pct')}% "
          f"tests {x.get('tests')} res {x.get('resistance')} wk {x.get('wk_vol_x')}x day {x.get('day_vol_x')}x")
print(f"\n{ok}/{len(cases)} passed")
