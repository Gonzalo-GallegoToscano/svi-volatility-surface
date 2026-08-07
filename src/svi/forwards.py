# Measure the forward F and discount factor D per expiry from put-call parity.

# C - P = D*(F - K) holds at every strike of an expiry, with one D and one F for
# the whole expiry. So plotting y = C - P against K gives a straight line:
# slope = -D, intercept = D*F. One OLS fit per expiry recovers both, no rate or
# dividend model needed.

import sys

import numpy as np
import pandas as pd

from chain import load_cboe_csv
from filter import MAX_REL_SPREAD, MIN_MATURITY_DAYS

MIN_PAIRS = 5           # we won't trust a line fitted through fewer points


def forward_from_parity(K, call_mid, put_mid):
    # we recover (F, D) for one expiry from C - P = D*(F - K).
    # the caller passes only strikes where both legs are live.
    y = call_mid - put_mid
    slope, intercept = np.polyfit(K, y, 1)   # polyfit returns slope first
    D = -slope
    F = intercept / D
    return F, D


def good_pairs(q, expiry):
    # one row per strike with the call and put side by side, kept only where
    # both legs are alive: bid > 0 and relative spread <= 25% on each. works on
    # the full chain and ignores the reason column -- every pair has one ITM leg
    # by construction, that filter was for a different job.
    cols = ["strike", "mid", "bid", "spread"]
    cs = q[(q.expiry == expiry) & (q.cp == "C")][cols]
    ps = q[(q.expiry == expiry) & (q.cp == "P")][cols]
    pairs = cs.merge(ps, on="strike", suffixes=("_c", "_p"))

    keep = ((pairs.bid_c > 0) & (pairs.bid_p > 0)
            & (pairs.spread_c / pairs.mid_c <= MAX_REL_SPREAD)
            & (pairs.spread_p / pairs.mid_p <= MAX_REL_SPREAD))
    return pairs[keep]


def measure_forwards(q):
    # one regression per expiry. returns a table with one row each:
    # expiry, T, how many pairs voted, D, F, and the implied rate r as a
    # sanity check (r = -ln(D)/T, continuously compounded).
    ex = (q.loc[q["T"] > MIN_MATURITY_DAYS / 365.25, ["expiry", "T"]]
            .drop_duplicates()
            .sort_values("expiry"))

    rows = []
    for expiry, T in ex.itertuples(index=False):
        p = good_pairs(q, expiry)
        if len(p) < MIN_PAIRS:
            continue
        F, D = forward_from_parity(p["strike"].values,
                                   p["mid_c"].values,
                                   p["mid_p"].values)
        rows.append({"expiry": expiry, "T": T, "n_pairs": len(p),
                     "D": D, "F": F, "r": -np.log(D) / T})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    snap = load_cboe_csv(sys.argv[1])
    fwd = measure_forwards(snap.quotes)

    print(f"spot (header, pre-market so possibly stale) {snap.spot:,.2f}   "
          f"asof {snap.asof.date()}   {len(fwd)} expiries measured\n")
    out = fwd.assign(expiry=fwd["expiry"].dt.date).round(
        {"T": 3, "D": 4, "F": 2, "r": 4})
    with pd.option_context("display.max_rows", None):
        print(out.to_string(index=False))