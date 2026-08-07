# Turn cleaned option mids into implied vols.

# Black-76 prices an option from teh variables: F, K, T, D, sigma. We have the first four 
# F and D measured per expiry from parity, K and T from the quote, so each mid
# pins down its sigma: the value that makes the formula reproduce the price.
# The map sigma -> price is strictly increasing, and as the root is unique
# brentq (bracketed root-finding) finds it fast.

# Output is the table the rest of the project runs on:
# one row per surviving quote with k = ln(K/F) and w = sigma^2 * T.

import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from chain import load_cboe_csv
from filter import apply_quote_filters, apply_shape_filters
from forwards import measure_forwards

SIGMA_LO = 1e-4   # bracket for the root search: 0.01% vol. Black-76 would be undefined for sigma=0
SIGMA_HI = 5.0    # to 500% vol. We assume anything above this is a broken quote


def black76_price(F, K, T, D, sigma, cp):
    #the pricing formula. N is the standard normal cdf
    s = sigma * np.sqrt(T)                      # sqrt of total variance w
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    if cp == "C":
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(mid, F, K, T, D, cp):
    # the sigma that makes black76_price equal the observed mid.
    # returns nan if no sigma in [SIGMA_LO, SIGMA_HI] can reach the mid
    # (a below-intrinsic or absurdly high quote) -- caller counts those.
    def gap(sigma):
        return black76_price(F, K, T, D, sigma, cp) - mid

    lo, hi = gap(SIGMA_LO), gap(SIGMA_HI)
    if lo * hi > 0:                # bracket doesn't straddle the root
        return np.nan
    return brentq(gap, SIGMA_LO, SIGMA_HI, xtol=1e-8)


def invert_chain(q, fwd):
    # survivors of filter.py + (F, D) of forwards.py -> implied vol table.
    # one row per quote: expiry, T, cp, strike, k, mid, sigma, w.
    iv = q[q["reason"] == ""].merge(fwd[["expiry", "F", "D"]], on="expiry")

    iv["k"] = np.log(iv["strike"] / iv["F"])    # moneyness against the forward
    iv["sigma"] = [
        implied_vol(r.mid, r.F, r.strike, r.T, r.D, r.cp)
        for r in iv.itertuples()
    ]
    iv["w"] = iv["sigma"] ** 2 * iv["T"]

    return iv[["expiry", "T", "cp", "strike", "k", "mid", "sigma", "w"]]


def twin_check(q_raw, fwd, n_strikes=5):
    # parity says the call and the put at one strike imply the same vol.
    # invert both legs at a few near-the-money strikes of a mid-dated expiry
    # and show the gap -- theory says ~0, a systematic gap means F is off
    f = fwd.iloc[(fwd["T"] - 0.25).abs().argmin()]     # expiry nearest 3 months
    sm = q_raw[q_raw["expiry"] == f["expiry"]]
    cs = sm[(sm.cp == "C") & (sm.bid > 0)][["strike", "mid"]]
    ps = sm[(sm.cp == "P") & (sm.bid > 0)][["strike", "mid"]]
    pairs = cs.merge(ps, on="strike", suffixes=("_c", "_p"))
    pairs = pairs.iloc[(pairs["strike"] - f["F"]).abs().argsort()[:n_strikes]]

    rows = []
    for r in pairs.itertuples():
        sc = implied_vol(r.mid_c, f["F"], r.strike, f["T"], f["D"], "C")
        sp = implied_vol(r.mid_p, f["F"], r.strike, f["T"], f["D"], "P")
        rows.append({"strike": r.strike, "sigma_call": sc, "sigma_put": sp,
                     "gap_bp": (sc - sp) * 1e4})
    return pd.DataFrame(rows).sort_values("strike")


if __name__ == "__main__":
    snap = load_cboe_csv(sys.argv[1])

    q = snap.quotes.copy()                 # copy: filtering must not touch the
    q["reason"] = ""                       # raw table, forwards still needs it
    apply_quote_filters(q)
    apply_shape_filters(q)

    fwd = measure_forwards(snap.quotes)    # raw chain: parity needs ITM legs
    iv = invert_chain(q, fwd)

    n, bad = len(iv), int(iv["sigma"].isna().sum())
    print(f"{n:,} quotes inverted, {bad} with no root in "
          f"[{SIGMA_LO}, {SIGMA_HI}]\n")

    qs = iv["sigma"].quantile([0, .05, .5, .95, 1]).round(4)
    print("sigma quantiles (min / 5% / median / 95% / max)")
    print("  " + "  ".join(f"{v:.1%}" for v in qs))

    print("\ntwin check, expiry nearest 3 months (gap in vol basis points):")
    print(twin_check(snap.quotes, fwd).round(4).to_string(index=False))