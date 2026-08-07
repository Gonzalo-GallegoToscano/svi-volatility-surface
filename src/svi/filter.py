# It flags bad quotes instead of deleting them. The reason column tells us what 
# got removed and why, and the counts become the rejection log.
# We do it this way to avoid losing the context of the quote not used, and to be able to analyse the resulted filtered data.

# Order matters: keep-OTM-only has to run before the arbitrage checks, because
# deep ITM prices are basically all intrinsic and fail convexity on noise we
# were going to throw away anyway.

import sys

import numpy as np

from chain import load_cboe_csv

MIN_MATURITY_DAYS = 7    # under a week there is almost no variance left to measure
MAX_REL_SPREAD = 0.25    # spread / mid (mid, not bid: bid can be ~0)


def mark(q, mask, why):
    # only label rows with no reason yet -> the first filter to catch a quote
    # owns it, so the counts add up to the total
    q.loc[(q["reason"] == "") & mask, "reason"] = why


def apply_quote_filters(q):
    # each quote judged on its own, no neighbours needed

    # expiring within a week: w = sigma^2 T is tiny, price is mostly noise
    mark(q, q["T"] <= MIN_MATURITY_DAYS / 365.25, "maturity < 7d")

    # ITM = call struck below spot / put struck above. the OTM twin at the
    # same strike has the same vol info (parity) with a cleaner price
    itm = ((q["cp"] == "C") & (q["k_spot"] < 0)) | \
          ((q["cp"] == "P") & (q["k_spot"] > 0))
    mark(q, itm, "in-the-money")

    # no committed buyer, the mid is made up
    mark(q, q["bid"] <= 0, "zero bid")

    # too wide: nobody agrees on the value
    rel_spread = q["spread"] / q["mid"].replace(0, np.nan)
    mark(q, rel_spread > MAX_REL_SPREAD, f"spread > {MAX_REL_SPREAD:.0%}")


def monotonicity_flags(K, price, D=1.0, cp="C"):
    # calls must fall in strike, puts must rise. the slope is D * P(finish
    # past K), a probability, so it stays in [-D, 0] for calls and [0, D] for
    # puts. D=1 until we measure it in 3.3. a bad slope is a pair and we can't
    # tell which quote is the stale one, so flag both.
    K, price = np.asarray(K, float), np.asarray(price, float)
    if len(K) < 2:
        return np.zeros(len(K), bool)
    slope = np.diff(price) / np.diff(K)
    if cp == "C":
        bad = (slope > 1e-10) | (slope < -D - 1e-10)
    else:
        bad = (slope < -1e-10) | (slope > D + 1e-10)
    flags = np.zeros(len(K), bool)
    flags[:-1] |= bad
    flags[1:] |= bad
    return flags


def convexity_flags(K, price):
    # middle of 3 consecutive strikes above the chord of its neighbours =
    # butterfly at a negative price, so a bad quote. lam handles uneven strike
    # spacing (0.5 when even). 1e-10 is float tolerance.
    K, price = np.asarray(K, float), np.asarray(price, float)
    if len(K) < 3:
        return np.zeros(len(K), bool)
    lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    chord = lam * price[:-2] + (1.0 - lam) * price[2:]
    return np.r_[False, price[1:-1] > chord + 1e-10, False]


def apply_shape_filters(q, D=1.0):
    # these compare neighbouring strikes, so first group into one smile
    # (one expiry, one side) sorted by strike. g.index[flags] maps each
    # group's verdicts back to the right rows of q. a quote failing both
    # checks ends up labelled convexity (second write wins)
    for _, g in q[q["reason"] == ""].groupby(["expiry", "cp"], sort=False):
        g = g.sort_values("strike")
        K, price, cp = g["strike"].values, g["mid"].values, g["cp"].iloc[0]
        q.loc[g.index[monotonicity_flags(K, price, D, cp)], "reason"] = "monotonicity"
        q.loc[g.index[convexity_flags(K, price)], "reason"] = "convexity"


def rejection_log(q):
    # counts per reason, for the README
    n = len(q)
    counts = q["reason"].value_counts()
    counts = counts[counts.index != ""]
    out = counts.rename("removed").to_frame()
    out["pct"] = (out["removed"] / n * 100).round(1)
    kept = int((q["reason"] == "").sum())
    out.loc["KEPT"] = [kept, round(kept / n * 100, 1)]
    return out


if __name__ == "__main__":
    snap = load_cboe_csv(sys.argv[1])
    q = snap.quotes.copy()
    q["reason"] = ""

    apply_quote_filters(q)
    apply_shape_filters(q)

    log = rejection_log(q)
    assert log["removed"].sum() == len(q), "labels do not partition the data"

    good = q[q["reason"] == ""]
    per_expiry = good.groupby("expiry").size()

    print(f"spot {snap.spot:,.2f}   asof {snap.asof.date()}   {len(q):,} quotes in\n")
    print(log.to_string())
    print(f"\nusable expiries {per_expiry.size}   "
          f"median quotes/expiry {int(per_expiry.median())}   "
          f"min {per_expiry.min()}")