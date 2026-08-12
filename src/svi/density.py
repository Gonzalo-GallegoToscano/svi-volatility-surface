# We read the market's probability distribution off the surface.
#
# Breeden-Litzenberger: the second derivative of call prices in strike is
# the (discounted) density of S_T. For an SVI smile in (k, w) coordinates
# that density is q(k) = g(k) / sqrt(2*pi*w) * exp(-d2^2 / 2) with
# d2 = -k/sqrt(w) - sqrt(w)/2, the same g(k) we used for the butterfly
# check. The arbitrage police and the forecast distribution are one
# object: g >= 0 IS "probabilities are non-negative".
#
# Output: the implied density of the index level at a few horizons.

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from chain import load_cboe_csv
from filter import apply_quote_filters, apply_shape_filters
from forwards import measure_forwards
from black76 import invert_chain
from fit import fit_all
from model import svi_derivs
from surface import g_function, repair_calendar
from plots import _style, style_of

TARGET_T = [0.08, 0.25, 0.49, 1.05]     # ~1m, 3m, 6m, 1y


def implied_density_k(k, a, b, rho, m, s):
    # density of log-moneyness k = ln(S_T / F), per unit of k
    w = svi_derivs(k, a, b, rho, m, s)[0]
    g = g_function(k, a, b, rho, m, s)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    return g / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * d2 * d2)


if __name__ == "__main__":
    snap = load_cboe_csv(sys.argv[1])
    q = snap.quotes.copy()
    q["reason"] = ""
    apply_quote_filters(q)
    apply_shape_filters(q)
    fwd = measure_forwards(snap.quotes)
    iv = invert_chain(q, fwd)
    ranges = {e: (g["k"].min(), g["k"].max()) for e, g in iv.groupby("expiry")}
    fits, _ = repair_calendar(fit_all(iv), ranges)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i, t in enumerate(TARGET_T):
        p = fits.iloc[(fits["T"] - t).abs().argmin()]
        F = float(fwd.loc[fwd["expiry"] == p["expiry"], "F"].iloc[0])
        lo, hi = ranges[p["expiry"]]
        k = np.linspace(lo, hi, 600)
        qk = implied_density_k(k, p["a"], p["b"], p["rho"], p["m"], p["s"])
        # change of variable to index level: S = F*e^k, q_S = q_k / S
        S = F * np.exp(k)
        c = style_of(i)
        ax.plot(S, qk / S * 1000, "-", lw=1.8, color=c,
                label=f"{p['expiry'].date()}   {p['T']:.2f}y")
        mass = np.trapezoid(qk, k)
        print(f"T={p['T']:.2f}  density mass on quoted range: {mass:.3f}")

    ax.axvline(float(fwd["F"].iloc[0]), lw=0.6, color="0.6")
    _style(ax, f"CBOE snapshot {snap.asof.date()}  ·  read off the "
               f"arbitrage-free SVI surface via Breeden-Litzenberger")
    ax.set_xlim(3500, 9500)
    ax.set_xlabel("SPX level at expiry", fontsize=10)
    ax.set_ylabel("implied density  (x1000)", fontsize=10)
    ax.set_title("Market-implied distribution of the SPX", fontsize=13,
                 loc="left", pad=22, weight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5,
              title="expiry", title_fontsize=9)
    fig.tight_layout()
    outdir = sys.argv[2] if len(sys.argv) > 2 else "figures"
    fig.savefig(f"{outdir}/implied_density.png", dpi=150)
    print(f"wrote {outdir}/implied_density.png")