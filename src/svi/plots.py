# The first pictures of the data are going to be: implied vol smiles, one colour per expiry.

# Each dot is one surviving quote at its (k, sigma). No curves yet, drawing
# the curve through each expiry's dots is exactly the job of the SVI fit.

import sys

import matplotlib

matplotlib.use("Agg")           # write files, no window needed
import matplotlib.pyplot as plt

from chain import load_cboe_csv
from filter import apply_quote_filters, apply_shape_filters
from forwards import measure_forwards
from black76 import invert_chain

# maturities worth looking at: ~2w, 1m, 3m, 6m, 1y, 2y, 3.5y
TARGET_T = [0.04, 0.08, 0.25, 0.5, 1.0, 2.0, 3.5]


def pick_expiries(fwd, targets=TARGET_T):
    # the fwd rows nearest each target maturity, without duplicates
    idx = {(fwd["T"] - t).abs().idxmin() for t in targets}
    return fwd.loc[sorted(idx)]


def _style(ax, subtitle):
    # shared look: no box, light horizontal grid, small outward ticks
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="out", length=3, labelsize=9)
    ax.axvline(0, linewidth=0.6, color="0.6")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9, color="0.4")


def _draw(ax, iv, expiries, ycol, yscale):
    cmap = plt.get_cmap("viridis")
    tmax = expiries["T"].max()
    for _, f in expiries.iterrows():
        sm = iv[iv["expiry"] == f["expiry"]].sort_values("k")
        ax.plot(sm["k"], sm[ycol] * yscale, ".", markersize=4.5, alpha=0.85,
                color=cmap(0.85 * f["T"] / tmax),
                label=f"{f['expiry'].date()}   {f['T']:.2f}y")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5,
              title="expiry", title_fontsize=9)


def plot_smiles(iv, expiries, path, subtitle="", klim=(-1.2, 0.5)):
    # sigma against k. short maturities dark, long ones light. klim trims
    # the sparse far wings for readability
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    _draw(ax, iv, expiries, "sigma", 100)
    _style(ax, subtitle)
    ax.set_xlim(*klim)
    ax.set_xlabel("log-moneyness  k = ln(K/F)", fontsize=10)
    ax.set_ylabel("implied vol  (%)", fontsize=10)
    ax.set_title("SPX implied volatility smiles", fontsize=13,
                 loc="left", pad=22, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_total_variance(iv, expiries, path, subtitle="", klim=(-1.2, 0.5)):
    # w = sigma^2 T against k: the plane SVI fits in. preview of section 5:
    # no calendar arbitrage means longer-maturity w curves sit above shorter
    # ones -- they should never cross
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    _draw(ax, iv, expiries, "w", 1)
    _style(ax, subtitle)
    ax.set_xlim(*klim)
    ax.set_xlabel("log-moneyness  k = ln(K/F)", fontsize=10)
    ax.set_ylabel("total variance  w = sigma$^2$T", fontsize=10)
    ax.set_title("SPX total variance", fontsize=13,
                 loc="left", pad=22, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    outdir = sys.argv[2] if len(sys.argv) > 2 else "figures"

    snap = load_cboe_csv(sys.argv[1])
    q = snap.quotes.copy()
    q["reason"] = ""
    apply_quote_filters(q)
    apply_shape_filters(q)

    fwd = measure_forwards(snap.quotes)
    iv = invert_chain(q, fwd)
    ex = pick_expiries(fwd)

    sub = f"CBOE snapshot {snap.asof.date()}  ·  {len(iv):,} quotes  ·  dots are quotes, not fits"
    plot_smiles(iv, ex, f"{outdir}/smiles.png", sub)
    plot_total_variance(iv, ex, f"{outdir}/total_variance.png", sub)
    print(f"wrote {outdir}/smiles.png and {outdir}/total_variance.png "
          f"({len(ex)} expiries shown, {len(iv):,} quotes behind them)")