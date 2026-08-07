# Load the CBOE SPX option chain CSV into one row per option 
# (instead of one row per strike).

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Snapshot:
    spot: float
    asof: pd.Timestamp
    quotes: pd.DataFrame


def load_cboe_csv(path: str | Path) -> Snapshot:
    path = Path(path)

    with open(path) as fh:
        header = [next(fh) for _ in range(3)]

    spot = float(re.search(r"Last:\s*([\d.]+)", header[1]).group(1))
    asof = pd.to_datetime(
        re.search(r"Date:\s*(.+?) at", header[2]).group(1).strip(),
        format="%d %B %Y",
    )

    raw = pd.read_csv(path, skiprows=3)
    raw.columns = [c.strip() for c in raw.columns]

    # SPX and SPXW share expiry dates but are different contracts (AM vs PM
    # settlement). the root is the first letters of the option symbol
    root = raw["Calls"].str.strip().str.extract(r"^([A-Z]+)", expand=False)

    common = {
        "expiry": pd.to_datetime(raw["Expiration Date"].str.strip(), format="%a %b %d %Y"),
        "strike": raw["Strike"].astype(float),
        "root": root,
    }

    def side(tag: str, suffix: str) -> pd.DataFrame:
        return pd.DataFrame({
            **common,
            "cp": tag,
            "bid": raw["Bid" + suffix].astype(float),
            "ask": raw["Ask" + suffix].astype(float),
            "volume": raw["Volume" + suffix].astype(float),
            "open_interest": raw["Open Interest" + suffix].astype(float),
        })

    df = pd.concat([side("C", ""), side("P", ".1")], ignore_index=True)

    # keep one root per expiry: whichever has the open interest
    dominant = (df.groupby(["expiry", "root"])["open_interest"].sum()
                  .reset_index()
                  .sort_values("open_interest", ascending=False)
                  .drop_duplicates("expiry")[["expiry", "root"]])
    df = df.merge(dominant, on=["expiry", "root"], how="inner")

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    df["T"] = (df["expiry"] - asof).dt.days / 365.25
    df["k_spot"] = np.log(df["strike"] / spot)

    df = df.sort_values(["expiry", "cp", "strike"]).reset_index(drop=True)
    return Snapshot(spot=spot, asof=asof, quotes=df)