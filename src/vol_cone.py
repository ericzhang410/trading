import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import re
from typing import Iterable, Optional, Union, Mapping, Tuple, Literal

def prep_index(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    def _norm(col: str) -> str:
        # remove surrounding brackets, lowercase, strip, replace spaces with underscore
        c = re.sub(r"^\[|\]$", "", str(col).strip())
        c = c.lower().strip().replace(" ", "_")
        return c
    out.columns = [_norm(c) for c in out.columns]


def yz_vol_series(
    df: pd.DataFrame, # I need high,low, 
    n: int,
    price_col: str = "close",
    trading_days: int = 252,
    date_col: Optional[str] = None,
) -> pd.series:
    i_sigma =
    o_sigma = (1/(n-1)) * np.log(i_sigma/c).rolling(n,min_periods = n).sum()
    c_sigma
    rs_sigma
    r = np.sqrt()
    yz_vol = 0
    return yz_vol
def r_vol_series(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
    trading_days: int = 252,
    date_col: Optional[str] = None,
) -> pd.Series:
    df = prep_index(df, date_col)
    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    r = np.log(prices).diff()
    r_vol = 100 * np.sqrt(trading_days / window * r.pow(2).rolling(window, min_periods=window).sum())
    r_vol.name = f"realized_vol_{window}d"
    return r_vol

def summarize_realized_vol(
    df: pd.DataFrame,
    windows: Iterable[int],
    price_col: str = "close",
    trading_days: int = 252,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    df = prep_index(df, date_col)
    out = []
    for w in windows:
        r_vol_ser = r_vol_series(df, price_col=price_col, window=w, trading_days=trading_days)
        r_vol_ser = r_vol_ser.dropna()
        if r_vol_ser.empty:
            out.append({"horizon_days": w, "min": np.nan, "q25": np.nan, "median": np.nan,
                        "q75": np.nan, "max": np.nan, "n": 0})
            continue
        out.append({
            "horizon_days": w,
            "min": float(r_vol_ser.min()),
            "q25": float(r_vol_ser.quantile(0.25)),
            "median": float(r_vol_ser.quantile(0.50)),
            "q75": float(r_vol_ser.quantile(0.75)),
            "max": float(r_vol_ser.max()),
            "n": int(r_vol_ser.size),
        })
    summary = pd.DataFrame(out).sort_values("horizon_days").reset_index(drop=True)
    return summary

def plot_realized_vol_summary(
    summary: pd.DataFrame,
    title: str = "Realized Volatility vs Horizon",
    iv_curve: Optional[pd.Series] = None,
):
    import matplotlib.pyplot as plt
    x = summary["horizon_days"].to_numpy()

    fig, ax = plt.subplots(figsize=(9,5))

    # Distinct lines for summary statistics
    ax.plot(x, summary["min"],     color="#1a9850", linewidth=1.5, marker="o", markersize=3, label="Low")
    ax.plot(x, summary["q25"],     color="#91bfdb", linewidth=1.5, marker="o", markersize=3, label="q(0.25)")
    ax.plot(x, summary["median"],  color="#4575b4", linewidth=2.0, marker="o", markersize=3, label="Median")
    ax.plot(x, summary["q75"],     color="#fc8d59", linewidth=1.5, marker="o", markersize=3, label="q(0.75)")
    ax.plot(x, summary["max"],     color="#d73027", linewidth=1.5, marker="o", markersize=3, label="Hi")

    # Optional IV overlay
    if iv_curve is not None:
        iv_aligned = iv_curve.reindex(summary["horizon_days"]).to_numpy()
        ax.plot(x, iv_aligned, color="purple", linewidth=2.0, marker="o", markersize=3, label="Implied (snapshot)")

    ax.set_xlabel("Window size in days")
    ax.set_ylabel("Volatility (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return ax



def iv_curve_from_snapshot(
    snapshot: pd.DataFrame,
    windows: Iterable[int],
    dte_col: str = "days_to_expiration",
    iv_col: str = "implied_vol",
    iv_in_pct: bool = False,          # set True if your IV column is already in %
    how: str = "nearest",             # "nearest" or "interp"
) -> pd.Series:
    """
    Build an IV term-curve aligned to the requested 'windows' in days.
    Returns a pandas Series indexed by window (days) with IV in PERCENT (%).
    """
    snap = snapshot.copy()
    snap = snap[[dte_col, iv_col]].dropna()
    snap[dte_col] = pd.to_numeric(snap[dte_col], errors="coerce")
    snap[iv_col] = pd.to_numeric(snap[iv_col], errors="coerce")
    snap = snap.dropna().sort_values(dte_col)
    if snap.empty:
        return pd.Series(index=list(windows), dtype=float, name="implied_vol")

    # convert to %
    if iv_in_pct:
        snap_iv_pct = snap[iv_col].to_numpy()
    else:
        snap_iv_pct = 100.0 * snap[iv_col].to_numpy()

    snap_dte = snap[dte_col].to_numpy()

    import numpy as np
    windows = np.array(list(windows), dtype=float)

    if how == "interp" and len(snap_dte) >= 2:
        # piecewise linear interpolation (extrapolate flat beyond ends)
        iv_interp = np.interp(windows, snap_dte, snap_iv_pct, left=snap_iv_pct[0], right=snap_iv_pct[-1])
        return pd.Series(iv_interp, index=windows.astype(int), name="implied_vol")

    # default: nearest‑neighbour pick
    idx = np.abs(windows[:, None] - snap_dte[None, :]).argmin(axis=1)
    iv_nn = snap_iv_pct[idx]
    return pd.Series(iv_nn, index=windows.astype(int), name="implied_vol")
