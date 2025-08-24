import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union, Mapping, Tuple, Literal

def prep_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().strip("[]").lower().replace(" ", "_") for c in out.columns]
    return out

def ohlc_index(df:pd.DataFrame) -> pd.DataFrame:
    copy = prep_index(df)
    # Build a timestamp from quote_date (+ optional quote_time_hours)
    if "quote_date" not in copy.columns:
        raise ValueError(f"No quote_date column. Got: {copy.columns.tolist()}")

    ts = pd.to_datetime(copy["quote_date"], errors="coerce")
    if "quote_time_hours" in copy.columns:
        ts = ts + pd.to_timedelta(pd.to_numeric(copy["quote_time_hours"], errors="coerce") * 3600, unit="s")
    copy["__ts"] = ts

    tmp = (
        copy.dropna(subset=["__ts", "underlying_last"])
          .assign(__ts=pd.to_datetime(copy["__ts"]))
          .sort_values("__ts")
          .set_index("__ts")
    )

    # Group into 1-day buckets and compute OHLC from the intraday 'close' series
    daily = (
        tmp.groupby(pd.Grouper(freq="1D"))["underlying_last"]
           .agg(open="first", high="max", low="min", close="last")
           .dropna(how="any")          # drop days with incomplete buckets
    )

    # Normalize index to date and move it into a 'date' column
    daily.index = daily.index.normalize()
    daily = daily.rename_axis("date").reset_index()

    return daily

def yz_vol_series(
    df: pd.DataFrame, 
    window: int,
    T: int,
    price_col: str = "close",
    trading_days: int = 252,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    h = window
    o_var = np.log(df["open"] / df["close"].shift(1)).pow(2).rolling(h, min_periods=h).sum() / (h - 1)      
    c_var = np.log(df["close"] / df["open"].shift(1)).pow(2).rolling(h, min_periods=h).sum() / (h - 1)      
    rs_var = ((np.log(df["high"]/df["close"]) * np.log(df["high"]/df["open"])) 
          + (np.log(df["low"]/df["close"])  * np.log(df["low"]/df["open"])))\
          .rolling(h, min_periods=h).sum() / (h- 1)
    k = 0.34/(1 + (h+ 1) / (h - 1))
    yz_vol = np.sqrt(o_var + k * c_var + (1-k)* rs_var) * np.sqrt(trading_days)
    yz_vol.name = f"rvol_{window}d"
    n = (T - h) + 1
    m = 1 / ( 1 - (h / n) + (h^2 -1)/(3*n^2))
    print(yz_vol)
    return yz_vol

def gkyz_vol_series(
    df: pd.DataFrame, # I need high,low, 
    window: int,
    price_col: str = "close",
    trading_days: int = 252,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    n = window
    oc_var = np.log(df["open"] / df["close"].shift(1)).pow(2).rolling(n, min_periods=n).sum() / (2*n)    
    hl_var = np.log(df["high"] / df["low"]).pow(2).rolling(n, min_periods=n).sum() / (2*n)   
    co_var = np.log(df["close"]/df["low"]).pow(2).rolling(n, min_periods=n).sum() * (2 * np.log(2) - 1) / (n - 1)
    gkyz_vol = np.sqrt(oc_var + hl_var + co_var)
    gkyz_vol.name = f"realized_vol_{window}d"
    return gkyz_vol

def r_vol_series(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
    trading_days: int = 252,
    date_col: Optional[str] = None,
) -> pd.Series:
    df = prep_index(df)
    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    r = np.log(prices).diff()
    r_vol = 100 * np.sqrt(trading_days / window * r.pow(2).rolling(window, min_periods=window).sum())
    r_vol.name = f"realized_vol_{window}d"
    return r_vol

def five_n_sum(
    df: pd.DataFrame,
    windows: Iterable[int],
    T: int,
    price_col: str = "close",
    trading_days: int = 252,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    df = prep_index(df)
    out = []
    for w in windows:
        r_vol_ser = yz_vol_series(df, price_col=price_col, window=w, trading_days=trading_days, T=T)
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

def plot_realized_vol_cone(
    summary: pd.DataFrame,
    title: str = "Volatility vs Days to Expiration",
    iv_curve: Optional[pd.Series] = None,
):
    import matplotlib.pyplot as plt
    x = summary["horizon_days"].to_numpy()

    fig, ax = plt.subplots(figsize=(9,5))

    # Distinct lines for summary statistics
    ax.plot(x, summary["min"],     color="#000000", linewidth=1.5, marker="o", markersize=0, label="Low")
    ax.plot(x, summary["q25"],     color="#fc8d59", linewidth=1.5, marker="o", markersize=0, label="q(0.25)")
    ax.plot(x, summary["median"],  color="#4575b4", linewidth=2.0, marker="o", markersize=0, label="Median")
    ax.plot(x, summary["q75"],     color="#fc8d59", linewidth=1.5, marker="o", markersize=0, label="q(0.75)")
    ax.plot(x, summary["max"],     color="#000000", linewidth=1.5, marker="o", markersize=0, label="Hi")

    # Optional IV overlay
    if iv_curve is not None:
        iv_aligned = iv_curve.reindex(summary["horizon_days"]).interpolate().to_numpy()
        ax.plot(x, iv_aligned, color="purple", linewidth=2.0, marker="o", markersize=0, label="Implied Volatility")

    ax.set_xlabel("Days to Expiration)")
    ax.set_ylabel("Volatility")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return ax

def iv_curve_from_snapshot(
    df: pd.DataFrame,
) -> pd.DataFrame:
    cy = prep_index(df).copy()
    # Coerce to numeric
    cy["dte"]  = pd.to_numeric(cy["dte"],  errors="coerce")
    cy["c_iv"] = pd.to_numeric(cy["c_iv"], errors="coerce")
    cy["p_iv"] = pd.to_numeric(cy["p_iv"], errors="coerce")

    # Average call/put IVs, assuming inputs like 23.4 (%)
    avg_iv = (cy["c_iv"] / 100 + cy["p_iv"] / 100) / 2

    # Drop bad rows
    out = pd.DataFrame({"dte": cy["dte"], "iv": avg_iv}).dropna()

    # Round DTE to whole days and group (take median across strikes)
    out["dte"] = out["dte"].round().astype(int)
    curve = out.groupby("dte")["iv"].median().sort_index()

    # Convert to % for plotting if your cone is in percent
    curve = 100 * curve
    curve.index.name = "days_to_expiration"
    curve.name = "implied_vol"
    return curve
