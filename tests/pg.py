import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union, Mapping, Tuple, Literal
import sys
sys.path.append("../src")

def OHLC_prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert intraday AAPL-style quotes into a daily OHLC dataframe.

    Expected columns (any casing; brackets optional):
      [QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], [UNDERLYING_LAST], ...

    Returns
    -------
    pd.DataFrame
        Index: datetime64[ns] normalized to date (name = 'date')
        Columns: ['open', 'high', 'low', 'close']
    """
    # --- work on a copy and normalize headers ---
    out = df.copy()
    out.columns = [c.strip().strip("[]").lower().replace(" ", "_") for c in out.columns]

    # --- check required price column ---
    if "underlying_last" not in out.columns:
        raise ValueError("Missing required column 'UNDERLYING_LAST' (underlying_last after normalization).")

    # --- build a timestamp for ordering within each day ---
    ts = None
    if "quote_unixtime" in out.columns:
        ts = pd.to_datetime(out["quote_unixtime"], unit="s", errors="coerce")
    elif "quote_readtime" in out.columns:
        ts = pd.to_datetime(out["quote_readtime"], errors="coerce")
    elif "quote_date" in out.columns and "quote_time_hours" in out.columns:
        d = pd.to_datetime(out["quote_date"], errors="coerce")
        h = pd.to_numeric(out["quote_time_hours"], errors="coerce")
        ts = d + pd.to_timedelta(h.fillna(0) * 3600, unit="s")
    elif "quote_date" in out.columns:
        ts = pd.to_datetime(out["quote_date"], errors="coerce")  # midnight fallback
    else:
        raise ValueError("Need one of QUOTE_UNIXTIME, QUOTE_READTIME, or QUOTE_DATE (+ optional QUOTE_TIME_HOURS).")

    out["__ts"] = ts
    out = out.dropna(subset=["__ts"])

    # --- ensure numeric price and drop missing ---
    out["underlying_last"] = pd.to_numeric(out["underlying_last"], errors="coerce")
    out = out.dropna(subset=["underlying_last"])

    # --- derive date, sort, and aggregate to OHLC ---
    out["__date"] = out["__ts"].dt.normalize()
    out = out.sort_values(["__date", "__ts"])

    grp = out.groupby("__date")["underlying_last"]
    daily = pd.DataFrame({
        "open": grp.first(),
        "high": grp.max(),
        "low":  grp.min(),
        "close": grp.last(),
    })

    daily.index.name = "date"
    return daily

df = pd.read_csv("../data/aapl_eod_202303.csv")
print(df)