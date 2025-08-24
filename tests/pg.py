import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union, Mapping, Tuple, Literal
import sys
sys.path.append("../src")

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

df = pd.read_csv("data/aapl_eod_202303.csv")
print(df)
print(prep_index(df))
print(ohlc_index(df))