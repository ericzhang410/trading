import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union, Mapping, Tuple, Literal

def prep_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().strip("[]").lower().replace(" ", "_") for c in out.columns]
    return out

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

def har_rv()-> pd.DataFrame:
    return out 

def id_vol_series(df: pd.DataFrame, window: int, T: int, price_col: str = "close", trading_days: int = 252, date_col: Optional[str] = None, ) -> pd.DataFrame:
    return 0

def ann_vol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df * np.sqrt(252/window)