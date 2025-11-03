import pandas as pd
import numpy as np
_FRACTION_MAP = {
    "½": 0.5,
    "¼": 0.25,
    "¾": 0.75,
    "⅛": 0.125,
    "⅜": 0.375,
    "⅝": 0.625,
    "⅞": 0.875,
}

def parse_price(s: str) -> float:
    if pd.isna(s):
        return float("nan")
    try:
        whole_str, frac_str = s.split("-", 1)
        whole = int(whole_str)
        ticks_part = "".join(ch for ch in frac_str if ch.isdigit())
        frac_chars = "".join(ch for ch in frac_str if not ch.isdigit())
        ticks = int(ticks_part) if ticks_part else 0
        frac = sum(_FRACTION_MAP.get(ch, 0) for ch in frac_chars)
        return whole + (ticks + frac) / 32
    except Exception:
        return float("nan")

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp['DateTime'] = pd.to_datetime(tmp['Date'])
    excel_epoch = pd.Timestamp("1899-12-30")
    tmp['Date']  = (tmp['DateTime'] - excel_epoch).dt.total_seconds() / 86400
    tmp['Price'] = tmp['Lst Trd/Lst Prxx'].map(parse_price)
    # Return Date (serial) so trading_day will find tmp['Date']
    return tmp[['Date','Price']]
    
def time(df: pd.DataFrame) -> pd.Series:
    # Convert Date (Excel serial floats) to numeric
    date = pd.to_numeric(df['Date'], errors='coerce')
    # Extract fractional days and convert to timedelta
    frac = (date % 1).round(6)
    td = pd.to_timedelta(frac, unit='D')
    # Round to nearest minute
    td_rounded = td.dt.round('min')
    # Add to a dummy midnight timestamp and extract the time component
    dummy = pd.Timestamp('2000-01-01')
    times = (dummy + td_rounded).dt.time
    return pd.Series(times, index=df.index, name='TimeOfDay')

def trading_day(df: pd.DataFrame) -> pd.Series:
    date = pd.to_numeric(df['Date'], errors='coerce')
    time = (date % 1).round(6)
    
    date_int = date.astype(int)
    trading_date_serial = np.where(time < 0.75, date_int - 1, date_int)

    excel_epoch = pd.Timestamp('1899-12-30')
    trading_date = pd.to_datetime(excel_epoch) + pd.to_timedelta(trading_date_serial, unit='D')
    
    return pd.Series(trading_date, index=df.index, name='TradingDay')

def week_day(df: pd.DataFrame, day_col: str = 'TradingDay') -> pd.Series:
    return pd.Series(df[day_col].dt.day_name(), index=df.index, name='WeekDay')

def relative_price(df: pd.DataFrame,
                   day_col: str = 'TradingDay',
                   price_col: str = 'Price') -> pd.Series:
    # Compute the opening price per trading day (first row in each group)
    day_open = df.groupby(day_col)[price_col].transform('first')
    # Subtract the open price from each price
    rel = df[price_col] - day_open
    # Return as a named Series
    return pd.Series(rel, index=df.index, name='RelPrice')

def wkday_sd(df: pd.DataFrame, day_col: str = 'WeekDay', price_col: str = 'Relative Price') -> pd.Series:
    sd = df.groupby(day_col)[price_col].transform('std')
    return pd.Series(sd, index=df.index, name='DayStdDev')

def wkday_mean(df: pd.DataFrame, day_col: str = 'WeekDay', price_col: str = 'Relative Price') -> pd.Series:
    mean = df.groupby(day_col)[price_col].transform('mean')
    return pd.Series(mean, index=df.index, name='DayStdDev')

def fill_trading_gaps(
    df: pd.DataFrame,
    date_col='Date',
    price_col='Price',
    trading_day_col='TradingDay'
) -> pd.DataFrame:
    # Input contract:
    # - Date is Excel serial float
    # - TradingDay is datetime64[ns]
    # - Price is numeric
    xls_epoch = pd.Timestamp('1899-12-30')

    df = df.copy()
    df[date_col] = pd.to_numeric(df[date_col], errors='coerce')
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[trading_day_col] = pd.to_datetime(df[trading_day_col], errors='coerce')

    # Keep one tick per timestamp
    df = df.drop_duplicates(subset=[date_col], keep='first')

    unique_days = sorted(df[trading_day_col].dropna().unique())
    all_filled = []

    for td in unique_days:
        td = pd.Timestamp(td)
        start_ts = td + pd.Timedelta(hours=18)
        end_ts = td + pd.Timedelta(days=1, hours=17)

        day = df[df[trading_day_col] == td].copy()
        if day.empty:
            continue

        # Build a real timestamp from Excel serial for indexing
        idx_ts = xls_epoch + pd.to_timedelta(day[date_col], unit='D')
        day = day.assign(_ts=idx_ts).set_index('_ts').sort_index()

        # Complete 5 minute grid
        grid = pd.date_range(start=start_ts, end=end_ts, freq='5min')
        day = day.reindex(grid)

        # Price fill. Require at least one real tick for the day
        if not day[price_col].notna().any():
            continue
        day[price_col] = day[price_col].ffill().bfill()

        # TradingDay constant over the window
        day[trading_day_col] = td

        # Recompute Excel serial Date from the index
        day[date_col] = (day.index - xls_epoch) / pd.Timedelta(days=1)

        # If TimeOfDay exists in input, recompute from the grid to avoid NaNs
        if 'TimeOfDay' in df.columns:
            day['TimeOfDay'] = (pd.Timestamp('2000-01-01') + (day.index - day.index.normalize())).time

        # For any other existing columns, forward fill then backfill to guarantee no NaNs
        # Do not touch Date or TradingDay or Price which are already set
        base_keep = {date_col, price_col, trading_day_col, 'TimeOfDay'}
        extra_cols = [c for c in df.columns if c not in base_keep and c in day.columns]
        if extra_cols:
            day[extra_cols] = day[extra_cols].ffill().bfill()

        # Preserve only columns that existed on input
        existing_cols = [c for c in df.columns if c in day.columns]
        all_filled.append(day[existing_cols])

    if not all_filled:
        return pd.DataFrame(columns=[c for c in df.columns])

    out = pd.concat(all_filled, axis=0, ignore_index=True)

    # Final guarantees
    out[trading_day_col] = pd.to_datetime(out[trading_day_col])
    out[date_col] = pd.to_numeric(out[date_col], errors='coerce')
    if price_col in out:
        out[price_col] = pd.to_numeric(out[price_col], errors='coerce').ffill().bfill()

    # As a safety net, fill any leftover gaps in non-key columns
    non_key = [c for c in out.columns if c not in {date_col, price_col, trading_day_col}]
    if non_key:
        out[non_key] = out[non_key].ffill().bfill()

    return out.sort_values(date_col).reset_index(drop=True)

def df_maker(df: pd.DataFrame) -> pd.DataFrame:
    copy = clean_data(df)
    out = pd.DataFrame()
    
    # Build columns that only depend on 'copy' first
    out['Date'] = copy['Date']
    out['Price'] = copy['Price']
    out['TradingDay'] = trading_day(copy)
    out['TimeOfDay'] = time(copy)
    #out = fill_trading_gaps(out, date_col='Date', price_col='Price', trading_day_col='TradingDay')
    out['WeekDay'] = week_day(out, day_col='TradingDay')
    
    out['Relative Price'] = relative_price(out, day_col='TradingDay', price_col='Price')
    
    out['DaySD'] = wkday_sd(out)

    out['DayMean'] = wkday_mean(out)
    
    return out
