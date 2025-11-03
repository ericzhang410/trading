from fastapi import FastAPI, Query, Path, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import json
from typing import Optional




# Make rel_data importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, 'src')
sys.path.insert(0, SRC_DIR)
from rel_data import df_maker




# Initialize FastAPI
app = FastAPI(title="Bond Futures Dashboard")




# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(THIS_DIR, "static")), name="static")




# ============================================================================
# MULTI-TICKER DATA LOADING
# ============================================================================



AVAILABLE_TICKERS = ['TUZ5', 'FVZ5', 'TYZ5']
ticker_data = {}



def load_all_tickers():
    """Load all ticker data on startup"""
    global ticker_data
    
    for ticker in AVAILABLE_TICKERS:
        try:
            # Try to load from file
            file_path = os.path.join(THIS_DIR, 'data', f'{ticker.lower()}.csv')
            
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
            
            print(f"Loading {ticker} from {file_path}...")
            data = pd.read_csv(file_path)
            df = df_maker(data)
            
            # Build shifted datetime for plotting
            df['TimeDT'] = pd.to_datetime(
                '2000-01-01 ' + df['TimeOfDay'].astype(str),
                format='%Y-%m-%d %H:%M:%S'
            )
            cutoff = pd.to_datetime('18:00:00', format='%H:%M:%S').time()
            df['TimePlot'] = df['TimeDT'].apply(
                lambda ts: ts + pd.Timedelta(days=1) if ts.time() < cutoff else ts
            )
            
            # Label columns
            df['DayStr'] = df['TradingDay'].dt.strftime('%Y-%m-%d')
            df['WeekDay'] = df['TradingDay'].dt.day_name()
            df['DayLabel'] = df['DayStr'] + ' - ' + df['WeekDay']
            
            ticker_data[ticker] = df
            print(f"✓ {ticker}: {len(df)} records, {df['DayStr'].nunique()} trading days")
            
        except Exception as e:
            print(f"✗ Error loading {ticker}: {e}")




# Load all tickers on startup
load_all_tickers()



# If no tickers loaded, load default TUZ5 data for backward compatibility
if not ticker_data:
    print("⚠️  No ticker data loaded, trying default TUZ5.csv...")
    try:
        data = pd.read_csv(os.path.join(THIS_DIR, 'data', 'TUZ5.csv'))
        df = df_maker(data)
        
        df['TimeDT'] = pd.to_datetime(
            '2000-01-01 ' + df['TimeOfDay'].astype(str),
            format='%Y-%m-%d %H:%M:%S'
        )
        cutoff = pd.to_datetime('18:00:00', format='%H:%M:%S').time()
        df['TimePlot'] = df['TimeDT'].apply(
            lambda ts: ts + pd.Timedelta(days=1) if ts.time() < cutoff else ts
        )
        
        df['DayStr'] = df['TradingDay'].dt.strftime('%Y-%m-%d')
        df['WeekDay'] = df['TradingDay'].dt.day_name()
        df['DayLabel'] = df['DayStr'] + ' - ' + df['WeekDay']
        
        ticker_data['TUZ5'] = df
        print(f"✓ Loaded default TUZ5 data: {len(df)} records")
    except Exception as e:
        print(f"✗ Critical error loading default data: {e}")




# ============================================================================
# UTILITY FUNCTIONS - NaN Handling
# ============================================================================



def convert_nan_to_none(obj):
    """Recursively convert NaN and Infinity values to None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj




def get_ticker_data(ticker):
    """Get dataframe for a specific ticker"""
    ticker_upper = ticker.upper()
    if ticker_upper not in ticker_data:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found. Available: {list(ticker_data.keys())}")
    return ticker_data[ticker_upper]




def get_trading_days_list(ticker):
    """Get list of unique trading days from a ticker"""
    try:
        df = get_ticker_data(ticker)
        unique_dates = df['DayStr'].unique()
        trading_days = sorted([d for d in unique_dates if d])
        return trading_days
    except Exception as e:
        print(f"Error getting trading days: {e}")
        return []




# ============================================================================
# API ENDPOINTS - Trading Days
# ============================================================================



@app.get("/api/trading-days")
async def get_trading_days(ticker: Optional[str] = Query(None)):
    """
    Returns list of trading days with data available.
    If ticker not specified, defaults to first available ticker.
    """
    try:
        # If no ticker specified, use first available
        if not ticker:
            ticker = list(ticker_data.keys())[0] if ticker_data else 'TUZ5'
        
        trading_days = get_trading_days_list(ticker)
        
        if not trading_days:
            return {
                "ticker": ticker,
                "trading_days": [],
                "total_days": 0,
                "earliest_date": None,
                "latest_date": None
            }
        
        result = {
            "ticker": ticker,
            "trading_days": trading_days,
            "total_days": len(trading_days),
            "earliest_date": trading_days[0],
            "latest_date": trading_days[-1]
        }
        
        print(f"✓ Trading Days - {ticker}: {len(trading_days)} days")
        
        return result
        
    except Exception as e:
        print(f"Error in get_trading_days: {str(e)}")
        return {
            "ticker": ticker or "unknown",
            "trading_days": [],
            "total_days": 0,
            "earliest_date": None,
            "latest_date": None
        }




# ============================================================================
# ROUTES
# ============================================================================



@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page"""
    html_path = os.path.join(THIS_DIR, "templates", "landing.html")
    if not os.path.exists(html_path):
        return "<h1>Landing page not found</h1>"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()




@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard - uses TUZ5 data by default"""
    html_path = os.path.join(THIS_DIR, "templates", "index.html")
    print(f"Dashboard route called, looking for: {html_path}")
    print(f"File exists: {os.path.exists(html_path)}")
    
    if not os.path.exists(html_path):
        return "<h1>Dashboard not found at " + html_path + "</h1>"
    
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content




# ============================================================================
# CHART DATA ENDPOINT - WITH PROPER PRICE HANDLING
# ============================================================================



@app.get("/api/chart-data")
async def get_chart_data(
    selection_mode: str = Query(..., description="'calendar' or 'weekday'"),
    ticker: Optional[str] = Query(None, description="Ticker symbol (defaults to TUZ5)"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    weekdays: Optional[str] = Query(None, description="Comma-separated weekdays"),
    agg_mode: str = Query("selected", description="none, selected, or total")
):
    """
    Get filtered chart data and statistics with proper price handling
    """
    try:
        # Default to TUZ5 if not specified
        if not ticker:
            ticker = 'TUZ5'
        
        df = get_ticker_data(ticker)
        filtered = df.copy()
        
        # Filter based on selection mode
        if selection_mode == 'calendar':
            if end_date is None:
                end_date = start_date
            
            selected_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            selected_date_strs = selected_date_range.strftime('%Y-%m-%d').tolist()
            mask = filtered['DayStr'].isin(selected_date_strs)
            filtered = filtered.loc[mask]
            mode_label = "Calendar"
            
        elif selection_mode == 'weekday':
            selected_weekdays = weekdays.split(',') if weekdays else []
            mask = filtered['WeekDay'].isin(selected_weekdays)
            filtered = filtered.loc[mask]
            mode_label = f"Weekday ({', '.join(selected_weekdays)})"
        
        filtered = filtered.sort_values('TimePlot')
        
        if filtered.empty:
            return {
                "data": {},
                "stats": {"description": "No data available for selected criteria"}
            }
        
        # Get unique day labels
        selected_days = filtered['DayLabel'].unique().tolist()
        
        # Build traces for each day
        traces = []
        for label in selected_days:
            day_df = filtered.query('DayLabel == @label')
            
            # Handle price data - use actual price if available, otherwise relative price
            actual_prices = []
            for idx, row in day_df.iterrows():
                if 'Price' in day_df.columns and pd.notna(row.get('Price')) and row.get('Price') != 0:
                    actual_prices.append(float(row['Price']))
                else:
                    actual_prices.append(float(row['Relative Price']))
            
            traces.append({
                'x': day_df['TimePlot'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'y': day_df['Relative Price'].tolist(),
                'customdata': day_df['DayStr'].tolist(),
                'actual_prices': actual_prices,
                'type': 'scatter',
                'mode': 'lines',
                'name': label,
                'opacity': 0.7
            })
        
        # Add mean/SD aggregation if requested
        mean_traces = []
        if agg_mode != 'none':
            agg_df, mean_label = get_agg_df(filtered, agg_mode, selected_days, df)
            
            if agg_df is not None and not agg_df.empty:
                mean_curve = agg_df.groupby('TimePlot')['Relative Price'].mean().sort_index()
                sd_curve = agg_df.groupby('TimePlot')['Relative Price'].std().sort_index()
                
                # Mean trace
                mean_traces.append({
                    'x': mean_curve.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'y': mean_curve.values.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': f'{mean_label}',
                    'line': {'color': 'black', 'width': 3, 'dash': 'dot'}
                })
                
                # +1 SD
                mean_traces.append({
                    'x': sd_curve.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'y': (mean_curve.values + sd_curve.values).tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'width': 0},
                    'showlegend': False
                })
                
                # -1 SD with fill
                mean_traces.append({
                    'x': sd_curve.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'y': (mean_curve.values - sd_curve.values).tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'fill': 'tonexty',
                    'line': {'width': 0, 'color': 'gray'},
                    'name': f'{mean_label} ±1SD',
                    'opacity': 0.3
                })
        
        # Calculate statistics
        if agg_mode != 'none':
            agg_df, _ = get_agg_df(filtered, agg_mode, selected_days, df)
            if agg_df is not None and not agg_df.empty:
                sel_mean = agg_df['Relative Price'].mean()
                sel_sd = agg_df['Relative Price'].std()
                desc = (f"Mode: {agg_mode.title()} | Selection: {mode_label} | Days: {len(selected_days)} | "
                        f"Avg Relative Price: {sel_mean:.4f} | SD: {sel_sd:.4f}")
            else:
                desc = f"Selected {len(selected_days)} day(s)"
        else:
            desc = f"Showing {len(selected_days)} individual day(s) — Selection: {mode_label}"
        
        result = {
            "ticker": ticker,
            "data": {
                "traces": traces + mean_traces,
                "layout": {
                    'template': 'plotly_white',
                    'xaxis': {
                        'tickformat': '%H:%M',
                        'range': ['2000-01-01 18:00:00', '2000-01-02 17:59:00'],
                        'title': 'Time (18:00–17:59)'
                    },
                    'yaxis': {'title': 'Relative Price'},
                    'legend': {'title': {'text': 'Trading Day'}},
                    'hovermode': 'x unified',
                    'paper_bgcolor': '#ffffff',
                    'plot_bgcolor': '#ffffff',
                    'font': {'color': '#000000'}
                }
            },
            "stats": {"description": desc}
        }
        
        # Convert NaN/Infinity values to None before returning
        result = convert_nan_to_none(result)
        
        return result
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "data": {},
            "stats": {"description": f"Server error: {str(e)}"}
        }




# ============================================================================
# HELPER FUNCTIONS
# ============================================================================



def get_agg_df(filtered, agg_mode, selected_days, full_df):
    """Helper function to get aggregation dataframe"""
    if agg_mode == 'total':
        agg_df = full_df.copy()
        label = 'Total Mean'
    elif agg_mode == 'weekday' and len(selected_days) == 1:
        weekday = filtered['WeekDay'].iloc[0]
        agg_df = full_df[full_df['WeekDay'] == weekday].copy()
        label = f'{weekday} Mean'
    elif agg_mode == 'selected':
        agg_df = filtered.copy()
        label = 'Selected Mean'
    else:
        agg_df = None
        label = None
    return agg_df, label




# ============================================================================
# STARTUP EVENT
# ============================================================================



@app.on_event("startup")
async def startup_event():
    """Print startup information"""
    print("=" * 70)
    print("✓ Bond Futures Dashboard - FastAPI Application Started")
    print("=" * 70)
    print(f"Available Tickers: {list(ticker_data.keys())}")
    for ticker, df in ticker_data.items():
        min_date = pd.to_datetime(df['DayStr']).min().strftime('%Y-%m-%d')
        max_date = pd.to_datetime(df['DayStr']).max().strftime('%Y-%m-%d')
        print(f"  {ticker}: {len(df)} records | {df['DayStr'].nunique()} days | {min_date} to {max_date}")
    print("=" * 70)
    print("✓ Available Endpoints:")
    print("  - GET / (Landing page)")
    print("  - GET /dashboard (Dashboard - uses TUZ5 by default)")
    print("  - GET /api/trading-days (Trading days list)")
    print("  - GET /api/chart-data (Chart data with filters)")
    print("=" * 70)




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)