import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os, sys


# Make rel_data importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
sys.path.append(SRC_DIR)
from rel_data import df_maker


# Load & process data
data = pd.read_csv(os.path.join(THIS_DIR, '..', 'data', 'oct_data.csv'))
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


# Get date range for calendar
min_date = pd.to_datetime(df['DayStr']).min()
max_date = pd.to_datetime(df['DayStr']).max()

# Get unique weekdays
unique_weekdays = sorted(df['WeekDay'].unique())


# Initialize Dash app
app = dash.Dash(__name__)


app.layout = html.Div([
    # White Banner with Title (Full Width)
    html.Div(
        html.H1("Bond Futures Overnight Rate", className="title"),
        id='title-banner',
        style={
            'width': '100%',
            'textAlign': 'center',
            'margin': '0',
            'padding': '0'
        }
    ),
    
    # Main Controls Container - Two columns
    html.Div([
        # Left Column - Date/Weekday Selection Mode
        html.Div([
            html.Label("Select Days By", style={
                'fontSize': '18px',
                'fontWeight': 'bold',
                'marginBottom': '15px',
                'display': 'block',
                'color': '#ffffff'
            }),
            
            dcc.RadioItems(
                id='selection-mode',
                options=[
                    {'label': ' Calendar', 'value': 'calendar'},
                    {'label': ' Weekday', 'value': 'weekday'}
                ],
                value='calendar',
                inline=False,
                style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px', 'marginBottom': '20px'},
                labelStyle={
                    'display': 'flex',
                    'alignItems': 'center',
                    'padding': '12px',
                    'backgroundColor': '#ffffff',
                    'border': '2px solid #3498db',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontWeight': '600',
                    'fontSize': '14px',
                    'color': '#000000',
                    'transition': 'all 0.2s'
                }
            ),
            
            # Calendar Selection (shown when mode = 'calendar')
            html.Div(id='calendar-container', style={
                'display': 'block'
            }, children=[
                html.Div([
                    html.Div([
                        html.Label("Start Date", style={'color': '#ffffff', 'fontSize': '14px', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.DatePickerSingle(
                            id='date-picker-start',
                            date=max_date,
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            display_format='YYYY-MM-DD',
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1', 'marginRight': '10px'}),
                    html.Div([
                        html.Label("End Date", style={'color': '#ffffff', 'fontSize': '14px', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.DatePickerSingle(
                            id='date-picker-end',
                            date=max_date,
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            display_format='YYYY-MM-DD',
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1'}),
                ], style={'display': 'flex', 'gap': '10px'}),
            ]),
            
            # Weekday Selection (shown when mode = 'weekday')
            html.Div(id='weekday-container', style={
                'display': 'none'
            }, children=[
                html.Label("Select Weekdays", style={'color': '#ffffff', 'fontSize': '14px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='weekday-multi-select',
                    options=[{'label': day, 'value': day} for day in unique_weekdays],
                    value=unique_weekdays,
                    multi=True,
                    style={'width': '100%'}
                ),
            ]),
        ], style={
            'width': '48%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'paddingRight': '15px',
            'boxSizing': 'border-box'
        }),
        
        # Right Column - Aggregation Buttons (2x2 via RadioItems)
        html.Div([
            html.Label("Aggregation Type", style={
                'fontSize': '18px',
                'fontWeight': 'bold',
                'marginBottom': '15px',
                'display': 'block',
                'color': '#ffffff'
            }),
            
            dcc.RadioItems(
                id='agg-toggle',
                options=[
                    {'label': ' None', 'value': 'none'},
                    {'label': ' Selected Mean/SD', 'value': 'selected'},
                    {'label': ' Total Mean/SD', 'value': 'total'},
                    {'label': ' Weekday Mean/SD', 'value': 'weekday'}
                ],
                value='selected',
                inline=False,
                style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'},
                labelStyle={
                    'display': 'flex',
                    'alignItems': 'center',
                    'padding': '12px',
                    'backgroundColor': '#ffffff',
                    'border': '2px solid #3498db',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontWeight': '600',
                    'fontSize': '14px',
                    'color': '#000000',
                    'transition': 'all 0.2s'
                }
            ),
        ], style={
            'width': '48%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'paddingLeft': '15px',
            'boxSizing': 'border-box'
        }),
    ], style={
        'maxWidth': '1200px',
        'margin': '20px auto',
        'padding': '25px',
        'backgroundColor': '#61a5da',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
        'display': 'flex',
        'gap': '20px'
    }),
    
    # Stats Output (Full Width Below)
    html.Div(id='stats-output', style={
        'textAlign': 'center',
        'fontSize': '16px',
        'padding': '15px',
        'margin': '20px auto',
        'maxWidth': '1200px',
        'backgroundColor': "#61a5da",
        'borderRadius': '6px',
        'fontWeight': '500',
        'color': '#ffffff',
        'border': '1px solid #3498db'
    }),
    
    # Chart (Full Width with White Background)
    html.Div([
        dcc.Graph(id='price-chart')
    ], style={
        'maxWidth': '1200px',
        'margin': '20px auto',
        'backgroundColor': '#ffffff',
        'borderRadius': '8px',
        'padding': '20px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
    }),
], style={
    'backgroundColor': "#ffffff",
    'minHeight': '100vh',
    'padding': '0',
    'margin': '0'
})


def get_agg_df(filtered, agg_mode, selected_days):
    if agg_mode == 'total':
        agg_df = df.copy()
        label = 'Total Mean'
    elif agg_mode == 'weekday' and len(selected_days) == 1:
        weekday = filtered['WeekDay'].iloc[0]
        agg_df = df[df['WeekDay'] == weekday].copy()
        label = f'{weekday} Mean'
    elif agg_mode == 'selected':
        agg_df = filtered.copy()
        label = 'Selected Mean'
    else:
        agg_df = None
        label = None
    return agg_df, label


# Callback to toggle between calendar and weekday containers
@app.callback(
    [Output('calendar-container', 'style'),
     Output('weekday-container', 'style')],
    Input('selection-mode', 'value')
)
def toggle_selection_mode(mode):
    if mode == 'calendar':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


# Main chart callback
@app.callback(
    [Output('price-chart', 'figure'),
     Output('stats-output', 'children')],
    [Input('selection-mode', 'value'),
     Input('date-picker-start', 'date'),
     Input('date-picker-end', 'date'),
     Input('weekday-multi-select', 'value'),
     Input('agg-toggle', 'value')]
)
def update_chart(selection_mode, start_date, end_date, selected_weekdays, agg_mode):
    filtered = df.copy()
    
    # Filter based on selection mode
    if selection_mode == 'calendar':
        # Calendar mode: use date range
        if end_date is None:
            end_date = start_date
        
        selected_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        selected_date_strs = selected_date_range.strftime('%Y-%m-%d').tolist()
        
        mask = filtered['DayStr'].isin(selected_date_strs)
        filtered = filtered.loc[mask]
        mode_label = "Calendar"
        
    elif selection_mode == 'weekday':
        # Weekday mode: filter by selected weekdays
        if not selected_weekdays:
            selected_weekdays = []
        
        mask = filtered['WeekDay'].isin(selected_weekdays)
        filtered = filtered.loc[mask]
        mode_label = f"Weekday ({', '.join(selected_weekdays)})"
    
    filtered = filtered.sort_values('TimePlot')
    
    if filtered.empty:
        return go.Figure(), "No data available for selected criteria"

    # Get unique day labels for this selection
    selected_days = filtered['DayLabel'].unique()

    # Per-day traces
    fig = go.Figure()
    for label in selected_days:
        day_df = filtered.query('DayLabel == @label')
        fig.add_trace(go.Scatter(
            x=day_df['TimePlot'], y=day_df['Relative Price'],
            mode='lines', name=label, opacity=0.7
        ))

    # Mean/SD aggregation (only if not 'none')
    if agg_mode != 'none':
        agg_df, mean_label = get_agg_df(filtered, agg_mode, selected_days)
        
        if agg_df is not None and not agg_df.empty:
            mean_curve = agg_df.groupby('TimePlot')['Relative Price'].mean().sort_index()
            sd_curve = agg_df.groupby('TimePlot')['Relative Price'].std().sort_index()

            # Mean trace
            fig.add_trace(go.Scatter(
                x=mean_curve.index, y=mean_curve.values,
                mode='lines', name=f'{mean_label}',
                line=dict(color='black', width=3, dash='dot')
            ))
            # ±1 SD fill
            fig.add_trace(go.Scatter(
                x=sd_curve.index, y=mean_curve.values + sd_curve.values,
                mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=sd_curve.index, y=mean_curve.values - sd_curve.values,
                mode='lines', fill='tonexty', line=dict(width=0, color='gray'),
                name=f'{mean_label} ±1SD', opacity=0.3
            ))

    fig.update_layout(
        template='plotly_white',
        xaxis=dict(
            tickformat='%H:%M',
            range=[pd.Timestamp('2000-01-01 18:00:00'),
                   pd.Timestamp('2000-01-02 17:59:00')],
            title='Time (18:00–17:59)'
        ),
        yaxis_title='Relative Price',
        legend_title='Trading Day',
        hovermode='x unified',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='#000000')
    )

    # Stats description
    if agg_mode != 'none':
        agg_df, _ = get_agg_df(filtered, agg_mode, selected_days)
        if agg_df is not None and not agg_df.empty:
            sel_mean = agg_df['Relative Price'].mean()
            sel_sd = agg_df['Relative Price'].std()
            desc = (f"Mode: {agg_mode.title()} | Selection: {mode_label} | Days: {len(selected_days)} | "
                    f"Avg Relative Price: {sel_mean:.4f} | SD: {sel_sd:.4f}")
        else:
            desc = f"Selected {len(selected_days)} day(s)"
    else:
        desc = f"Showing {len(selected_days)} individual day(s) — Selection: {mode_label}"

    return fig, desc


if __name__ == '__main__':
    app.run(debug=True)
