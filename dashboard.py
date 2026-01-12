import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
try:
    from auth_local import KEY, SECRET  # For local testing
except ImportError:
    print("Import error")
    from auth import KEY, SECRET  # For GitHub Actions

# Initialize Dash app
app = dash.Dash(__name__)

# Connect to Alpaca
trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

def liquidate_all_positions():
    """Liquidate all positions and cancel all orders"""
    results = {'success': True, 'message': '', 'positions_closed': 0, 'errors': []}
    
    try:
        # Cancel all open orders
        trading_client.cancel_orders()
        results['message'] += "Canceled all open orders. "
    except Exception as e:
        results['errors'].append(f"Error canceling orders: {e}")
    
    try:
        # Liquidate all positions
        positions = trading_client.get_all_positions()
        
        if not positions:
            results['message'] += "No positions to liquidate."
        else:
            for position in positions:
                symbol = position.symbol
                qty = abs(float(position.qty))
                side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY
                
                try:
                    order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    order = trading_client.submit_order(order_data)
                    results['positions_closed'] += 1
                except Exception as e:
                    results['errors'].append(f"{symbol}: {e}")
            
            results['message'] += f"Closed {results['positions_closed']} positions."
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Error liquidating: {e}")
    
    return results

def get_timeframe_params(timeframe):
    """Convert timeframe string to days and portfolio history period"""
    timeframe_map = {
        '1D': {'days': 1, 'period': '1D', 'alpaca_timeframe': '1Min'},
        '1W': {'days': 7, 'period': '1W', 'alpaca_timeframe': '5Min'},
        '1M': {'days': 30, 'period': '1M', 'alpaca_timeframe': '1D'},
        '3M': {'days': 90, 'period': '3M', 'alpaca_timeframe': '1D'},
        '1Y': {'days': 365, 'period': '1A', 'alpaca_timeframe': '1D'},
        'MAX': {'days': 1000, 'period': 'all', 'alpaca_timeframe': '1D'}
    }
    return timeframe_map.get(timeframe, timeframe_map['1M'])

def calculate_statistics(timeframe='1M'):
    """Calculate all trading statistics from Alpaca data"""
    
    params = get_timeframe_params(timeframe)
    days = params['days']
    
    # Get account information
    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    
    # Get current positions
    positions = trading_client.get_all_positions()
    current_positions_value = sum(float(p.market_value) for p in positions)
    unrealized_pl = sum(float(p.unrealized_pl) for p in positions)
    unrealized_pl_pct = (unrealized_pl / (portfolio_value - unrealized_pl)) * 100 if portfolio_value - unrealized_pl > 0 else 0
    
    # Calculate % of capital allocated
    pct_allocated = (current_positions_value / portfolio_value) * 100 if portfolio_value > 0 else 0
    
    # Get closed positions (trades history)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        activities = trading_client.get_activities(
            activity_types='FILL',
            after=start_date.isoformat()
        )
    except:
        activities = []
    
    # Process trades to calculate statistics
    trades = []
    for activity in activities:
        trades.append({
            'symbol': activity.symbol,
            'side': activity.side,
            'qty': float(activity.qty),
            'price': float(activity.price),
            'timestamp': activity.transaction_time,
            'value': float(activity.qty) * float(activity.price)
        })
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Calculate trade-level statistics
    if not trades_df.empty:
        # Group by symbol to find matched trades
        trades_df = trades_df.sort_values('timestamp')
        
        # Simple P&L calculation: match buys and sells
        completed_trades = []
        holdings = {}
        
        for _, trade in trades_df.iterrows():
            symbol = trade['symbol']
            
            if symbol not in holdings:
                holdings[symbol] = {'qty': 0, 'cost_basis': 0, 'entry_time': None}
            
            if trade['side'] == 'buy':
                # Add to position
                total_cost = holdings[symbol]['cost_basis'] * holdings[symbol]['qty'] + trade['value']
                holdings[symbol]['qty'] += trade['qty']
                holdings[symbol]['cost_basis'] = total_cost / holdings[symbol]['qty'] if holdings[symbol]['qty'] > 0 else 0
                if holdings[symbol]['entry_time'] is None:
                    holdings[symbol]['entry_time'] = trade['timestamp']
            else:  # sell
                # Close position (full or partial)
                if holdings[symbol]['qty'] > 0:
                    sell_qty = min(trade['qty'], holdings[symbol]['qty'])
                    pnl = (trade['price'] - holdings[symbol]['cost_basis']) * sell_qty
                    
                    holding_time = (trade['timestamp'] - holdings[symbol]['entry_time']).total_seconds() / 3600 if holdings[symbol]['entry_time'] else 0
                    
                    completed_trades.append({
                        'symbol': symbol,
                        'pnl': pnl,
                        'holding_time_hours': holding_time,
                        'exit_time': trade['timestamp']
                    })
                    
                    holdings[symbol]['qty'] -= sell_qty
                    if holdings[symbol]['qty'] == 0:
                        holdings[symbol]['entry_time'] = None
        
        completed_df = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()
        
        if not completed_df.empty:
            # Last 100 trades
            recent_trades = completed_df.tail(100)
            
            wins = recent_trades[recent_trades['pnl'] > 0]
            losses = recent_trades[recent_trades['pnl'] < 0]
            
            num_wins = len(wins)
            num_losses = len(losses)
            total_trades = len(recent_trades)
            
            win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = wins['pnl'].mean() if not wins.empty else 0
            avg_loss = losses['pnl'].mean() if not losses.empty else 0
            
            expected_value = recent_trades['pnl'].mean() if not recent_trades.empty else 0
            
            avg_holding_time = recent_trades['holding_time_hours'].mean() if not recent_trades.empty else 0
        else:
            num_wins = num_losses = total_trades = 0
            win_rate = avg_win = avg_loss = expected_value = avg_holding_time = 0
            completed_df = pd.DataFrame()
    else:
        num_wins = num_losses = total_trades = 0
        win_rate = avg_win = avg_loss = expected_value = avg_holding_time = 0
        completed_df = pd.DataFrame()
    
    # Get portfolio history for returns and Sharpe ratio
    portfolio_history_data = []
    try:
        portfolio_history = trading_client.get_portfolio_history(
            period=params['period'],
            timeframe=params['alpaca_timeframe']
        )
        
        if portfolio_history and portfolio_history.equity:
            equity_curve = np.array(portfolio_history.equity)
            timestamps = portfolio_history.timestamp
            
            # Store for graphing
            portfolio_history_data = [
                {'timestamp': datetime.fromtimestamp(ts), 'equity': eq, 'profit_loss': eq - equity_curve[0]}
                for ts, eq in zip(timestamps, equity_curve)
            ]
            
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            total_return = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100
            
            # Annualized rate of return
            days_actual = len(equity_curve)
            rate_of_return = ((equity_curve[-1] / equity_curve[0]) ** (252 / days_actual) - 1) * 100 if days_actual > 0 else 0
            
            # Sharpe ratio (annualized, assuming risk-free rate = 0)
            if len(returns) > 1:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - cumulative) / cumulative * 100
            max_drawdown = drawdown.min()
            
            # Drawdown duration (days in current drawdown)
            if equity_curve[-1] < cumulative[-1]:
                peak_idx = np.where(equity_curve == cumulative[-1])[0][-1]
                drawdown_duration = len(equity_curve) - peak_idx
            else:
                drawdown_duration = 0
        else:
            total_return = rate_of_return = sharpe_ratio = max_drawdown = drawdown_duration = 0
    except:
        total_return = rate_of_return = sharpe_ratio = max_drawdown = drawdown_duration = 0
    
    # Get current positions details
    positions_data = []
    for p in positions:
        positions_data.append({
            'symbol': p.symbol,
            'qty': float(p.qty),
            'market_value': float(p.market_value),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc) * 100,
            'side': 'long' if float(p.qty) > 0 else 'short'
        })
    
    return {
        'portfolio_value': portfolio_value,
        'cash': cash,
        'win_rate': win_rate,
        'num_wins': num_wins,
        'num_losses': num_losses,
        'total_trades': total_trades,
        'expected_value': expected_value,
        'avg_holding_time': avg_holding_time,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pct_allocated': pct_allocated,
        'total_return': total_return,
        'rate_of_return': rate_of_return,
        'unrealized_pl_pct': unrealized_pl_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'drawdown_duration': drawdown_duration,
        'completed_trades': completed_df,
        'portfolio_history': portfolio_history_data,
        'positions': positions_data
    }

# Layout
app.layout = html.Div([
    html.Div([
        # Header with title and liquidate button
        html.Div([
            html.H1("Trading Statistics Dashboard", 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px', 'flex': '1'}),
            html.Button('LIQUIDATE ALL', id='liquidate-btn', n_clicks=0,
                       style={
                           'padding': '12px 24px',
                           'backgroundColor': '#e74c3c',
                           'color': 'white',
                           'border': 'none',
                           'borderRadius': '6px',
                           'fontSize': '16px',
                           'fontWeight': 'bold',
                           'cursor': 'pointer',
                           'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                           'transition': 'all 0.3s',
                           'position': 'absolute',
                           'right': '20px',
                           'top': '20px'
                       })
        ], style={'position': 'relative', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        
        # Liquidation status message
        html.Div(id='liquidate-status', style={'textAlign': 'center', 'marginBottom': '10px', 'minHeight': '30px'}),
        
        # Timeframe selector
        html.Div([
            html.Div([
                html.Button('1D', id='btn-1D', n_clicks=0, 
                           style={'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
                                  'backgroundColor': '#ecf0f1', 'border': '1px solid #bdc3c7',
                                  'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500'}),
                html.Button('1W', id='btn-1W', n_clicks=0,
                           style={'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
                                  'backgroundColor': '#ecf0f1', 'border': '1px solid #bdc3c7',
                                  'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500'}),
                html.Button('1M', id='btn-1M', n_clicks=0,
                           style={'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
                                  'backgroundColor': '#3498db', 'border': '1px solid #2980b9',
                                  'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500',
                                  'color': 'white'}),
                html.Button('3M', id='btn-3M', n_clicks=0,
                           style={'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
                                  'backgroundColor': '#ecf0f1', 'border': '1px solid #bdc3c7',
                                  'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500'}),
                html.Button('1Y', id='btn-1Y', n_clicks=0,
                           style={'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
                                  'backgroundColor': '#ecf0f1', 'border': '1px solid #bdc3c7',
                                  'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500'}),
                html.Button('MAX', id='btn-MAX', n_clicks=0,
                           style={'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
                                  'backgroundColor': '#ecf0f1', 'border': '1px solid #bdc3c7',
                                  'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500'}),
            ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
            
            # Hidden div to store current timeframe
            html.Div(id='current-timeframe', children='1M', style={'display': 'none'})
        ]),
        
        html.Div(id='stats-container', style={'padding': '20px'}),
        
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # Update every 30 seconds
            n_intervals=0
        )
    ], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})
])

@app.callback(
    Output('liquidate-status', 'children'),
    Input('liquidate-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_liquidation(n_clicks):
    """Handle liquidation button click"""
    if n_clicks > 0:
        results = liquidate_all_positions()
        
        if results['success']:
            message_style = {
                'padding': '12px 20px',
                'backgroundColor': '#2ecc71',
                'color': 'white',
                'borderRadius': '6px',
                'fontWeight': '500',
                'display': 'inline-block'
            }
            message = f"✓ {results['message']}"
        else:
            message_style = {
                'padding': '12px 20px',
                'backgroundColor': '#e74c3c',
                'color': 'white',
                'borderRadius': '6px',
                'fontWeight': '500',
                'display': 'inline-block'
            }
            message = f"✗ Liquidation failed: {' '.join(results['errors'])}"
        
        if results['errors'] and results['success']:
            message += f" (Errors: {', '.join(results['errors'])})"
        
        return html.Div(message, style=message_style)
    
    return ""

@app.callback(
    Output('current-timeframe', 'children'),
    [Input('btn-1D', 'n_clicks'),
     Input('btn-1W', 'n_clicks'),
     Input('btn-1M', 'n_clicks'),
     Input('btn-3M', 'n_clicks'),
     Input('btn-1Y', 'n_clicks'),
     Input('btn-MAX', 'n_clicks')],
    prevent_initial_call=False
)
def update_timeframe(n1d, n1w, n1m, n3m, n1y, nmax):
    """Update the selected timeframe based on button clicks"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return '1M'
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    timeframe_map = {
        'btn-1D': '1D',
        'btn-1W': '1W',
        'btn-1M': '1M',
        'btn-3M': '3M',
        'btn-1Y': '1Y',
        'btn-MAX': 'MAX'
    }
    
    return timeframe_map.get(button_id, '1M')

@app.callback(
    [Output('btn-1D', 'style'),
     Output('btn-1W', 'style'),
     Output('btn-1M', 'style'),
     Output('btn-3M', 'style'),
     Output('btn-1Y', 'style'),
     Output('btn-MAX', 'style')],
    Input('current-timeframe', 'children')
)
def update_button_styles(timeframe):
    """Update button styles to highlight the selected timeframe"""
    base_style = {
        'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
        'backgroundColor': '#ecf0f1', 'border': '1px solid #bdc3c7',
        'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500'
    }
    
    active_style = {
        'margin': '0 5px', 'padding': '8px 16px', 'cursor': 'pointer',
        'backgroundColor': '#3498db', 'border': '1px solid #2980b9',
        'borderRadius': '4px', 'fontSize': '14px', 'fontWeight': '500',
        'color': 'white'
    }
    
    styles = {
        '1D': base_style.copy(),
        '1W': base_style.copy(),
        '1M': base_style.copy(),
        '3M': base_style.copy(),
        '1Y': base_style.copy(),
        'MAX': base_style.copy()
    }
    
    styles[timeframe] = active_style
    
    return styles['1D'], styles['1W'], styles['1M'], styles['3M'], styles['1Y'], styles['MAX']

@app.callback(
    Output('stats-container', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('current-timeframe', 'children')]
)
def update_stats(n, timeframe):
    stats = calculate_statistics(timeframe)
    
    # Load cointegrated groups if available
    try:
        groups_df = pd.read_csv("data/cointegrated_groups.csv")
        executed_trades_df = pd.read_csv("data/executed_trades.csv") if pd.io.common.file_exists("data/executed_trades.csv") else pd.DataFrame()
    except:
        groups_df = pd.DataFrame()
        executed_trades_df = pd.DataFrame()
    
    # Match positions to groups
    current_positions = {p['symbol'] for p in stats['positions']}
    group_status = []
    
    if not groups_df.empty and not executed_trades_df.empty:
        for _, group_row in groups_df.iterrows():
            group_tickers = set(group_row['group'].split(','))
            # Check if any ticker from this group is in current positions
            in_position = group_tickers.intersection(current_positions)
            if in_position:
                # Find trades for this group
                group_trades = executed_trades_df[executed_trades_df['group'] == group_row['group']]
                if not group_trades.empty:
                    status = 'OPEN'
                    tickers_str = group_row['group'][:50] + '...' if len(group_row['group']) > 50 else group_row['group']
                    group_status.append({
                        'group': tickers_str,
                        'status': status,
                        'size': group_row['size']
                    })
    
    stat_cards = [
        # Portfolio Overview
        html.Div([
            html.H2("Portfolio Overview", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div([
                create_stat_card("Portfolio Value", f"${stats['portfolio_value']:,.2f}", "#3498db"),
                create_stat_card("Total Return", f"{stats['total_return']:.2f}%", 
                               "#2ecc71" if stats['total_return'] >= 0 else "#e74c3c"),
                create_stat_card("Rate of Return (Annualized)", f"{stats['rate_of_return']:.2f}%",
                               "#2ecc71" if stats['rate_of_return'] >= 0 else "#e74c3c"),
                create_stat_card("Unrealized P&L", f"{stats['unrealized_pl_pct']:.2f}%",
                               "#2ecc71" if stats['unrealized_pl_pct'] >= 0 else "#e74c3c"),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}),
        ]),
        
        # Trade Performance
        html.Div([
            html.H2("Trade Performance (Last 100 Trades)", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div([
                create_stat_card("Win Rate", f"{stats['win_rate']:.1f}%", 
                               "#2ecc71" if stats['win_rate'] >= 50 else "#e74c3c"),
                create_stat_card("Wins / Losses", f"{stats['num_wins']} / {stats['num_losses']}", "#95a5a6"),
                create_stat_card("Total Trades", f"{stats['total_trades']}", "#95a5a6"),
                create_stat_card("Expected Value", f"${stats['expected_value']:.2f}",
                               "#2ecc71" if stats['expected_value'] >= 0 else "#e74c3c"),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}),
        ]),
        
        # Trade Metrics
        html.Div([
            html.H2("Trade Metrics", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div([
                create_stat_card("Average Win", f"${stats['avg_win']:.2f}", "#2ecc71"),
                create_stat_card("Average Loss", f"${stats['avg_loss']:.2f}", "#e74c3c"),
                create_stat_card("Avg Holding Time", f"{stats['avg_holding_time']:.1f} hours", "#9b59b6"),
                create_stat_card("Capital Allocated", f"{stats['pct_allocated']:.1f}%", "#f39c12"),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}),
        ]),
        
        # Risk Metrics
        html.Div([
            html.H2("Risk Metrics", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div([
                create_stat_card("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}", 
                               "#2ecc71" if stats['sharpe_ratio'] >= 1 else "#e74c3c"),
                create_stat_card("Max Drawdown", f"{stats['max_drawdown']:.2f}%", "#e74c3c"),
                create_stat_card("Drawdown Duration", f"{stats['drawdown_duration']} days", "#e67e22"),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}),
        ]),
        
        # Portfolio P&L Graph
        html.Div([
            html.H2(f"Portfolio P&L Over Time ({timeframe})", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            dcc.Graph(
                figure=create_portfolio_pl_graph(stats['portfolio_history']),
                config={'displayModeBar': False}
            )
        ], style={'marginBottom': '30px'}),
        
        # P&L Bar Chart
        html.Div([
            html.H2("Trade P&L Distribution", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            dcc.Graph(
                figure=create_pnl_barchart(stats['completed_trades']),
                config={'displayModeBar': False}
            )
        ], style={'marginBottom': '30px'}),
        
        # Current Groups Status
        html.Div([
            html.H2("Current Trading Groups", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            create_groups_table(group_status)
        ], style={'marginBottom': '30px'}),
        
        # Current Exposure Bar Chart
        html.Div([
            html.H2("Current Exposure", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            dcc.Graph(
                figure=create_exposure_chart(stats['positions'], stats['cash']),
                config={'displayModeBar': False}
            )
        ], style={'marginBottom': '30px'}),
        
        html.Div(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Timeframe: {timeframe}", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '20px', 'fontSize': '14px'})
    ]
    
    return stat_cards

def create_stat_card(title, value, color):
    return html.Div([
        html.Div(title, style={
            'fontSize': '14px',
            'color': '#7f8c8d',
            'marginBottom': '10px',
            'fontWeight': '500'
        }),
        html.Div(value, style={
            'fontSize': '28px',
            'fontWeight': 'bold',
            'color': color
        })
    ], style={
        'backgroundColor': '#f8f9fa',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'border': f'1px solid {color}',
        'borderLeft': f'4px solid {color}'
    })

def create_portfolio_pl_graph(portfolio_history):
    """Create portfolio P&L line graph"""
    if not portfolio_history:
        return go.Figure().add_annotation(
            text="No portfolio history data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(portfolio_history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['profit_loss'],
        mode='lines',
        name='P&L',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Cumulative Profit/Loss",
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')
    
    return fig

def create_pnl_barchart(completed_trades):
    """Create P&L bar chart for completed trades"""
    if completed_trades.empty:
        return go.Figure().add_annotation(
            text="No completed trades available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Take last 50 trades for visibility
    df = completed_trades.tail(50).copy()
    df['color'] = df['pnl'].apply(lambda x: '#2ecc71' if x > 0 else '#e74c3c')
    df['trade_num'] = range(len(df))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['trade_num'],
        y=df['pnl'],
        marker_color=df['color'],
        text=df['symbol'],
        hovertemplate='<b>%{text}</b><br>P&L: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Last 50 Completed Trades",
        xaxis_title="Trade Number",
        yaxis_title="P&L ($)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')
    
    return fig

def create_groups_table(group_status):
    """Create table showing current trading groups"""
    if not group_status:
        return html.Div(
            "No active trading groups",
            style={
                'textAlign': 'center',
                'padding': '40px',
                'color': '#7f8c8d',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px'
            }
        )
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Group", style={'padding': '12px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db'}),
            html.Th("Size", style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '2px solid #3498db'}),
            html.Th("Status", style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '2px solid #3498db'})
        ]))
    ]
    
    table_body = [
        html.Tbody([
            html.Tr([
                html.Td(g['group'], style={'padding': '12px', 'borderBottom': '1px solid #ecf0f1'}),
                html.Td(g['size'], style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '1px solid #ecf0f1'}),
                html.Td(
                    html.Span(
                        g['status'],
                        style={
                            'padding': '4px 12px',
                            'borderRadius': '12px',
                            'backgroundColor': '#2ecc71' if g['status'] == 'OPEN' else '#95a5a6',
                            'color': 'white',
                            'fontSize': '12px',
                            'fontWeight': 'bold'
                        }
                    ),
                    style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '1px solid #ecf0f1'}
                )
            ]) for g in group_status
        ])
    ]
    
    return html.Table(
        table_header + table_body,
        style={
            'width': '100%',
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'overflow': 'hidden',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
    )

def create_exposure_chart(positions, cash):
    """Create exposure bar chart including cash"""
    exposure_data = []
    
    # Add cash
    exposure_data.append({
        'name': 'Cash',
        'value': cash,
        'color': '#95a5a6'
    })
    
    # Add positions
    for p in positions:
        exposure_data.append({
            'name': p['symbol'],
            'value': abs(p['market_value']),
            'color': '#2ecc71' if p['side'] == 'long' else '#e74c3c'
        })
    
    if not exposure_data:
        return go.Figure().add_annotation(
            text="No exposure data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(exposure_data)
    df = df.sort_values('value', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['value'],
        y=df['name'],
        orientation='h',
        marker_color=df['color'],
        text=df['value'].apply(lambda x: f'${x:,.0f}'),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Value: $%{x:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Allocation",
        xaxis_title="Value ($)",
        yaxis_title="",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(400, len(exposure_data) * 30),
        showlegend=False,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=False)
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)