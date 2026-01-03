import pandas as pd
import numpy as np
import os
import importlib.util
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET
from scipy.stats import norm

# Load TP/SL parameters from tp_sl.py
spec = importlib.util.spec_from_file_location("tp_sl", "tp_sl.py")
tp_sl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp_sl_module)

TAKE_PROFIT_Z = tp_sl_module.TAKE_PROFIT_Z
STOP_LOSS_Z = tp_sl_module.STOP_LOSS_Z

# Connect to Alpaca
trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

# Load signals and historical prices for spread statistics
signals_df = pd.read_csv("data/entry_signals.csv")
prices_df = pd.read_csv("data/sp500_prices_clean.csv")
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df = prices_df.set_index('date')

# Get account equity
account = trading_client.get_account()
total_equity = float(account.equity)

print(f"Total Account Equity: ${total_equity:,.2f}")
print(f"Processing {len(signals_df)} signals for Kelly sizing\n")

# Get current positions to exclude from allocation
positions = trading_client.get_all_positions()
current_holdings = {p.symbol for p in positions}

def calculate_adaptive_win_probability(z_score):
    """
    Calculate win probability using historical trade data when available,
    falling back to theoretical estimates.
    
    NOTE: Only uses trades with definitive outcomes (win=0 or win=1).
    Manual liquidations (win=None) are excluded from win rate calculations.
    """
    abs_z = abs(z_score)
    
    # Theoretical baseline
    if abs_z < 2.5:
        theoretical = 0.65
    elif abs_z < 3.0:
        theoretical = 0.72
    elif abs_z < 3.5:
        theoretical = 0.77
    else:
        theoretical = 0.80
    
    # Try to use empirical data if available
    if os.path.exists("data/trade_history.csv"):
        try:
            history = pd.read_csv("data/trade_history.csv")
            
            # Only use completed trades with definitive outcomes
            # Exclude manual liquidations (win=None/NaN)
            completed = history[
                (history['exit_date'].notna()) & 
                (history['win'].notna())
            ]
            
            if len(completed) >= 10:  # Minimum sample size
                # Filter similar z-score ranges (±0.5)
                similar_trades = completed[
                    (completed['entry_z'].abs() >= abs_z - 0.5) & 
                    (completed['entry_z'].abs() <= abs_z + 0.5)
                ]
                
                if len(similar_trades) >= 5:
                    empirical_win_rate = similar_trades['win'].mean()
                    # Blend: 60% empirical, 40% theoretical
                    blended = 0.6 * empirical_win_rate + 0.4 * theoretical
                    print(f"  Using blended win prob for |Z|={abs_z:.1f}: "
                          f"{blended:.1%} (empirical: {empirical_win_rate:.1%}, "
                          f"n={len(similar_trades)} definitive outcomes)")
                    return blended
                    
        except Exception as e:
            print(f"  Warning: Could not load trade history - {e}")
    
    return theoretical

def calculate_kelly_parameters(z_score, spread_mean, spread_std):
    """
    Calculate Kelly based on actual TP/SL levels from tp_sl.py
    """
    current_z = z_score
    abs_z = abs(current_z)
    
    # Calculate actual dollar distances to TP and SL
    if current_z > 0:  # Short spread (Z too high)
        profit_distance = abs(current_z - TAKE_PROFIT_Z)
        loss_distance = abs(STOP_LOSS_Z - current_z)
    else:  # Long spread (Z too low)
        profit_distance = abs(current_z + TAKE_PROFIT_Z)
        loss_distance = abs(-STOP_LOSS_Z - current_z)
    
    # Convert to dollar terms
    expected_profit = profit_distance * spread_std
    expected_loss = loss_distance * spread_std
    
    # Win/loss ratio from actual TP/SL
    win_loss_ratio = expected_profit / expected_loss if expected_loss > 0 else 1.0
    
    # Adaptive win probability (excludes manual liquidations)
    win_prob = calculate_adaptive_win_probability(z_score)
    
    # Kelly formula: f = (p*b - q) / b
    loss_prob = 1 - win_prob
    kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
    
    # Conservative 1/3 Kelly
    kelly_fraction = max(0, kelly_fraction) * 0.33
    
    # Cap at 8% per pair
    kelly_fraction = min(kelly_fraction, 0.08)
    
    return kelly_fraction, win_prob, win_loss_ratio

def calculate_hedge_positions(capital, price1, price2, beta):
    """
    Calculate position sizes maintaining hedge ratio from regression.
    
    From: stock1 = alpha + beta * stock2
    We need: (shares1 * price1) = beta * (shares2 * price2)
    
    Solving for capital allocation:
    value_stock1 = beta * capital / (1 + beta)
    value_stock2 = capital / (1 + beta)
    """
    # Allocate capital maintaining hedge ratio
    value_stock2 = capital / (1 + abs(beta))
    value_stock1 = abs(beta) * value_stock2
    
    # Convert to integer shares
    shares1 = int(value_stock1 / price1)
    shares2 = int(value_stock2 / price2)
    
    if shares1 < 1 or shares2 < 1:
        return None, None, None, None
    
    # Calculate actual capital used
    actual_value1 = shares1 * price1
    actual_value2 = shares2 * price2
    actual_capital = actual_value1 + actual_value2
    
    # Verify hedge ratio accuracy
    actual_ratio = actual_value1 / actual_value2
    expected_ratio = abs(beta)
    hedge_error = abs(actual_ratio - expected_ratio) / expected_ratio * 100
    
    # If error > 5%, adjust shares2 to match shares1 exactly
    if hedge_error > 5.0:
        shares2 = int((shares1 * price1) / (abs(beta) * price2))
        if shares2 < 1:
            return None, None, None, None
        
        actual_value1 = shares1 * price1
        actual_value2 = shares2 * price2
        actual_capital = actual_value1 + actual_value2
        hedge_error = abs((actual_value1/actual_value2) - abs(beta)) / abs(beta) * 100
    
    return shares1, shares2, actual_capital, hedge_error

def check_trade_feasibility(stock1, stock2, beta, capital, signal):
    """Check if we can execute trade with proper hedge ratio"""
    try:
        quotes = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[stock1, stock2])
        )
        
        price1 = (quotes[stock1].bid_price + quotes[stock1].ask_price) / 2
        price2 = (quotes[stock2].bid_price + quotes[stock2].ask_price) / 2
        
        if price1 <= 0 or price2 <= 0:
            return False, None, None, None, None, None
        
        shares1, shares2, actual_capital, hedge_error = \
            calculate_hedge_positions(capital, price1, price2, beta)
        
        if shares1 is None or hedge_error > 10.0:
            return False, None, None, None, None, None
        
        return True, shares1, shares2, price1, price2, actual_capital
    
    except Exception as e:
        return False, None, None, None, None, None

# Display trade history stats if available
if os.path.exists("data/trade_history.csv"):
    try:
        history = pd.read_csv("data/trade_history.csv")
        completed = history[history['exit_date'].notna()]
        definitive = completed[completed['win'].notna()]
        liquidated = completed[completed['win'].isna()]
        
        if len(completed) > 0:
            print("\n" + "=" * 80)
            print("TRADE HISTORY SUMMARY")
            print("=" * 80)
            print(f"Total completed trades: {len(completed)}")
            print(f"  Definitive outcomes (used for win rate): {len(definitive)}")
            if len(definitive) > 0:
                print(f"    Wins: {definitive['win'].sum():.0f}")
                print(f"    Losses: {(definitive['win'] == 0).sum():.0f}")
                print(f"    Win rate: {definitive['win'].mean():.1%}")
            print(f"  Manual liquidations (neutral, excluded): {len(liquidated)}")
            print("=" * 80 + "\n")
    except Exception as e:
        print(f"Warning: Could not load trade history stats - {e}\n")

# Calculate raw Kelly fractions for each pair
raw_kelly_data = []

for idx, row in signals_df.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    
    # Skip if already have positions
    if stock1 in current_holdings or stock2 in current_holdings:
        continue
    
    z_score = row['z_score']
    beta = row['hedge_ratio']
    abs_z = abs(z_score)
    
    # Calculate spread statistics from historical data
    spread = prices_df[stock1] - beta * prices_df[stock2]
    spread_mean = spread.mean()
    spread_std = spread.std()
    
    # Calculate Kelly using actual TP/SL levels
    kelly_fraction, win_prob, win_loss_ratio = calculate_kelly_parameters(
        z_score, spread_mean, spread_std
    )
    
    raw_kelly_data.append({
        'stock1': row['stock1'],
        'stock2': row['stock2'],
        'signal': row['signal'],
        'z_score': row['z_score'],
        'hedge_ratio': row['hedge_ratio'],
        'win_prob': win_prob,
        'win_loss_ratio': win_loss_ratio,
        'raw_kelly_fraction': kelly_fraction,
        'priority': abs_z
    })

# Convert to DataFrame and sort by priority
kelly_df = pd.DataFrame(raw_kelly_data)
kelly_df = kelly_df.sort_values('priority', ascending=False).reset_index(drop=True)

print(f"Raw Kelly Total: {kelly_df['raw_kelly_fraction'].sum():.2%}\n")

# Pre-filter for feasibility
print("Pre-filtering for feasibility...")
print("=" * 80)

feasible_pairs = []

for idx, row in kelly_df.iterrows():
    test_capital = row['raw_kelly_fraction'] * total_equity
    
    feasible, s1, s2, p1, p2, actual_cap = check_trade_feasibility(
        row['stock1'], row['stock2'], row['hedge_ratio'], 
        test_capital, row['signal']
    )
    
    if feasible:
        feasible_pairs.append({
            **row.to_dict(),
            'price1': p1,
            'price2': p2
        })

print(f"Feasible pairs: {len(feasible_pairs)}/{len(kelly_df)}\n")

if not feasible_pairs:
    print("No feasible pairs found!")
    exit(1)

# Recalculate based only on feasible pairs
feasible_df = pd.DataFrame(feasible_pairs)
total_feasible_kelly = feasible_df['raw_kelly_fraction'].sum()

print(f"Feasible Kelly Total: {total_feasible_kelly:.2%}")

# Target 95% of equity
target_total_allocation = 0.95 * total_equity

# Scale Kelly fractions to hit target
print("\nOptimal Allocation:")
print("=" * 80)

allocated_pairs = []
allocated_capital = 0

scaling_factor = min(target_total_allocation / (total_feasible_kelly * total_equity), 1.0)
print(f"Scaling factor: {scaling_factor:.4f}\n")

for idx, row in feasible_df.iterrows():
    target_kelly = row['raw_kelly_fraction'] * scaling_factor
    target_capital = target_kelly * total_equity
    
    feasible, s1, s2, p1, p2, actual_cap = check_trade_feasibility(
        row['stock1'], row['stock2'], row['hedge_ratio'], 
        target_capital, row['signal']
    )
    
    if feasible:
        allocated_pairs.append({
            **row,
            'allocated_capital': actual_cap,
            'shares1': s1,
            'shares2': s2,
            'price1': p1,
            'price2': p2,
            'kelly_fraction': actual_cap / total_equity
        })
        allocated_capital += actual_cap
        print(f"[OK] {row['stock1']:6s}/{row['stock2']:6s}: ${actual_cap:>9,.2f} ({actual_cap/total_equity:>5.2%})")
    else:
        print(f"[X]  {row['stock1']:6s}/{row['stock2']:6s}: Failed at ${target_capital:>9,.2f}")

# Create final DataFrame
final_df = pd.DataFrame(allocated_pairs)

if len(final_df) > 0:
    final_df = final_df.sort_values('priority', ascending=False)
    
    # Display all positions
    print("\n" + "=" * 80)
    print(f"FINAL ALLOCATION - All {len(final_df)} Positions:")
    print("=" * 80)
    for idx, row in final_df.iterrows():
        val1 = row['shares1'] * row['price1']
        val2 = row['shares2'] * row['price2']
        actual_ratio = val1 / val2
        hedge_error = abs(actual_ratio - abs(row['hedge_ratio'])) / abs(row['hedge_ratio']) * 100
        
        print(f"{row['stock1']}/{row['stock2']}:")
        print(f"  Z-score: {row['z_score']:6.2f}, Signal: {row['signal']}")
        print(f"  Kelly: {row['kelly_fraction']:.2%}, Capital: ${row['allocated_capital']:,.2f}")
        print(f"  Trade: {row['shares1']} {row['stock1']} @ ${row['price1']:.2f} = ${val1:,.2f}")
        print(f"        {row['shares2']} {row['stock2']} @ ${row['price2']:.2f} = ${val2:,.2f}")
        print(f"  Hedge Ratio: {actual_ratio:.3f} (target: {abs(row['hedge_ratio']):.3f}, error: {hedge_error:.1f}%)\n")
    
    # Save sized signals
    output_cols = ['stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio', 
                   'kelly_fraction', 'capital_allocation', 'shares1', 'shares2',
                   'price1', 'price2', 'win_prob', 'win_loss_ratio']
    
    final_df['capital_allocation'] = final_df['allocated_capital']
    final_df[output_cols].to_csv("data/sized_signals.csv", index=False)
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Pairs: {len(final_df)}")
    print(f"Total Capital Allocated: ${final_df['allocated_capital'].sum():,.2f}")
    print(f"Target Allocation: ${target_total_allocation:,.2f}")
    print(f"Allocation Efficiency: {final_df['allocated_capital'].sum() / target_total_allocation:.2%}")
    print(f"Portfolio Utilization: {final_df['allocated_capital'].sum() / total_equity:.2%}")
    print(f"Total Kelly Fraction: {final_df['kelly_fraction'].sum():.2%}")
    print(f"\nSized signals saved to data/sized_signals.csv")
else:
    print("\nNo feasible pairs found!")