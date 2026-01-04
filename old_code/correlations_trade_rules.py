import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tp_sl import TAKE_PROFIT_Z, STOP_LOSS_Z

# Entry criteria (matches get_entry_criteria.py)
ENTRY_Z_MIN = 2.5
ENTRY_Z_MAX = 3.0

# Load data
P = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)
S = pd.read_csv("data/entry_signals.csv")

if S.empty:
    print("No entry signals found")
    exit()

print(f"Entry range: {ENTRY_Z_MIN} <= |z-score| < {ENTRY_Z_MAX}")
print(f"Take profit: Â±{TAKE_PROFIT_Z}")
print(f"Stop loss: Â±{STOP_LOSS_Z}")

# Calculate in-trade returns for all pairs
trade_returns = {}

for _, r in S.iterrows():
    if r['stock1'] not in P.columns or r['stock2'] not in P.columns:
        continue
    
    # Calculate spread and z-score
    spread = P[r['stock1']] - r['hedge_ratio'] * P[r['stock2']]
    mu = spread.mean()
    sigma = spread.std()
    z_score = (spread - mu) / sigma
    
    # Identify trade periods based on entry/exit logic
    in_trade = False
    trade_direction = None  # 'long' or 'short'
    trade_periods = []
    current_trade_start = None
    
    for date, z in z_score.items():
        if not in_trade:
            # Check for entry signal (2.5 <= |z| < 3.0)
            if z < -ENTRY_Z_MIN and z >= -ENTRY_Z_MAX:  # Spread too low, go long spread
                in_trade = True
                trade_direction = 'long'
                current_trade_start = date
            elif z > ENTRY_Z_MIN and z <= ENTRY_Z_MAX:  # Spread too high, go short spread
                in_trade = True
                trade_direction = 'short'
                current_trade_start = date
        else:
            # Check for exit signal
            exit_trade = False
            
            if trade_direction == 'long':
                # Exit long: take profit (z crosses above -TAKE_PROFIT_Z) or stop loss (z goes below -STOP_LOSS_Z)
                if z > -TAKE_PROFIT_Z or z < -STOP_LOSS_Z:
                    exit_trade = True
            else:  # short
                # Exit short: take profit (z crosses below TAKE_PROFIT_Z) or stop loss (z goes above STOP_LOSS_Z)
                if z < TAKE_PROFIT_Z or z > STOP_LOSS_Z:
                    exit_trade = True
            
            if exit_trade:
                trade_periods.append((current_trade_start, date, trade_direction))
                in_trade = False
                trade_direction = None
                current_trade_start = None
    
    # If still in trade at end of data, close it
    if in_trade and current_trade_start is not None:
        trade_periods.append((current_trade_start, z_score.index[-1], trade_direction))
    
    if not trade_periods:
        continue
    
    # Extract returns during trade periods
    spread_returns = spread.pct_change()
    in_trade_returns = []
    
    for start_date, end_date, direction in trade_periods:
        period_returns = spread_returns.loc[start_date:end_date].dropna()
        
        # Flip sign for short trades (we're shorting the spread)
        if direction == 'short':
            period_returns = -period_returns
        
        in_trade_returns.extend(period_returns.values)
    
    if len(in_trade_returns) > 0:
        pair_name = f"{r['stock1']}/{r['stock2']}"
        trade_returns[pair_name] = pd.Series(in_trade_returns)
        print(f"{pair_name}: {len(trade_periods)} trades, {len(in_trade_returns)} return observations")

if len(trade_returns) < 1:
    print("No valid pairs with trade periods for correlation analysis")
    pd.DataFrame().to_csv("data/pair_correlation_matrix.csv")
    exit()

# Find common length for all series (use minimum length to avoid NaN issues)
min_length = min(len(series) for series in trade_returns.values())
print(f"\nTruncating all return series to {min_length} observations for correlation calculation")

# Truncate all series to same length and build dataframe
truncated_returns = {pair: series.iloc[:min_length].reset_index(drop=True) 
                     for pair, series in trade_returns.items()}

df = pd.DataFrame(truncated_returns)
corr = df.corr()

# Save correlation matrix
corr.to_csv("data/pair_correlation_matrix.csv")

print(f"\n{'='*60}")
print(f"Correlation matrix saved: {len(corr)} pairs")
print(f"Based on {min_length} in-trade return observations per pair")
print(f"Mean correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].mean():.3f}")
print(f"Median correlation: {np.median(corr.values[np.triu_indices_from(corr.values, k=1)]):.3f}")
print(f"Max correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].max():.3f}")
print(f"Min correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].min():.3f}")
print(f"{'='*60}")