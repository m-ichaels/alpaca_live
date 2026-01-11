import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tp_sl import TAKE_PROFIT_Z, STOP_LOSS_Z

# Load data
P = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)
E = pd.read_csv("data/pair_edges.csv")

if E.empty:
    print("No pairs found in edge analysis")
    exit()

print(f"Analyzing correlations for {len(E)} pairs from edge analysis")
print(f"Take profit: ±{TAKE_PROFIT_Z}")
print(f"Stop loss: ±{STOP_LOSS_Z}")

# Calculate in-trade returns for all pairs aligned to calendar dates
trade_returns = {}

for _, r in E.iterrows():
    if r['stock1'] not in P.columns or r['stock2'] not in P.columns:
        continue
    
    # Calculate spread and z-score
    spread = P[r['stock1']] - r['hedge_ratio'] * P[r['stock2']]
    mu = spread.mean()
    sigma = spread.std()
    z_score = (spread - mu) / sigma
    
    # Get the current z-score and signal from edge analysis
    current_z = r['z_score']
    signal = r['signal']
    
    # Determine initial trade direction based on signal
    if signal == 'LONG_SPREAD':
        initial_direction = 'long'
    elif signal == 'SHORT_SPREAD':
        initial_direction = 'short'
    else:
        continue
    
    # Create a series to track when we're in a trade
    in_trade_mask = pd.Series(False, index=z_score.index)
    
    in_trade = False
    trade_direction = None
    entry_date = None
    
    for date, z in z_score.items():
        if not in_trade:
            # Check for entry signal matching the edge analysis
            # Entry occurs when z-score is between TAKE_PROFIT_Z and STOP_LOSS_Z
            abs_z = abs(z)
            
            if abs_z >= TAKE_PROFIT_Z and abs_z <= STOP_LOSS_Z:
                # Determine direction based on sign
                if z < 0 and initial_direction == 'long':
                    in_trade = True
                    trade_direction = 'long'
                    entry_date = date
                    in_trade_mask.loc[date] = True
                elif z > 0 and initial_direction == 'short':
                    in_trade = True
                    trade_direction = 'short'
                    entry_date = date
                    in_trade_mask.loc[date] = True
        else:
            # Check for exit signal
            exit_trade = False
            
            if trade_direction == 'long':
                # Long spread: exit if z-score crosses above -TAKE_PROFIT_Z or below -STOP_LOSS_Z
                if z > -TAKE_PROFIT_Z or z < -STOP_LOSS_Z:
                    exit_trade = True
            else:  # short
                # Short spread: exit if z-score crosses below TAKE_PROFIT_Z or above STOP_LOSS_Z
                if z < TAKE_PROFIT_Z or z > STOP_LOSS_Z:
                    exit_trade = True
            
            if exit_trade:
                in_trade = False
                trade_direction = None
                entry_date = None
            else:
                in_trade_mask.loc[date] = True
    
    # Calculate spread returns
    spread_returns = spread.pct_change()
    
    # Extract returns only during trade periods, keep dates aligned
    pair_returns = pd.Series(0.0, index=spread_returns.index)
    pair_returns[in_trade_mask] = spread_returns[in_trade_mask]
    
    # Handle short trades (flip sign)
    # We need to track direction per date
    in_trade = False
    trade_direction = None
    
    for date, z in z_score.items():
        if not in_trade:
            abs_z = abs(z)
            if abs_z >= TAKE_PROFIT_Z and abs_z <= STOP_LOSS_Z:
                if z < 0 and initial_direction == 'long':
                    in_trade = True
                    trade_direction = 'long'
                elif z > 0 and initial_direction == 'short':
                    in_trade = True
                    trade_direction = 'short'
        else:
            exit_trade = False
            
            if trade_direction == 'long':
                if z > -TAKE_PROFIT_Z or z < -STOP_LOSS_Z:
                    exit_trade = True
            else:
                if z < TAKE_PROFIT_Z or z > STOP_LOSS_Z:
                    exit_trade = True
                else:
                    # Flip sign for short positions
                    if date in pair_returns.index and in_trade_mask.loc[date]:
                        pair_returns.loc[date] = -pair_returns.loc[date]
            
            if exit_trade:
                in_trade = False
                trade_direction = None
    
    pair_name = f"{r['stock1']}/{r['stock2']}"
    num_trade_days = in_trade_mask.sum()
    
    if num_trade_days > 0:
        trade_returns[pair_name] = pair_returns.dropna()
        print(f"{pair_name}: {num_trade_days} days in trade (edge: {r['edge']:.4f})")

if len(trade_returns) < 1:
    print("No valid pairs with trade periods for correlation analysis")
    pd.DataFrame().to_csv("data/pair_correlation_matrix.csv")
    exit()

# Build dataframe with all dates - pairs naturally have zeros when not trading
all_dates = P.index
df_returns = pd.DataFrame(index=all_dates)

for pair_name, returns in trade_returns.items():
    df_returns[pair_name] = returns.reindex(all_dates, fill_value=0.0)

# Calculate correlation on the full time series
corr = df_returns.corr()

# Save correlation matrix
corr.to_csv("data/pair_correlation_matrix.csv")

# Calculate statistics
upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]

print(f"\n{'='*60}")
print(f"Correlation matrix saved: {len(corr)} pairs")
print(f"Based on {len(all_dates)} total trading days")
print(f"\nCorrelation statistics:")
print(f"  Mean correlation: {upper_tri.mean():.3f}")
print(f"  Median correlation: {np.median(upper_tri):.3f}")
print(f"  Max correlation: {upper_tri.max():.3f}")
print(f"  Min correlation: {upper_tri.min():.3f}")
print(f"{'='*60}")