import pandas as pd
import numpy as np

# Load data
df_prices = pd.read_csv("data/sp500_prices_clean.csv")
df_prices['date'] = pd.to_datetime(df_prices['date'])
df_prices = df_prices.set_index('date')

df_pairs = pd.read_csv("data/pairs_with_edge.csv")

print(f"Calculating correlations for {len(df_pairs)} pairs with edge data")

# Calculate in-trade returns for all pairs
trade_returns = {}

for idx, row in df_pairs.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    beta = row['hedge_ratio']
    spread_mean = row['spread_mean']
    spread_std = row['spread_std']
    half_life = row['half_life']
    take_profit_z = row['take_profit']
    
    # Get price series
    y = df_prices[stock1]
    x = df_prices[stock2]
    
    # Calculate spread and z-score time series
    spread = y - beta * x
    z_scores = (spread - spread_mean) / spread_std
    
    # Calculate EMA of z-scores (trailing stop loss)
    alpha = 1 - np.exp(-half_life * 1)  # dt = 1 day
    ema_z = np.zeros(len(z_scores))
    ema_z[0] = z_scores.iloc[0]
    
    for i in range(1, len(z_scores)):
        ema_z[i] = alpha * z_scores.iloc[i] + (1 - alpha) * ema_z[i-1]
    
    # Calculate spread returns (percentage change)
    spread_returns = spread.pct_change()
    
    # Create returns series
    pair_returns = pd.Series(0.0, index=spread_returns.index)
    
    # Simulate all historical trades for this pair
    in_trade = False
    trade_direction = None
    
    for i, date in enumerate(z_scores.index):
        z = z_scores.iloc[i]
        ema = ema_z[i]
        
        if not in_trade:
            # Entry condition: |z_score| < |EMA| (absolute z-score below absolute EMA)
            if abs(z) < abs(ema):
                # Determine direction based on sign of z-score
                if z < 0:
                    # Negative z-score: spread is below mean, go long
                    in_trade = True
                    trade_direction = 'long'
                elif z > 0:
                    # Positive z-score: spread is above mean, go short
                    in_trade = True
                    trade_direction = 'short'
        
        if in_trade:
            # Check exit conditions
            exit_trade = False
            
            if trade_direction == 'long':
                # Long: exit if z crosses above EMA (stop) or reaches take profit
                if abs(z) >= abs(ema):  # Stop loss: crossed back above EMA
                    exit_trade = True
                elif z >= take_profit_z:  # Take profit reached
                    exit_trade = True
            else:  # short
                # Short: exit if z crosses below EMA (stop) or reaches take profit
                if abs(z) >= abs(ema):  # Stop loss: crossed back above EMA
                    exit_trade = True
                elif z <= take_profit_z:  # Take profit reached
                    exit_trade = True
            
            if exit_trade:
                in_trade = False
                trade_direction = None
            else:
                # Position is active - record return
                if date in spread_returns.index and i > 0:
                    if trade_direction == 'long':
                        pair_returns.iloc[i] = spread_returns.iloc[i]
                    else:  # short - invert return
                        pair_returns.iloc[i] = -spread_returns.iloc[i]
    
    # Store returns for this pair
    pair_name = f"{stock1}/{stock2}"
    trade_returns[pair_name] = pair_returns
    
    # Count active trading days
    active_days = (pair_returns != 0).sum()
    print(f"  {pair_name}: {active_days} active trading days (edge: {row['edge']:.4f})")

print(f"\n{'='*80}")
print(f"Building correlation matrix")
print(f"{'='*80}")

# Build dataframe with all pairs' returns aligned to common calendar
df_returns = pd.DataFrame(trade_returns)

# Calculate correlation matrix
corr_matrix = df_returns.corr()

# Save correlation matrix
corr_matrix.to_csv("data/pair_correlation_matrix.csv")

# Calculate statistics
upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

print(f"\nCorrelation matrix saved: {len(corr_matrix)} pairs")
print(f"Based on {len(df_returns)} total trading days")
print(f"\nCorrelation statistics:")
print(f"  Mean correlation: {upper_triangle.mean():.4f}")
print(f"  Median correlation: {np.median(upper_triangle):.4f}")
print(f"  Max correlation: {upper_triangle.max():.4f}")
print(f"  Min correlation: {upper_triangle.min():.4f}")
print(f"  Std correlation: {upper_triangle.std():.4f}")

# Show most and least correlated pairs
if len(corr_matrix) > 1:
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.where(mask)
    
    max_corr = corr_values.max().max()
    max_pair = corr_values.stack().idxmax()
    
    min_corr = corr_values.min().min()
    min_pair = corr_values.stack().idxmin()
    
    print(f"\nMost correlated pairs:")
    print(f"  {max_pair[0]} <-> {max_pair[1]}: {max_corr:.4f}")
    print(f"\nLeast correlated pairs:")
    print(f"  {min_pair[0]} <-> {min_pair[1]}: {min_corr:.4f}")

print(f"\n{'='*80}")