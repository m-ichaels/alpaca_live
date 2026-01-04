import pandas as pd
import numpy as np
import os

# Entry criteria (not from tp_sl.py - these are entry-specific)
ENTRY_Z_MIN = 2.5  # Minimum z-score magnitude to trigger entry signal
ENTRY_Z_MAX = 3.0  # Maximum z-score magnitude to enter (beyond this is too risky)

# Load data
prices_df = pd.read_csv("data/sp500_prices_clean.csv")
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df = prices_df.set_index('date')

pairs_df = pd.read_csv("data/cointegrated_pairs.csv")

print(f"Analyzing {len(pairs_df)} cointegrated pairs")
print(f"Entry range: {ENTRY_Z_MIN} <= |z-score| < {ENTRY_Z_MAX}")

results = []

for idx, row in pairs_df.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    beta = row['hedge_ratio']
    
    # Calculate spread: stock1 - beta * stock2
    spread = prices_df[stock1] - beta * prices_df[stock2]
    
    # Calculate z-score
    mu = spread.mean()
    sigma = spread.std()
    current_spread = spread.iloc[-1]
    z_score = (current_spread - mu) / sigma
    
    # Calculate half-life
    spread_diff = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()
    
    # Align arrays
    spread_lag = spread_lag[spread_diff.index]
    
    X = spread_lag.values
    y = spread_diff.values
    
    beta_hl = np.polyfit(X, y, 1)[0]
    half_life = -np.log(2) / beta_hl if beta_hl < 0 else np.inf
    
    results.append({
        'stock1': stock1,
        'stock2': stock2,
        'hedge_ratio': beta,
        'z_score': z_score,
        'half_life': half_life,
        'signal': 'BUY' if z_score < -ENTRY_Z_MIN else ('SELL' if z_score > ENTRY_Z_MIN else 'NEUTRAL')
    })

results_df = pd.DataFrame(results)
results_df = results_df[results_df['signal'] != 'NEUTRAL']
results_df = results_df[abs(results_df['z_score']) < ENTRY_Z_MAX]  # Filter out signals too extreme to enter
results_df = results_df.sort_values('z_score', key=abs, ascending=False)

print(f"\nInitial signals found: {len(results_df)}")
print(f"BUY signals: {len(results_df[results_df['signal'] == 'BUY'])}")
print(f"SELL signals: {len(results_df[results_df['signal'] == 'SELL'])}")

# Filter out pairs that are already tracked as open
TRACKER_FILE = "data/open_pairs.csv"

try:
    if os.path.exists(TRACKER_FILE):
        tracker = pd.read_csv(TRACKER_FILE)
        open_tracker = tracker[tracker['status'] == 'open']
        
        if len(open_tracker) > 0:
            # Create set of open pairs (both orderings)
            open_pairs = set()
            for _, row in open_tracker.iterrows():
                open_pairs.add((row['stock1'], row['stock2']))
                open_pairs.add((row['stock2'], row['stock1']))  # Add reverse too
            
            print(f"\nOpen pairs detected in tracker: {len(open_tracker)}")
            
            # Filter function to check if this exact pair is already open
            def pair_is_open(row):
                return (row['stock1'], row['stock2']) in open_pairs
            
            # Count how many signals we're filtering out
            filtered_out = results_df[results_df.apply(pair_is_open, axis=1)]
            if len(filtered_out) > 0:
                print(f"\nFiltering out {len(filtered_out)} pairs that are already open:")
                for _, row in filtered_out.iterrows():
                    print(f"  - {row['stock1']}/{row['stock2']} (exact pair already tracked)")
            
            # Apply filter
            results_df = results_df[~results_df.apply(pair_is_open, axis=1)]
            
            print(f"\nAfter filtering tracked pairs: {len(results_df)} signals remain")
        else:
            print(f"\nNo open pairs in tracker - all signals remain valid")
    else:
        print(f"\nNo tracker file found - all signals remain valid")
        
except Exception as e:
    print(f"\n⚠️  Warning: Could not check tracked pairs - {e}")
    print("Proceeding with all signals (no filtering applied)")

# Save filtered results
results_df.to_csv("data/entry_signals.csv", index=False)

print(f"\nTop 10 entry opportunities:")
print(results_df[['stock1', 'stock2', 'z_score', 'half_life', 'signal']].head(10))

print(f"\nFinal Summary:")
print(f"BUY signals: {len(results_df[results_df['signal'] == 'BUY'])}")
print(f"SELL signals: {len(results_df[results_df['signal'] == 'SELL'])}")
print(f"\nSaved to data/entry_signals.csv")