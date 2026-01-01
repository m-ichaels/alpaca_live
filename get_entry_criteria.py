import pandas as pd
import numpy as np

# Load data
prices_df = pd.read_csv("data/sp500_prices_clean.csv")
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df = prices_df.set_index('date')

pairs_df = pd.read_csv("data/cointegrated_pairs.csv")

print(f"Analyzing {len(pairs_df)} cointegrated pairs")

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
        'signal': 'BUY' if z_score < -2 else ('SELL' if z_score > 2 else 'NEUTRAL')
    })

results_df = pd.DataFrame(results)
results_df = results_df[results_df['signal'] != 'NEUTRAL']
results_df = results_df.sort_values('z_score', key=abs, ascending=False)
results_df.to_csv("data/entry_signals.csv", index=False)

print(f"\nTop 10 entry opportunities:")
print(results_df[['stock1', 'stock2', 'z_score', 'half_life', 'signal']].head(10))

print(f"\nSummary:")
print(f"BUY signals: {len(results_df[results_df['signal'] == 'BUY'])}")
print(f"SELL signals: {len(results_df[results_df['signal'] == 'SELL'])}")