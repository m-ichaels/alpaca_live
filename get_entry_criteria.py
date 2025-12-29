import pandas as pd
import numpy as np
from scipy import stats

# Load data
prices_df = pd.read_csv("data/sp500_prices_clean.csv")
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df = prices_df.set_index('date')

groups_df = pd.read_csv("data/cointegrated_groups.csv")

print(f"Analyzing {len(groups_df)} cointegrated groups")

results = []

for idx, row in groups_df.iterrows():
    tickers = row['group'].split(',')
    weights = np.array([float(x) for x in row['eigenvector'].split(',')])
    
    # Get prices for group
    group_prices = prices_df[tickers].values
    
    # Calculate spread using eigenvector weights
    spread = group_prices @ weights
    
    # Fit OU process: dX = theta * (mu - X) * dt + sigma * dW
    # Using discrete approximation: X(t+1) - X(t) = theta * (mu - X(t)) + epsilon
    
    X = spread[:-1]
    dX = np.diff(spread)
    
    # OLS regression: dX = alpha + beta * X + epsilon
    # where beta = -theta and alpha = theta * mu
    A = np.vstack([np.ones(len(X)), X]).T
    params = np.linalg.lstsq(A, dX, rcond=None)[0]
    alpha, beta = params
    
    theta = -beta
    mu = alpha / theta if theta != 0 else np.mean(spread)
    sigma = np.std(dX)
    
    # Half-life of mean reversion
    half_life = -np.log(2) / beta if beta < 0 else np.inf
    
    # Current spread position
    current_spread = spread[-1]
    z_score = (current_spread - mu) / sigma if sigma > 0 else 0
    
    # Entry score: higher absolute z-score + faster mean reversion = better entry
    # Negative z_score means undervalued (buy signal), positive means overvalued (sell signal)
    if half_life > 0 and half_life < 60:  # Only consider reasonable half-lives
        entry_score = abs(z_score) * (30 / half_life)  # Normalize by half-life
    else:
        entry_score = 0
    
    results.append({
        'group': row['group'],
        'size': row['size'],
        'trace_stat': row['trace_stat'],
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life': half_life,
        'current_spread': current_spread,
        'z_score': z_score,
        'entry_score': entry_score,
        'signal': 'BUY' if z_score < -1.5 else ('SELL' if z_score > 1.5 else 'NEUTRAL')
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('entry_score', ascending=False)
results_df.to_csv("data/ou_entry_scores.csv", index=False)

print(f"\nTop 10 entry opportunities:")
print(results_df[['group', 'size', 'z_score', 'half_life', 'entry_score', 'signal']].head(10))

print(f"\nSummary:")
print(f"BUY signals: {len(results_df[results_df['signal'] == 'BUY'])}")
print(f"SELL signals: {len(results_df[results_df['signal'] == 'SELL'])}")
print(f"NEUTRAL: {len(results_df[results_df['signal'] == 'NEUTRAL'])}")