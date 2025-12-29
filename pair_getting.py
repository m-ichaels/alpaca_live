import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Load data
df = pd.read_csv("data/sp500_prices_clean.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

tickers = df.columns.tolist()
prices = df.values

print(f"Analyzing {len(tickers)} stocks with {len(df)} days of data")

# Parameters
max_group_size = 5
min_group_size = 2
significance_level = 0.05
min_trace_stat = 15
n_samples = 10000  # Sample this many combinations per group size

results = []
np.random.seed(42)

for size in range(min_group_size, max_group_size + 1):
    print(f"\nTesting groups of size {size}...")
    print(f"Sampling {n_samples} random combinations...")
    
    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        try:
            # Sample random combination
            combo = np.random.choice(len(tickers), size=size, replace=False)
            group_prices = prices[:, combo]
            
            # Johansen test
            result = coint_johansen(group_prices, det_order=0, k_ar_diff=1)
            
            trace_stats = result.lr1
            critical_values = result.cvt[:, 1]
            n_coint = np.sum(trace_stats > critical_values)
            
            if n_coint > 0 and trace_stats[0] > min_trace_stat:
                group_tickers = [tickers[j] for j in combo]
                eigenvector = result.evec[:, 0]
                
                results.append({
                    'group': ','.join(group_tickers),
                    'size': size,
                    'n_coint': n_coint,
                    'trace_stat': trace_stats[0],
                    'critical_value': critical_values[0],
                    'eigenvector': ','.join([f"{v:.4f}" for v in eigenvector])
                })
        
        except Exception as e:
            continue

print(f"\nFound {len(results)} cointegrated groups")

# Sort by trace statistic
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('trace_stat', ascending=False)

# Remove overlapping groups (keep strongest)
final_groups = []
used_tickers = set()

for _, row in results_df.iterrows():
    group_tickers = set(row['group'].split(','))
    
    if not group_tickers.intersection(used_tickers):
        final_groups.append(row)
        used_tickers.update(group_tickers)

final_df = pd.DataFrame(final_groups)
final_df.to_csv("data/cointegrated_groups.csv", index=False)

print(f"\nSaved {len(final_df)} non-overlapping cointegrated groups")
print(f"\nTop 10 groups by trace statistic:")
print(final_df[['group', 'size', 'n_coint', 'trace_stat']].head(10))