import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

# Load data
df = pd.read_csv("data/sp500_prices_clean.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

tickers = df.columns.tolist()
n_tickers = len(tickers)

print(f"Analyzing {n_tickers} stocks")
print(f"Testing all pairs: {n_tickers * (n_tickers - 1) // 2} combinations")

results = []

for i in range(n_tickers):
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{n_tickers}")
    
    for j in range(i + 1, n_tickers):
        try:
            score, pvalue, _ = coint(df.iloc[:, i], df.iloc[:, j])
            
            if pvalue < 0.05:
                # Calculate hedge ratio
                beta = np.polyfit(df.iloc[:, j], df.iloc[:, i], 1)[0]
                
                results.append({
                    'stock1': tickers[i],
                    'stock2': tickers[j],
                    'hedge_ratio': beta,
                    'pvalue': pvalue,
                    'test_stat': score
                })
        
        except Exception:
            continue

print(f"\nFound {len(results)} cointegrated pairs")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('pvalue')
results_df.to_csv("data/cointegrated_pairs.csv", index=False)

print(f"\nTop 10 pairs by p-value:")
print(results_df[['stock1', 'stock2', 'pvalue', 'test_stat']].head(10))