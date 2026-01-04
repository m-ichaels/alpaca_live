import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
P = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)
S = pd.read_csv("data/entry_signals.csv")

if S.empty:
    print("No entry signals found")
    exit()

# Calculate z-scores for all pairs
Z = {}
for _, r in S.iterrows():
    if r['stock1'] in P.columns and r['stock2'] in P.columns:
        spread = P[r['stock1']] - r['hedge_ratio'] * P[r['stock2']]
        z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        pair_name = f"{r['stock1']}/{r['stock2']}"
        Z[pair_name] = z.dropna()

if len(Z) < 1:
    print("No valid pairs for correlation analysis")
    # Create empty correlation matrix
    pd.DataFrame().to_csv("data/pair_correlation_matrix.csv")
    exit()

# Build z-score dataframe and correlation matrix
df = pd.DataFrame(Z).dropna()
corr = df.corr()

# Save correlation matrix
corr.to_csv("data/pair_correlation_matrix.csv")

print(f"Correlation matrix saved: {len(corr)} pairs, {len(df)} overlapping days")
print(f"Mean correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].mean():.3f}")