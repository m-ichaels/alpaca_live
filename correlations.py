import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

# 1. Load Data & Calculate Z-Scores
P = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)
S = pd.read_csv("data/entry_signals.csv")
if S.empty: exit()

Z = {}
for _, r in S.iterrows():
    if r.stock1 in P and r.stock2 in P:
        # Replicates original spread and 60-day rolling z-score logic
        s = P[r.stock1] - r.hedge_ratio * P[r.stock2]
        z = (s - s.rolling(60).mean()) / s.rolling(60).std()
        Z[f"{r.stock1}/{r.stock2}"] = z.dropna()

df = pd.DataFrame(Z).dropna()
if df.shape[1] < 2: exit()

# 2. Generate Correlation Matrix & Heatmap
corr = df.corr()
corr.to_csv("data/zscore_correlations.csv")

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=len(corr)<=10, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True)
plt.title('Z-Score Correlation Heatmap'); plt.tight_layout()
plt.savefig('data/correlation_heatmap.png', dpi=300, bbox_inches='tight'); plt.close()

# 3. Basic Overlap Analysis
stocks = pd.concat([S.stock1, S.stock2])
overlaps = list(stocks[stocks.duplicated()].unique())
print(f"Analysis Complete. Pairs: {len(Z)} | Overlapping Days: {len(df)} | Stock Overlaps: {overlaps}")