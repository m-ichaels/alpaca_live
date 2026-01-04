import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PAIR CORRELATION ANALYSIS")
print("=" * 80)

# Load historical prices and current signals
prices_df = pd.read_csv("data/sp500_prices_clean.csv")
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df = prices_df.set_index('date')

signals_df = pd.read_csv("data/entry_signals.csv")

if len(signals_df) == 0:
    print("\nNo signals to analyze")
    exit(0)

print(f"\nAnalyzing {len(signals_df)} pairs...")

# Calculate z-score time series for each pair
pair_zscore_series = {}

for idx, row in signals_df.iterrows():
    stock1, stock2 = row['stock1'], row['stock2']
    beta = row['hedge_ratio']
    
    if stock1 not in prices_df.columns or stock2 not in prices_df.columns:
        continue
    
    # Calculate spread
    spread = prices_df[stock1] - beta * prices_df[stock2]
    
    # Rolling z-score (60-day window)
    window = 60
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    zscore_series = (spread - rolling_mean) / rolling_std
    zscore_series = zscore_series.dropna()
    
    if len(zscore_series) > 0:
        pair_zscore_series[f"{stock1}/{stock2}"] = zscore_series

print(f"Successfully calculated z-scores for {len(pair_zscore_series)} pairs\n")

# Calculate correlation matrix
print("=" * 80)
print("CORRELATION MATRIX")
print("=" * 80)

pair_names = list(pair_zscore_series.keys())
n_pairs = len(pair_names)

if n_pairs < 2:
    print("\nNeed at least 2 pairs to calculate correlations")
    exit(0)

# Align all time series to common dates
aligned_data = pd.DataFrame({name: series for name, series in pair_zscore_series.items()})
aligned_data = aligned_data.dropna()

if len(aligned_data) < 30:
    print(f"\nWarning: Only {len(aligned_data)} overlapping days - correlations may be unreliable")

# Calculate correlation matrix
corr_matrix = aligned_data.corr()

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True if n_pairs <= 10 else False,  # Only annotate if <= 10 pairs
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation'})
plt.title('Z-Score Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Save full correlation matrix
corr_matrix.to_csv("data/zscore_correlations.csv")
print(f"Full correlation matrix saved to data/zscore_correlations.csv")
print(f"Correlation heatmap saved to data/correlation_heatmap.png")

# Find highly correlated pairs (|corr| > 0.5)
print("\n" + "=" * 80)
print("HIGH CORRELATIONS (|ρ| > 0.5)")
print("=" * 80)

high_corr_pairs = []

for i in range(n_pairs):
    for j in range(i + 1, n_pairs):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.5:
            high_corr_pairs.append({
                'pair1': pair_names[i],
                'pair2': pair_names[j],
                'correlation': corr
            })

if len(high_corr_pairs) > 0:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
    
    for idx, row in high_corr_df.iterrows():
        print(f"  {row['pair1']} <-> {row['pair2']}: {row['correlation']:+.3f}")
else:
    print("  No pairs with |correlation| > 0.5")

# Calculate portfolio concentration risk
print("\n" + "=" * 80)
print("PORTFOLIO CONCENTRATION ANALYSIS")
print("=" * 80)

# Load current sized signals if they exist
try:
    sized_df = pd.read_csv("data/sized_signals.csv")
    
    # Filter correlation matrix to only pairs we're trading
    trading_pairs = [f"{row['stock1']}/{row['stock2']}" for _, row in sized_df.iterrows()]
    trading_pairs_in_corr = [p for p in trading_pairs if p in pair_names]
    
    if len(trading_pairs_in_corr) > 1:
        trading_corr = corr_matrix.loc[trading_pairs_in_corr, trading_pairs_in_corr]
        
        # Calculate average absolute correlation
        avg_corr = trading_corr.abs().sum().sum() / (len(trading_pairs_in_corr) ** 2 - len(trading_pairs_in_corr))
        
        print(f"Pairs in portfolio: {len(trading_pairs_in_corr)}")
        print(f"Average absolute correlation: {avg_corr:.3f}")
        
        # Effective number of independent bets (diversification ratio)
        # ENB = n / (1 + (n-1)*avg_corr)
        n = len(trading_pairs_in_corr)
        enb = n / (1 + (n - 1) * avg_corr)
        
        print(f"Effective number of bets: {enb:.2f} (out of {n} pairs)")
        print(f"Diversification efficiency: {enb/n:.1%}")
        
        # Identify clusters of correlated positions
        print("\nHighly correlated positions in portfolio:")
        found_cluster = False
        for i in range(len(trading_pairs_in_corr)):
            for j in range(i + 1, len(trading_pairs_in_corr)):
                corr = trading_corr.iloc[i, j]
                if abs(corr) > 0.5:
                    found_cluster = True
                    print(f"  ⚠  {trading_pairs_in_corr[i]} <-> {trading_pairs_in_corr[j]}: {corr:+.3f}")
        
        if not found_cluster:
            print("  No highly correlated pairs (good diversification)")
        
        # Calculate concentration-adjusted Kelly fractions
        print("\n" + "=" * 80)
        print("CONCENTRATION-ADJUSTED SIZING RECOMMENDATIONS")
        print("=" * 80)
        
        # Map pair names back to sized_df
        pair_to_capital = {}
        for _, row in sized_df.iterrows():
            pair_name = f"{row['stock1']}/{row['stock2']}"
            pair_to_capital[pair_name] = {
                'stock1': row['stock1'],
                'stock2': row['stock2'],
                'current_fraction': row['kelly_fraction'],
                'capital': row['capital_allocation']
            }
        
        adjustments = []
        
        for pair in trading_pairs_in_corr:
            if pair not in pair_to_capital:
                continue
            
            # Calculate correlation-based adjustment
            # Higher average correlation with other positions = reduce size
            pair_corrs = trading_corr.loc[pair].drop(pair)
            avg_pair_corr = pair_corrs.abs().mean()
            
            # Adjustment factor: reduce by correlation intensity
            # 0 correlation = no adjustment (1.0x)
            # High correlation = reduce (e.g., 0.7x at avg_corr=0.6)
            adjustment_factor = 1.0 - (avg_pair_corr * 0.5)  # Max 50% reduction
            adjustment_factor = max(0.5, adjustment_factor)  # Floor at 50%
            
            info = pair_to_capital[pair]
            adjusted_fraction = info['current_fraction'] * adjustment_factor
            
            adjustments.append({
                'pair': pair,
                'stock1': info['stock1'],
                'stock2': info['stock2'],
                'avg_correlation': avg_pair_corr,
                'current_fraction': info['current_fraction'],
                'adjustment_factor': adjustment_factor,
                'adjusted_fraction': adjusted_fraction,
                'current_capital': info['capital'],
                'suggested_capital': info['capital'] * adjustment_factor
            })
        
        adj_df = pd.DataFrame(adjustments)
        adj_df = adj_df.sort_values('avg_correlation', ascending=False)
        
        print(f"\nSuggested adjustments based on correlation:")
        for _, row in adj_df.iterrows():
            if row['adjustment_factor'] < 0.95:  # Only show significant adjustments
                print(f"\n  {row['pair']}:")
                print(f"    Avg correlation: {row['avg_correlation']:.3f}")
                print(f"    Current: {row['current_fraction']:.2%} (${row['current_capital']:,.0f})")
                print(f"    Suggested: {row['adjusted_fraction']:.2%} (${row['suggested_capital']:,.0f})")
                print(f"    Reduction: {(1 - row['adjustment_factor']) * 100:.1f}%")
        
        # Calculate new total allocation
        total_current = adj_df['current_capital'].sum()
        total_suggested = adj_df['suggested_capital'].sum()
        freed_capital = total_current - total_suggested
        
        if freed_capital > 0:
            print(f"\nCapital freed by adjustments: ${freed_capital:,.0f}")
            print(f"  This could be reallocated to less correlated pairs")
    else:
        print("Need at least 2 pairs in portfolio for concentration analysis")
        
except FileNotFoundError:
    print("No sized_signals.csv found - run sizing.py first")

# Stock overlap analysis
print("\n" + "=" * 80)
print("STOCK OVERLAP ANALYSIS")
print("=" * 80)

stock_pairs = {}
for idx, row in signals_df.iterrows():
    for stock in [row['stock1'], row['stock2']]:
        if stock not in stock_pairs:
            stock_pairs[stock] = []
        stock_pairs[stock].append(f"{row['stock1']}/{row['stock2']}")

overlapping = {stock: pairs for stock, pairs in stock_pairs.items() if len(pairs) > 1}

if overlapping:
    print("Stocks appearing in multiple pairs:")
    for stock, pairs in sorted(overlapping.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n  {stock} appears in {len(pairs)} pairs:")
        for pair in pairs[:5]:  # Show max 5
            print(f"    - {pair}")
        if len(pairs) > 5:
            print(f"    ... and {len(pairs) - 5} more")
else:
    print("No stock overlap - all pairs use unique stocks")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - data/zscore_correlations.csv (full matrix)")
print("  - data/correlation_heatmap.png")