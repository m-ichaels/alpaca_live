import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# Load data
df_pairs = pd.read_csv("data/pairs_with_edge.csv")
df_corr = pd.read_csv("data/pair_correlation_matrix.csv", index_col=0)

print(f"{'='*80}")
print(f"POSITION SIZING - Mean-Variance Optimization with Kelly Criterion")
print(f"{'='*80}")

# Filter to only positive edge pairs
df_pairs = df_pairs[df_pairs['edge'] > 0].copy()
print(f"\nPositive edge pairs: {len(df_pairs)}")

if len(df_pairs) == 0:
    print("No positive edge pairs to size. Exiting.")
    pd.DataFrame().to_csv("data/final_positions.csv", index=False)
    exit()

# Build pair names matching correlation matrix
df_pairs['pair_name'] = df_pairs['stock1'] + '/' + df_pairs['stock2']

# Align with correlation matrix
valid_pairs = [p for p in df_pairs['pair_name'] if p in df_corr.index]
df_pairs = df_pairs[df_pairs['pair_name'].isin(valid_pairs)].reset_index(drop=True)

print(f"Pairs with correlation data: {len(df_pairs)}")

# Extract correlation matrix for valid pairs
corr_matrix = df_corr.loc[valid_pairs, valid_pairs].values

# Calculate expected returns (mu) from edge values
mu = df_pairs['edge'].values

# Calculate volatilities (std dev) from edge components
sigma = df_pairs['spread_std'].values

# Build covariance matrix from correlation matrix
D = np.diag(sigma)
cov_matrix = D @ corr_matrix @ D

print(f"\n{'='*80}")
print(f"STEP 1: Relative Position Sizing (Mean-Variance Optimization)")
print(f"{'='*80}")

# Continuous optimization to find optimal weights
# Maximize Sharpe ratio: max w^T*mu / sqrt(w^T*Sigma*w)
# Subject to: sum(w) = 1, w >= 0

# Track progress with callback
pbar = None

def negative_sharpe(w, mu, cov):
    """Negative Sharpe ratio for minimization"""
    global pbar
    if pbar is not None:
        pbar.update(1)
    
    portfolio_return = np.dot(w, mu)
    portfolio_variance = np.dot(w, np.dot(cov, w))
    portfolio_std = np.sqrt(portfolio_variance)
    if portfolio_std == 0:
        return 1e10
    return -portfolio_return / portfolio_std

# Initial guess: equal weights
w0 = np.ones(len(df_pairs)) / len(df_pairs)

# Constraints: weights sum to 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

# Bounds: no short selling (weights >= 0, <= 1)
bounds = tuple((0.0, 1.0) for _ in range(len(df_pairs)))

print(f"Running mean-variance optimization...")
print(f"  Number of pairs: {len(df_pairs)}")

# Create progress bar
pbar = tqdm(total=1000, desc="  Optimizing", unit="iter")

result = minimize(
    negative_sharpe,
    w0,
    args=(mu, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-9}
)

pbar.close()
pbar = None

if result.success:
    w_optimal = result.x
    sharpe = -result.fun
    print(f"  Optimization successful")
    print(f"  Portfolio Sharpe ratio: {sharpe:.4f}")
else:
    print(f"  Optimization warning: {result.message}")
    print(f"  Using result anyway")
    w_optimal = result.x

# Remove pairs with negligible weights (< 0.1%)
threshold = 0.001
significant_mask = w_optimal >= threshold
w_filtered = w_optimal[significant_mask]
df_pairs = df_pairs[significant_mask].reset_index(drop=True)

# Renormalize weights
w_filtered = w_filtered / w_filtered.sum()

print(f"  Pairs after filtering (w >= {threshold:.1%}): {len(df_pairs)}")
print(f"  Max weight: {w_filtered.max():.4f}")
print(f"  Min weight: {w_filtered.min():.4f}")

df_pairs['weight'] = w_filtered

print(f"\n{'='*80}")
print(f"STEP 2: Absolute Position Sizing (Portfolio Kelly Criterion)")
print(f"{'='*80}")

# Calculate portfolio-level parameters for Kelly
p_portfolio = np.sum(w_filtered * df_pairs['prob_win'].values)
avg_win = np.sum(w_filtered * df_pairs['tp_magnitude'].values)
avg_loss = np.sum(w_filtered * df_pairs['sl_magnitude'].values)
b = avg_win / avg_loss if avg_loss > 0 else 1.0

# Kelly fraction
q = 1 - p_portfolio
f_kelly = (b * p_portfolio - q) / b

# Fractional Kelly
kappa = 0.5
f_actual = kappa * f_kelly
f_actual = np.clip(f_actual, 0.0, 1.0)

print(f"Portfolio Kelly parameters:")
print(f"  Win probability (p): {p_portfolio:.4f}")
print(f"  Net odds (b): {b:.4f}")
print(f"  Full Kelly: {f_kelly:.4f}")
print(f"  Fractional Kelly (κ={kappa}): {f_actual:.4f}")

print(f"\n{'='*80}")
print(f"STEP 3: Iterative Refinement for Discrete Constraints")
print(f"{'='*80}")

# Account capital
CAPITAL = 100000  # $100k

# Get latest prices
df_prices = pd.read_csv("data/sp500_prices_clean.csv")
df_prices['date'] = pd.to_datetime(df_prices['date'])
df_prices = df_prices.set_index('date')
latest_prices = df_prices.iloc[-1]

df_pairs['price_stock1'] = df_pairs['stock1'].map(latest_prices)
df_pairs['price_stock2'] = df_pairs['stock2'].map(latest_prices)
df_pairs['position_size_per_unit'] = (
    df_pairs['price_stock1'] + 
    np.abs(df_pairs['hedge_ratio']) * df_pairs['price_stock2']
)

iteration = 0
max_iterations = 5

while iteration < max_iterations:
    iteration += 1
    print(f"\nIteration {iteration}:")
    
    # Scale positions by Kelly fraction
    df_pairs['shares'] = np.floor(
        df_pairs['weight'] * f_actual * CAPITAL / df_pairs['position_size_per_unit']
    ).astype(int)
    
    # Remove pairs with less than 1 share
    viable_mask = df_pairs['shares'] >= 1
    n_removed = (~viable_mask).sum()
    
    if n_removed == 0:
        print(f"  All {len(df_pairs)} pairs viable (shares >= 1)")
        break
    
    print(f"  Removed {n_removed} pairs (shares < 1)")
    
    # Filter to viable pairs
    df_pairs = df_pairs[viable_mask].reset_index(drop=True)
    
    if len(df_pairs) == 0:
        print("  No viable pairs remaining!")
        break
    
    # Re-optimize with reduced universe
    print(f"  Re-optimizing with {len(df_pairs)} pairs...")
    
    valid_pairs = df_pairs['pair_name'].tolist()
    corr_matrix_new = df_corr.loc[valid_pairs, valid_pairs].values
    mu_new = df_pairs['edge'].values
    sigma_new = df_pairs['spread_std'].values
    D_new = np.diag(sigma_new)
    cov_matrix_new = D_new @ corr_matrix_new @ D_new
    
    w0_new = np.ones(len(df_pairs)) / len(df_pairs)
    constraints_new = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds_new = tuple((0.0, 1.0) for _ in range(len(df_pairs)))
    
    # Progress bar for re-optimization
    pbar = tqdm(total=1000, desc="  Re-optimizing", unit="iter", leave=False)
    
    result_new = minimize(
        negative_sharpe,
        w0_new,
        args=(mu_new, cov_matrix_new),
        method='SLSQP',
        bounds=bounds_new,
        constraints=constraints_new,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    pbar.close()
    pbar = None
    
    w_optimal_new = result_new.x
    df_pairs['weight'] = w_optimal_new
    
    # Recalculate Kelly
    p_portfolio = np.sum(w_optimal_new * df_pairs['prob_win'].values)
    avg_win = np.sum(w_optimal_new * df_pairs['tp_magnitude'].values)
    avg_loss = np.sum(w_optimal_new * df_pairs['sl_magnitude'].values)
    b = avg_win / avg_loss if avg_loss > 0 else 1.0
    q = 1 - p_portfolio
    f_kelly = (b * p_portfolio - q) / b
    f_actual = kappa * f_kelly
    f_actual = np.clip(f_actual, 0.0, 1.0)
    
    print(f"  Re-optimized: {len(df_pairs)} pairs, Kelly={f_actual:.4f}")

print(f"\n{'='*80}")
print(f"FINAL PORTFOLIO")
print(f"{'='*80}")

if len(df_pairs) > 0:
    # Final positions
    df_pairs['capital_per_pair'] = df_pairs['shares'] * df_pairs['position_size_per_unit']
    total_capital_risk = df_pairs['capital_per_pair'].sum()
    
    # Sort by capital allocated
    df_final = df_pairs.sort_values('capital_per_pair', ascending=False)
    
    # Save
    df_final.to_csv("data/final_positions.csv", index=False)
    
    print(f"\nPortfolio Summary:")
    print(f"  Account capital: ${CAPITAL:,.2f}")
    print(f"  Total capital at risk: ${total_capital_risk:,.2f}")
    print(f"  Leverage: {total_capital_risk/CAPITAL:.2f}x")
    print(f"  Number of pairs: {len(df_final)}")
    print(f"  Average shares per pair: {df_final['shares'].mean():.1f}")
    print(f"  Portfolio win probability: {p_portfolio:.4f}")
    print(f"  Weighted average edge: {np.sum(df_final['weight'] * df_final['edge']):.4f}")
    
    print(f"\nTop 10 positions by capital:")
    print(df_final[['stock1', 'stock2', 'shares', 'weight', 'capital_per_pair', 'edge']].head(10).to_string(index=False))
    
    print(f"\nSaved to data/final_positions.csv")
else:
    print("\nNo viable positions after iterative refinement")
    pd.DataFrame().to_csv("data/final_positions.csv", index=False)

print(f"\n{'='*80}")