import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize, linprog
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET
from tp_sl import TAKE_PROFIT_Z, STOP_LOSS_Z

# Setup
tc = TradingClient(KEY, SECRET, paper=True)
dc = StockHistoricalDataClient(KEY, SECRET)
acct = float(tc.get_account().equity)
held = {p.symbol for p in tc.get_all_positions()}

# Load Data
df = pd.read_csv("data/entry_signals.csv")
px = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)

# Load correlation matrix
try:
    corr_matrix_df = pd.read_csv("data/pair_correlation_matrix.csv", index_col=0)
    print(f"Loaded correlation matrix: {len(corr_matrix_df)} pairs")
except:
    print("ERROR: Correlation matrix not found. Run correlations.py first.")
    exit()

TRADING_DAYS_PER_YEAR = 252
MAX_POSITION_SIZE = 0.25  # Max 25% of capital per position

# 1. Calculate Win Probability
wp = 0.51
if os.path.exists("data/trade_history.csv"):
    try:
        h = pd.read_csv("data/trade_history.csv")
        c = h[h['exit_date'].notna() & h['win'].notna()]
        if len(c) >= 10:
            wp = 0.6 * c['win'].mean() + 0.4 * 0.51
    except:
        pass

# 2. Calculate Expected Returns, Volatilities, and Risk per Trade
pair_metrics = []

for _, r in df.iterrows():
    if r['stock1'] in held or r['stock2'] in held:
        continue
    
    if pd.isna(r['half_life']) or r['half_life'] <= 0 or np.isinf(r['half_life']):
        continue
    
    s = px[r['stock1']] - r['hedge_ratio'] * px[r['stock2']]
    
    # Expected profit/loss distances in z-score units
    if r['z_score'] > 0:
        d_p = abs(r['z_score'] - TAKE_PROFIT_Z)
        d_l = abs(STOP_LOSS_Z - r['z_score'])
    else:
        d_p = abs(r['z_score'] + TAKE_PROFIT_Z)
        d_l = abs(-STOP_LOSS_Z - r['z_score'])
    
    # Convert to dollar distances
    exp_p, exp_l = d_p * s.std(), d_l * s.std()
    wl = exp_p / exp_l if exp_l > 0 else 1.0
    
    # CRITICAL: Calculate risk per trade as fraction of position
    # This is the % of deployed capital we lose if stopped out
    risk_fraction = d_l / abs(r['z_score']) if abs(r['z_score']) > 0 else 0.5
    
    # Sanity check on risk fraction
    if risk_fraction <= 0 or risk_fraction > 1.0:
        continue
    
    # Per-trade return
    per_trade_return = wp * wl - (1 - wp)
    
    # Annualized return
    trades_per_year = TRADING_DAYS_PER_YEAR / r['half_life']
    annualized_return = per_trade_return * trades_per_year
    
    # Annualized volatility
    spread_returns = s.pct_change().dropna()
    trade_vol = spread_returns.std() * np.sqrt(trades_per_year)
    annualized_vol = trade_vol if trade_vol > 0 else 0.01
    
    if annualized_return <= 0 or annualized_vol <= 0 or annualized_vol > 5.0:
        continue
    
    pair_name = f"{r['stock1']}/{r['stock2']}"
    
    pair_metrics.append({
        'pair': pair_name,
        'stock1': r['stock1'],
        'stock2': r['stock2'],
        'signal': r['signal'],
        'z_score': r['z_score'],
        'hedge_ratio': r['hedge_ratio'],
        'half_life': r['half_life'],
        'win_prob': wp,
        'win_loss_ratio': wl,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'risk_fraction': risk_fraction  # NEW: risk as % of position
    })

if not pair_metrics:
    print("No valid pairs found")
    pd.DataFrame(columns=['stock1','stock2','signal','z_score','hedge_ratio','kelly_fraction',
                          'capital_allocation','shares1','shares2','price1','price2','win_prob',
                          'win_loss_ratio','half_life','annualized_return','risk_pct']).to_csv("data/sized_signals.csv", index=False)
    exit()

metrics_df = pd.DataFrame(pair_metrics)
print(f"Analyzing {len(metrics_df)} valid pairs")

print(f"\nPair metrics summary:")
print(f"  Avg annualized return: {metrics_df['annualized_return'].mean():.2%}")
print(f"  Avg annualized vol: {metrics_df['annualized_vol'].mean():.2%}")
print(f"  Avg Sharpe: {(metrics_df['annualized_return'] / metrics_df['annualized_vol']).mean():.2f}")
print(f"  Avg risk per trade: {metrics_df['risk_fraction'].mean():.2%}")

# 3. Build Covariance Matrix from Correlation Matrix
pair_names = metrics_df['pair'].tolist()
n_pairs = len(pair_names)

# Extract correlation submatrix for our pairs
try:
    corr_sub = corr_matrix_df.loc[pair_names, pair_names]
    corr_matrix = corr_sub.values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
except:
    print("WARNING: Could not extract correlations for all pairs, using identity matrix")
    corr_matrix = np.eye(n_pairs)

print(f"Mean correlation: {corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean():.3f}")

# Build covariance matrix
vols = metrics_df['annualized_vol'].values
D = np.diag(vols)
cov_matrix = D @ corr_matrix @ D

# 4. Portfolio-Level Kelly Optimization
# NOTE: These weights represent RISK allocation, not position size
mu = metrics_df['annualized_return'].values

def negative_kelly_objective(w):
    """Negative Kelly criterion (for minimization)"""
    portfolio_return = w @ mu
    portfolio_variance = w @ cov_matrix @ w
    return -(portfolio_return - 0.5 * portfolio_variance)

# Constraints: sum(w) <= 1 (total risk <= 100% of capital)
constraints = [{'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)}]

# Bounds: 0 <= w_i <= 0.15 (max 15% risk per pair)
bounds = [(0, 0.15) for _ in range(n_pairs)]

# Initial guess - proportional to Sharpe ratio
sharpe_ratios = mu / vols
w0 = sharpe_ratios / sharpe_ratios.sum() * 0.5
w0 = np.clip(w0, 0, 0.15)

# Optimize
result = minimize(
    negative_kelly_objective,
    w0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

if not result.success:
    print(f"Optimization failed: {result.message}")
    sharpe_ratios = mu / vols
    sharpe_weights = sharpe_ratios / sharpe_ratios.sum()
    optimal_risk_weights = np.clip(sharpe_weights * 0.95, 0, 0.15)
    print("Using Sharpe-weighted fallback")
else:
    optimal_risk_weights = result.x
    print("Optimization succeeded")

# Apply fractional Kelly (1/3 for safety)
optimal_risk_weights *= 0.33

print(f"\nPortfolio Kelly risk allocation:")
print(f"  Total risk: {optimal_risk_weights.sum():.2%}")
print(f"  Max risk per pair: {optimal_risk_weights.max():.2%}")
print(f"  Min risk per pair: {optimal_risk_weights[optimal_risk_weights > 0].min():.2%}")
print(f"  Non-zero allocations: {(optimal_risk_weights > 0.001).sum()}/{n_pairs}")

# Portfolio metrics
port_return = optimal_risk_weights @ mu
port_vol = np.sqrt(optimal_risk_weights @ cov_matrix @ optimal_risk_weights)
print(f"\nExpected portfolio:")
print(f"  Return: {port_return:.2%}, Vol: {port_vol:.2%}, Sharpe: {port_return/port_vol:.2f}")

# 5. Convert RISK allocation to POSITION SIZE
metrics_df['kelly_risk_weight'] = optimal_risk_weights
metrics_df['risk_capital'] = optimal_risk_weights * acct

# Convert risk to position size: position = risk / risk_fraction
metrics_df['target_capital'] = metrics_df['risk_capital'] / metrics_df['risk_fraction']

# Apply maximum position size constraint
metrics_df['target_capital'] = metrics_df['target_capital'].clip(upper=acct * MAX_POSITION_SIZE)

print(f"\nPosition sizing:")
print(f"  Total risk allocated: ${metrics_df['risk_capital'].sum():,.0f} ({metrics_df['risk_capital'].sum()/acct:.1%})")
print(f"  Total capital deployed: ${metrics_df['target_capital'].sum():,.0f} ({metrics_df['target_capital'].sum()/acct:.1%})")
print(f"  Leverage ratio: {metrics_df['target_capital'].sum() / metrics_df['risk_capital'].sum():.2f}x")

final = []

for _, r in metrics_df.iterrows():
    if r['target_capital'] < acct * 0.01:  # Skip positions < 1% of account
        continue
    
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[r['stock1'], r['stock2']]))
        p1, p2 = (q[r['stock1']].bid_price + q[r['stock1']].ask_price)/2, (q[r['stock2']].bid_price + q[r['stock2']].ask_price)/2
    except:
        continue
    
    # Find feasible allocations
    feas = []
    min_c = max(acct * 0.01, r['target_capital'] * 0.7)
    max_c = min(acct * MAX_POSITION_SIZE, r['target_capital'] * 1.3)
    s1_min = max(1, int(min_c / (p1 * (1 + 1/abs(r['hedge_ratio'])))))
    s1_max = int(max_c / p1) + 10
    
    for s1 in range(s1_min, min(s1_max, 500)):
        ts2 = (s1 * p1) / (abs(r['hedge_ratio']) * p2)
        for s2 in [int(ts2), int(ts2)+1]:
            if s2 < 1:
                continue
            cap = s1*p1 + s2*p2
            hedge_err = abs(s1*p1/(s2*p2) - abs(r['hedge_ratio']))/abs(r['hedge_ratio'])*100
            if min_c <= cap <= max_c and hedge_err <= 5.0:
                if not any(abs(f['capital'] - cap) < 10 for f in feas):
                    feas.append({'shares1': s1, 'shares2': s2, 'capital': cap, 'hedge_error': hedge_err})
    
    if feas:
        best = min(feas, key=lambda x: abs(x['capital'] - r['target_capital']))
        risk_dollars = best['capital'] * r['risk_fraction']
        
        final.append({
            'stock1': r['stock1'], 'stock2': r['stock2'], 'signal': r['signal'],
            'z_score': r['z_score'], 'hedge_ratio': r['hedge_ratio'], 'half_life': r['half_life'],
            'annualized_return': r['annualized_return'], 
            'kelly_fraction': r['kelly_risk_weight'],  # This is the RISK fraction
            'capital_allocation': best['capital'],  # Capital DEPLOYED
            'risk_dollars': risk_dollars,  # Capital at RISK
            'risk_pct': r['risk_fraction'] * 100,  # Risk as % of position
            'shares1': best['shares1'], 'shares2': best['shares2'],
            'price1': p1, 'price2': p2, 'win_prob': r['win_prob'], 'win_loss_ratio': r['win_loss_ratio'],
            'hedge_error': best['hedge_error']
        })

# 6. Output
if final:
    out = pd.DataFrame(final).sort_values('annualized_return', ascending=False)
    out.to_csv("data/sized_signals.csv", index=False)
    
    print(f"\n{'='*70}")
    print(f"Saved {len(out)} allocations")
    print(f"Total RISK: ${out['risk_dollars'].sum():,.0f} ({out['risk_dollars'].sum()/acct:.1%} of account)")
    print(f"Total DEPLOYED: ${out['capital_allocation'].sum():,.0f} ({out['capital_allocation'].sum()/acct:.1%} of account)")
    print(f"Average risk per position: {out['risk_pct'].mean():.1f}%")
    print(f"{'='*70}")
    
    # Show breakdown
    print("\nPosition details:")
    for _, row in out.iterrows():
        print(f"  {row['stock1']}/{row['stock2']}: Deploy ${row['capital_allocation']:,.0f}, Risk ${row['risk_dollars']:,.0f} ({row['risk_pct']:.1f}%)")
else:
    pd.DataFrame(columns=['stock1','stock2','signal','z_score','hedge_ratio','kelly_fraction',
                          'capital_allocation','risk_dollars','risk_pct','shares1','shares2','price1','price2','win_prob',
                          'win_loss_ratio','half_life','annualized_return']).to_csv("data/sized_signals.csv", index=False)