import pandas as pd
import numpy as np
<<<<<<< HEAD
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET
from tp_sl import TAKE_PROFIT_Z, STOP_LOSS_Z

# Setup Alpaca
tc = TradingClient(KEY, SECRET, paper=True)
dc = StockHistoricalDataClient(KEY, SECRET)
acct = float(tc.get_account().equity)
held = {p.symbol for p in tc.get_all_positions()}

print(f"Account equity: ${acct:,.2f}")
print(f"Currently held positions: {len(held)}")

# Load optimized portfolio from diversification.py
try:
    portfolio_df = pd.read_csv("data/optimized_portfolio.csv")
    print(f"\nLoaded {len(portfolio_df)} pairs from optimized_portfolio.csv")
except FileNotFoundError:
    print("ERROR: optimized_portfolio.csv not found. Run diversification.py first.")
    exit()

# Load edge analysis data for additional metrics
try:
    edges_df = pd.read_csv("data/pair_edges.csv")
    edges_dict = edges_df.set_index(['stock1', 'stock2']).to_dict('index')
except FileNotFoundError:
    print("ERROR: pair_edges.csv not found. Run edge.py first.")
    exit()

# Load price data for spread calculations
try:
    prices_df = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)
except FileNotFoundError:
    print("ERROR: sp500_prices_clean.csv not found.")
    exit()

# Constants
MAX_POSITION_SIZE = 0.25  # Max 25% of capital per position
MIN_POSITION_SIZE = 0.01  # Min 1% of capital per position
MAX_HEDGE_ERROR = 5.0  # Max 5% hedge ratio error
KELLY_FRACTION = 1.0  # Full Kelly (can adjust to 0.5 for half Kelly, 0.25 for quarter Kelly)

# Filter out pairs with zero weight and already held positions
portfolio_df = portfolio_df[portfolio_df['weight'] > 1e-6].copy()
portfolio_df = portfolio_df[~portfolio_df['stock1'].isin(held) & ~portfolio_df['stock2'].isin(held)].copy()

if len(portfolio_df) == 0:
    print("No valid pairs to trade after filtering.")
    pd.DataFrame().to_csv("data/kelly_positions.csv", index=False)
    exit()

print(f"Valid pairs after filtering: {len(portfolio_df)}")

# Merge with edge data
portfolio_df['edge_key'] = list(zip(portfolio_df['stock1'], portfolio_df['stock2']))
portfolio_df['z_score'] = portfolio_df['edge_key'].apply(lambda x: edges_dict.get(x, {}).get('z_score', np.nan))
portfolio_df['p_target'] = portfolio_df['edge_key'].apply(lambda x: edges_dict.get(x, {}).get('p_target', np.nan))
portfolio_df['p_stop'] = portfolio_df['edge_key'].apply(lambda x: edges_dict.get(x, {}).get('p_stop', np.nan))
portfolio_df['signal'] = portfolio_df['edge_key'].apply(lambda x: edges_dict.get(x, {}).get('signal', ''))
portfolio_df['edge'] = portfolio_df['edge_key'].apply(lambda x: edges_dict.get(x, {}).get('edge', 0))
portfolio_df['theta'] = portfolio_df['edge_key'].apply(lambda x: edges_dict.get(x, {}).get('theta', np.nan))
portfolio_df['volatility'] = portfolio_df['edge_key'].apply(lambda x: edges_dict.get(x, {}).get('volatility', np.nan))

# Drop pairs with missing data
portfolio_df = portfolio_df.dropna(subset=['z_score', 'p_target', 'signal']).copy()

if len(portfolio_df) == 0:
    print("No pairs with complete edge data.")
    pd.DataFrame().to_csv("data/kelly_positions.csv", index=False)
    exit()

print(f"Pairs with complete data: {len(portfolio_df)}")

# Calculate preliminary metrics to determine Kelly criterion
print("\nCalculating portfolio metrics for Kelly sizing...")

preliminary_positions = []

for idx, row in portfolio_df.iterrows():
    # Calculate spread statistics
    spread = prices_df[row['stock1']] - row['hedge_ratio'] * prices_df[row['stock2']]
    spread_std = spread.std()
    
    # Calculate distance to take-profit and stop-loss in z-score units
    if row['z_score'] > 0:  # Short spread
        dist_profit = abs(row['z_score'] - TAKE_PROFIT_Z)
        dist_loss = abs(STOP_LOSS_Z - row['z_score'])
    else:  # Long spread
        dist_profit = abs(row['z_score'] + TAKE_PROFIT_Z)
        dist_loss = abs(-STOP_LOSS_Z - row['z_score'])
    
    # Convert to dollar distances
    expected_profit_dollars = dist_profit * spread_std
    expected_loss_dollars = dist_loss * spread_std
    
    preliminary_positions.append({
        'stock1': row['stock1'],
        'stock2': row['stock2'],
        'weight': row['weight'],
        'capital': row['capital'],
        'p_target': row['p_target'],
        'expected_profit': expected_profit_dollars,
        'expected_loss': expected_loss_dollars
    })

prelim_df = pd.DataFrame(preliminary_positions)

# Calculate portfolio-level metrics
total_expected_profit = prelim_df['expected_profit'].sum()
total_expected_loss = prelim_df['expected_loss'].sum()
portfolio_reward_risk = total_expected_profit / total_expected_loss if total_expected_loss > 0 else 0
portfolio_win_prob = (prelim_df['p_target'] * prelim_df['capital']).sum() / prelim_df['capital'].sum()

# Calculate Full Kelly
# f = (p * b - q) / b
# where p = win prob, q = loss prob, b = reward/risk ratio
kelly_fraction = (portfolio_win_prob * portfolio_reward_risk - (1 - portfolio_win_prob)) / portfolio_reward_risk if portfolio_reward_risk > 0 else 0
kelly_fraction = max(0, min(kelly_fraction, 1.0))  # Clamp between 0 and 1

# Apply Kelly fraction modifier (1.0 for full, 0.5 for half, etc.)
adjusted_kelly = kelly_fraction * KELLY_FRACTION

print(f"\nPortfolio metrics (before Kelly adjustment):")
print(f"  Reward/Risk Ratio: {portfolio_reward_risk:.3f}")
print(f"  Expected Win Probability: {portfolio_win_prob:.2%}")
print(f"  Full Kelly fraction: {kelly_fraction:.2%}")
print(f"  Using Kelly multiplier: {KELLY_FRACTION}x")
print(f"  Target risk level: {adjusted_kelly:.2%} of account (${adjusted_kelly * acct:,.2f})")

# We'll scale positions to achieve the target risk level
# First pass: scale by Kelly to get initial targets
portfolio_df['target_capital'] = portfolio_df['capital'] * adjusted_kelly

# Clip to position size limits
portfolio_df['target_capital'] = portfolio_df['target_capital'].clip(
    lower=acct * MIN_POSITION_SIZE,
    upper=acct * MAX_POSITION_SIZE
)

print(f"\nCapital allocation (after Kelly adjustment):")
print(f"  Total target: ${portfolio_df['target_capital'].sum():,.2f} ({portfolio_df['target_capital'].sum()/acct:.1%})")
print(f"  Mean per pair: ${portfolio_df['target_capital'].mean():,.2f}")
print(f"  Max per pair: ${portfolio_df['target_capital'].max():,.2f}")

# Calculate final positions with actual share allocations
final_positions = []

for idx, row in portfolio_df.iterrows():
    try:
        # Get current prices
        quote = dc.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[row['stock1'], row['stock2']])
        )
        price1 = (quote[row['stock1']].bid_price + quote[row['stock1']].ask_price) / 2
        price2 = (quote[row['stock2']].bid_price + quote[row['stock2']].ask_price) / 2
        
        if price1 <= 0 or price2 <= 0:
            print(f"  Skipping {row['stock1']}/{row['stock2']}: Invalid prices")
            continue
            
    except Exception as e:
        print(f"  Error getting quotes for {row['stock1']}/{row['stock2']}: {e}")
        continue
    
    # Calculate spread statistics for risk assessment
    spread = prices_df[row['stock1']] - row['hedge_ratio'] * prices_df[row['stock2']]
    spread_std = spread.std()
    
    # Calculate distance to take-profit and stop-loss in z-score units
    if row['z_score'] > 0:  # Short spread
        dist_profit = abs(row['z_score'] - TAKE_PROFIT_Z)
        dist_loss = abs(STOP_LOSS_Z - row['z_score'])
    else:  # Long spread
        dist_profit = abs(row['z_score'] + TAKE_PROFIT_Z)
        dist_loss = abs(-STOP_LOSS_Z - row['z_score'])
    
    # Convert to dollar distances
    expected_profit_dollars = dist_profit * spread_std
    expected_loss_dollars = dist_loss * spread_std
    
    # Risk fraction: % of position at risk if stopped out
    risk_fraction = dist_loss / abs(row['z_score']) if abs(row['z_score']) > 0 else 0.5
    risk_fraction = np.clip(risk_fraction, 0.1, 1.0)
    
    # Win/loss ratio
    win_loss_ratio = expected_profit_dollars / expected_loss_dollars if expected_loss_dollars > 0 else 1.0
    
    # Find feasible share allocations
    target_cap = row['target_capital']
    min_cap = max(acct * MIN_POSITION_SIZE, target_cap * 0.8)
    max_cap = min(acct * MAX_POSITION_SIZE, target_cap * 1.2)
    
    # Calculate share range for stock1
    shares1_min = max(1, int(min_cap / (price1 * (1 + 1/abs(row['hedge_ratio'])))))
    shares1_max = int(max_cap / price1) + 10
    
    feasible_allocations = []
    
    for shares1 in range(shares1_min, min(shares1_max, 1000)):
        # Calculate target shares2 based on hedge ratio
        target_shares2 = (shares1 * price1) / (abs(row['hedge_ratio']) * price2)
        
        # Try both floor and ceiling
        for shares2 in [int(target_shares2), int(target_shares2) + 1]:
            if shares2 < 1:
                continue
            
            # Calculate actual capital deployment
            capital = shares1 * price1 + shares2 * price2
            
            # Calculate hedge ratio error
            actual_ratio = (shares1 * price1) / (shares2 * price2)
            hedge_error = abs(actual_ratio - abs(row['hedge_ratio'])) / abs(row['hedge_ratio']) * 100
            
            # Check constraints
            if min_cap <= capital <= max_cap and hedge_error <= MAX_HEDGE_ERROR:
                # Avoid duplicates
                if not any(abs(f['capital'] - capital) < 10 for f in feasible_allocations):
                    feasible_allocations.append({
                        'shares1': shares1,
                        'shares2': shares2,
                        'capital': capital,
                        'hedge_error': hedge_error
                    })
    
    if not feasible_allocations:
        print(f"  No feasible allocation for {row['stock1']}/{row['stock2']}")
        continue
    
    # Select allocation closest to target capital
    best_allocation = min(feasible_allocations, key=lambda x: abs(x['capital'] - target_cap))
    
    # Calculate risk dollars
    risk_dollars = best_allocation['capital'] * risk_fraction
    
    final_positions.append({
        'stock1': row['stock1'],
        'stock2': row['stock2'],
        'signal': row['signal'],
        'z_score': row['z_score'],
        'edge': row['edge'],
        'p_target': row['p_target'],
        'p_stop': row['p_stop'],
        'hedge_ratio': row['hedge_ratio'],
        'weight': row['weight'],
        'kelly_weight': row['weight'],
        'capital_allocation': best_allocation['capital'],
        'risk_dollars': risk_dollars,
        'risk_pct': risk_fraction * 100,
        'shares1': best_allocation['shares1'],
        'shares2': best_allocation['shares2'],
        'price1': price1,
        'price2': price2,
        'hedge_error': best_allocation['hedge_error'],
        'win_loss_ratio': win_loss_ratio,
        'expected_profit': expected_profit_dollars,
        'expected_loss': expected_loss_dollars,
        'theta': row['theta'],
        'volatility': row['volatility']
    })

# Create output dataframe
if final_positions:
    output_df = pd.DataFrame(final_positions)
    output_df = output_df.sort_values('edge', ascending=False)
    
    # Save to CSV
    output_df.to_csv("data/kelly_positions.csv", index=False)
    
    # Calculate final portfolio-level metrics
    total_expected_profit = output_df['expected_profit'].sum()
    total_expected_loss = output_df['expected_loss'].sum()
    portfolio_reward_risk = total_expected_profit / total_expected_loss if total_expected_loss > 0 else 0
    
    # Portfolio win probability (weighted by capital allocation)
    portfolio_win_prob = (output_df['p_target'] * output_df['capital_allocation']).sum() / output_df['capital_allocation'].sum()
    
    # Calculate actual total risk and apply scaling if needed
    total_deployed = output_df['capital_allocation'].sum()
    total_risk = output_df['risk_dollars'].sum()
    target_risk = adjusted_kelly * acct
    
    # If we're over the target risk, scale down all positions proportionally
    if total_risk > target_risk * 1.01:  # Allow 1% tolerance
        scale_factor = target_risk / total_risk
        print(f"\nAdjusting positions to meet Kelly risk target...")
        print(f"  Current total risk: ${total_risk:,.2f} ({total_risk/acct:.1%})")
        print(f"  Target risk: ${target_risk:,.2f} ({adjusted_kelly:.1%})")
        print(f"  Scaling factor: {scale_factor:.3f}")
        
        # Scale down the existing final positions instead of rebuilding
        for pos in final_positions:
            pos['capital_allocation'] *= scale_factor
            pos['risk_dollars'] *= scale_factor
            
            # Recalculate shares based on new capital
            price1 = pos['price1']
            price2 = pos['price2']
            hedge_ratio = pos['hedge_ratio']
            target_cap = pos['capital_allocation']
            
            # Find new feasible allocation
            min_cap = max(acct * MIN_POSITION_SIZE, target_cap * 0.8)
            max_cap = min(acct * MAX_POSITION_SIZE, target_cap * 1.2)
            shares1_min = max(1, int(min_cap / (price1 * (1 + 1/abs(hedge_ratio)))))
            shares1_max = int(max_cap / price1) + 10
            
            feasible = []
            for s1 in range(shares1_min, min(shares1_max, 1000)):
                ts2 = (s1 * price1) / (abs(hedge_ratio) * price2)
                for s2 in [int(ts2), int(ts2) + 1]:
                    if s2 < 1:
                        continue
                    cap = s1 * price1 + s2 * price2
                    actual_ratio = (s1 * price1) / (s2 * price2)
                    hedge_error = abs(actual_ratio - abs(hedge_ratio)) / abs(hedge_ratio) * 100
                    if min_cap <= cap <= max_cap and hedge_error <= MAX_HEDGE_ERROR:
                        if not any(abs(f['capital'] - cap) < 10 for f in feasible):
                            feasible.append({
                                'shares1': s1,
                                'shares2': s2,
                                'capital': cap,
                                'hedge_error': hedge_error
                            })
            
            if feasible:
                best = min(feasible, key=lambda x: abs(x['capital'] - target_cap))
                pos['shares1'] = best['shares1']
                pos['shares2'] = best['shares2']
                pos['capital_allocation'] = best['capital']
                pos['hedge_error'] = best['hedge_error']
                
                # Recalculate risk with new capital
                risk_frac = pos['risk_pct'] / 100
                pos['risk_dollars'] = best['capital'] * risk_frac
        
        output_df = pd.DataFrame(final_positions)
        output_df = output_df.sort_values('edge', ascending=False)
        output_df.to_csv("data/kelly_positions.csv", index=False)
        
        # Recalculate metrics
        total_expected_profit = output_df['expected_profit'].sum()
        total_expected_loss = output_df['expected_loss'].sum()
        portfolio_reward_risk = total_expected_profit / total_expected_loss if total_expected_loss > 0 else 0
        portfolio_win_prob = (output_df['p_target'] * output_df['capital_allocation']).sum() / output_df['capital_allocation'].sum()

    
    print(f"\n{'='*80}")
    print(f"FINAL PORTFOLIO - {len(output_df)} Positions")
    print(f"{'='*80}")
    print(f"Total DEPLOYED: ${output_df['capital_allocation'].sum():,.2f} ({output_df['capital_allocation'].sum()/acct:.1%} of account)")
    print(f"Total RISK: ${output_df['risk_dollars'].sum():,.2f} ({output_df['risk_dollars'].sum()/acct:.1%} of account)")
    print(f"Target RISK: ${target_risk:,.2f} ({adjusted_kelly:.1%} of account)")
    print(f"Average risk per position: {output_df['risk_pct'].mean():.1f}%")
    print(f"Weighted average edge: {(output_df['edge'] * output_df['weight']).sum():.4f}")
    print(f"Mean hedge error: {output_df['hedge_error'].mean():.2f}%")
    print(f"\nPortfolio-level metrics:")
    print(f"  Reward/Risk Ratio: {portfolio_reward_risk:.3f}")
    print(f"  Expected Win Probability: {portfolio_win_prob:.2%}")
    
    print(f"\nTop 10 positions by edge:")
    display_cols = ['stock1', 'stock2', 'edge', 'capital_allocation', 'risk_dollars', 
                    'shares1', 'shares2', 'signal']
    print(output_df[display_cols].head(10).to_string(index=False))
    
    print(f"\nPosition breakdown:")
    for _, row in output_df.iterrows():
        print(f"  {row['stock1']}/{row['stock2']}:")
        print(f"    Signal: {row['signal']}, Z-score: {row['z_score']:.2f}, Edge: {row['edge']:.4f}")
        print(f"    Deploy: ${row['capital_allocation']:,.0f}, Risk: ${row['risk_dollars']:,.0f} ({row['risk_pct']:.1f}%)")
        print(f"    Shares: {row['shares1']} {row['stock1']}, {row['shares2']} {row['stock2']}")
        print(f"    Prices: ${row['price1']:.2f}, ${row['price2']:.2f}, Hedge error: {row['hedge_error']:.2f}%")
        print()
    
    print(f"Saved to data/kelly_positions.csv")
    print(f"{'='*80}")
    
else:
    print("\nNo valid positions to trade.")
    pd.DataFrame().to_csv("data/kelly_positions.csv", index=False)
=======
from scipy.optimize import minimize

def calculate_spread_returns(prices_df, stock1, stock2, beta):
    """Calculate returns of the spread for a pair."""
    spread = prices_df[stock1] - beta * prices_df[stock2]
    returns = spread.pct_change().dropna()
    return returns

def build_correlation_matrix(edges_df, prices_df):
    """Build correlation matrix of spread returns for all pairs."""
    n_pairs = len(edges_df)
    returns_list = []
    pair_ids = []
    
    print(f"Calculating spread returns for {n_pairs} pairs...")
    
    for idx, row in edges_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{n_pairs}")
        
        pair_id = f"{row['stock1']}_{row['stock2']}"
        returns = calculate_spread_returns(
            prices_df, row['stock1'], row['stock2'], row['hedge_ratio']
        )
        returns_list.append(returns)
        pair_ids.append(pair_id)
    
    # Align all returns to common dates
    returns_df = pd.DataFrame({pair_ids[i]: returns_list[i] for i in range(len(pair_ids))})
    returns_df = returns_df.dropna()
    
    print(f"Computing correlation matrix...")
    corr_matrix = returns_df.corr()
    
    return corr_matrix, returns_df

def kelly_objective(weights, expected_returns, cov_matrix):
    """
    Negative Kelly criterion (for minimization).
    Kelly = argmax[w^T * mu - 0.5 * w^T * Sigma * w]
    """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    
    kelly_value = portfolio_return - 0.5 * portfolio_variance
    
    return -kelly_value

def optimize_kelly_portfolio(edges_df, corr_matrix, returns_df, max_correlation=0.7):
    """
    Optimize portfolio using Kelly criterion with correlation constraints.
    """
    n_pairs = len(edges_df)
    
    # Use edge as expected return proxy
    expected_returns = edges_df['edge'].values
    
    # Covariance matrix
    cov_matrix = returns_df.cov().values
    
    # Initial weights
    w0 = np.ones(n_pairs) / n_pairs
    
    # Constraints: weights sum to 1
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    ]
    
    # Bounds: 0 to 0.2 per position
    bounds = [(0, 0.2) for _ in range(n_pairs)]
    
    # Optimize
    print("Running Kelly optimization...")
    result = minimize(
        kelly_objective,
        w0,
        args=(expected_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'disp': False}
    )
    
    optimal_weights = result.x
    
    # Filter by correlation
    print("Filtering by correlation threshold...")
    selected_pairs = []
    selected_indices = []
    
    pair_ids = [f"{edges_df.iloc[i]['stock1']}_{edges_df.iloc[i]['stock2']}" 
                for i in range(n_pairs)]
    
    # Sort by weight descending
    sorted_indices = np.argsort(-optimal_weights)
    
    for i in sorted_indices:
        if optimal_weights[i] > 0.001:
            too_correlated = False
            
            for j in selected_indices:
                correlation = corr_matrix.loc[pair_ids[i], pair_ids[j]]
                
                if abs(correlation) > max_correlation:
                    too_correlated = True
                    break
            
            if not too_correlated:
                selected_pairs.append(i)
                selected_indices.append(i)
    
    return optimal_weights, selected_pairs, result

def main():
    # Load data
    print("Loading data...")
    prices_df = pd.read_csv("data/sp500_prices_clean.csv")
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.set_index('date')
    
    edges_df = pd.read_csv("data/pair_edges.csv")
    
    # Filter for positive edges
    positive_edges = edges_df[edges_df['edge'] > 0].copy().reset_index(drop=True)
    
    print(f"Working with {len(positive_edges)} pairs")
    
    # Build correlation matrix
    corr_matrix, returns_df = build_correlation_matrix(positive_edges, prices_df)
    
    # Optimize portfolio
    MAX_CORRELATION = 0.7
    
    optimal_weights, selected_indices, result = optimize_kelly_portfolio(
        positive_edges, corr_matrix, returns_df, MAX_CORRELATION
    )
    
    # Create results
    positive_edges['kelly_weight'] = optimal_weights
    positive_edges['selected'] = False
    positive_edges.loc[selected_indices, 'selected'] = True
    
    selected_pairs = positive_edges[positive_edges['selected']].copy()
    
    # Renormalize weights
    if len(selected_pairs) > 0:
        selected_pairs['kelly_weight'] = selected_pairs['kelly_weight'] / selected_pairs['kelly_weight'].sum()
    
    print(f"\nOptimization complete:")
    print(f"  Initial pairs: {len(positive_edges)}")
    print(f"  Selected pairs: {len(selected_pairs)}")
    print(f"  Removed (correlation): {len(positive_edges) - len(selected_pairs)}")
    
    # Calculate portfolio statistics
    if len(selected_pairs) > 0:
        portfolio_edge = (selected_pairs['edge'] * selected_pairs['kelly_weight']).sum()
        
        TOTAL_CAPITAL = 100000
        selected_pairs['capital'] = selected_pairs['kelly_weight'] * TOTAL_CAPITAL
        
        print(f"\nPortfolio statistics:")
        print(f"  Weighted average edge: {portfolio_edge:.4f}")
        print(f"  Expected value: ${portfolio_edge * TOTAL_CAPITAL:,.2f}")
        print(f"  Mean allocation: ${selected_pairs['capital'].mean():,.2f}")
        print(f"  Max allocation: ${selected_pairs['capital'].max():,.2f}")
        
        # Save results
        output_cols = ['stock1', 'stock2', 'edge', 'z_score', 'p_target', 
                       'signal', 'kelly_weight', 'capital']
        selected_pairs[output_cols].to_csv("data/kelly_positions.csv", index=False)
        
        print("\nTop 10 positions by Kelly weight:")
        display_cols = ['stock1', 'stock2', 'edge', 'kelly_weight', 'capital']
        print(selected_pairs.nlargest(10, 'kelly_weight')[display_cols].to_string(index=False))
        
        print(f"\nSaved to data/kelly_positions.csv")
        
        # Correlation stats
        selected_pair_ids = [
            f"{row['stock1']}_{row['stock2']}" 
            for _, row in selected_pairs.iterrows()
        ]
        selected_corr = corr_matrix.loc[selected_pair_ids, selected_pair_ids]
        
        upper_tri = selected_corr.where(
            np.triu(np.ones(selected_corr.shape), k=1).astype(bool)
        )
        correlations = upper_tri.stack().values
        
        if len(correlations) > 0:
            print(f"\nCorrelation statistics of selected pairs:")
            print(f"  Mean: {correlations.mean():.3f}")
            print(f"  Max: {correlations.max():.3f}")
            print(f"  Min: {correlations.min():.3f}")
    else:
        print("\nNo pairs selected. Try adjusting correlation threshold.")

if __name__ == "__main__":
    main()
>>>>>>> 9e9798d47b88df72db715ebfdd10d4e6f4298a1c
