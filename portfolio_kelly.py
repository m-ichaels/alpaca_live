import pandas as pd
import numpy as np
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