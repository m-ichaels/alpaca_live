import pandas as pd
import numpy as np

def mcdiarmid_sample_size(epsilon, delta, c):
    """
    Calculate required sample size using McDiarmid's inequality.
    
    P(|X_avg - E[X]| >= epsilon) <= 2*exp(-2*n*epsilon^2 / c^2)
    
    Setting this <= delta and solving for n:
    n >= (c^2 / (2*epsilon^2)) * ln(2/delta)
    
    Args:
        epsilon: Maximum deviation from expected value (e.g., 0.10 for 10%)
        delta: Confidence level (probability of exceeding epsilon)
        c: Range of individual outcomes (max edge - min edge)
    
    Returns:
        Required number of positions
    """
    n = (c**2 / (2 * epsilon**2)) * np.log(2 / delta)
    return int(np.ceil(n))

def calculate_position_sizes(edges_df, total_capital):
    """
    Size positions proportionally to absolute edge.
    
    Args:
        edges_df: DataFrame with edge values
        total_capital: Total capital to allocate
    
    Returns:
        DataFrame with position sizes
    """
    # Use absolute edge for proportional sizing
    edges_df['abs_edge'] = edges_df['edge'].abs()
    
    # Calculate weights
    total_abs_edge = edges_df['abs_edge'].sum()
    edges_df['weight'] = edges_df['abs_edge'] / total_abs_edge
    
    # Calculate capital allocation
    edges_df['capital'] = edges_df['weight'] * total_capital
    
    return edges_df

def main():
    # Load edge calculations
    print("Loading edge calculations...")
    edges_df = pd.read_csv("data/pair_edges.csv")
    
    # Filter for positive edges only
    positive_edges = edges_df[edges_df['edge'] > 0].copy()
    
    print(f"Found {len(positive_edges)} pairs with positive edge")
    
    if len(positive_edges) == 0:
        print("No pairs with positive edge. Cannot calculate position sizing.")
        return
    
    # McDiarmid parameters
    EPSILON = 0.10  # Within 10% of expected value
    DELTA = 0.05    # 95% confidence (5% probability of exceeding epsilon)
    
    # Calculate range c (assuming edges are bounded)
    max_edge = positive_edges['edge'].max()
    min_edge = positive_edges['edge'].min()
    c = max_edge - min_edge
    
    # Calculate required positions
    n_required = mcdiarmid_sample_size(EPSILON, DELTA, c)
    
    print(f"\nMcDiarmid's Inequality Analysis:")
    print(f"  Target deviation: {EPSILON*100:.0f}% of expected value")
    print(f"  Confidence level: {(1-DELTA)*100:.0f}%")
    print(f"  Edge range (c): {c:.4f}")
    print(f"  Required positions: {n_required}")
    
    # Select top N pairs by edge
    n_positions = min(n_required, len(positive_edges))
    selected_pairs = positive_edges.nlargest(n_positions, 'edge').copy()
    
    print(f"\nSelecting top {n_positions} pairs by edge")
    
    # Calculate position sizes (example with $100k capital)
    TOTAL_CAPITAL = 100000
    selected_pairs = calculate_position_sizes(selected_pairs, TOTAL_CAPITAL)
    
    # Save results
    output_cols = ['stock1', 'stock2', 'edge', 'z_score', 'p_target', 
                   'signal', 'weight', 'capital']
    selected_pairs[output_cols].to_csv("data/position_sizing.csv", index=False)
    
    print(f"\nPosition Sizing (Total Capital: ${TOTAL_CAPITAL:,.0f}):")
    print(f"  Mean allocation: ${selected_pairs['capital'].mean():,.2f}")
    print(f"  Median allocation: ${selected_pairs['capital'].median():,.2f}")
    print(f"  Min allocation: ${selected_pairs['capital'].min():,.2f}")
    print(f"  Max allocation: ${selected_pairs['capital'].max():,.2f}")
    
    print("\nTop 10 positions by capital allocation:")
    display_cols = ['stock1', 'stock2', 'edge', 'capital', 'weight']
    print(selected_pairs[display_cols].head(10).to_string(index=False))
    
    print(f"\nSaved to data/position_sizing.csv")
    
    # Summary statistics
    total_edge = (selected_pairs['edge'] * selected_pairs['weight']).sum()
    print(f"\nPortfolio statistics:")
    print(f"  Weighted average edge: {total_edge:.4f}")
    print(f"  Expected value: ${total_edge * TOTAL_CAPITAL:,.2f}")

if __name__ == "__main__":
    main()