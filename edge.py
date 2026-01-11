import pandas as pd
import numpy as np
from tp_sl import TAKE_PROFIT_Z, STOP_LOSS_Z

def estimate_ou_parameters(spread_series):
    """
    Estimate Ornstein-Uhlenbeck parameters from spread time series.
    Uses discrete-time estimation for theta and vol.
    
    Returns:
        theta: mean reversion speed
        vol: volatility
    """
    # Convert spread to numpy array
    S = spread_series.values
    n = len(S)
    
    # Time step (assuming daily data)
    dt = 1.0
    
    # Calculate mean
    mu = np.mean(S)
    
    # Demean the series
    S_demean = S - mu
    
    # Calculate theta using regression: dS = -theta*(S - mu)*dt + vol*dW
    # Rearrange: S[t+1] - S[t] = -theta*(S[t] - mu)*dt + noise
    S_lag = S_demean[:-1]
    S_diff = np.diff(S_demean)
    
    # Regression coefficient
    numerator = np.sum(S_lag * S_diff)
    denominator = np.sum(S_lag ** 2) * dt
    
    if denominator > 1e-10:
        theta = -numerator / denominator
    else:
        theta = 0.1  # Default fallback
    
    # Ensure theta is positive
    theta = max(theta, 0.01)
    
    # Calculate volatility from residuals
    predicted_diff = -theta * S_lag * dt
    residuals = S_diff - predicted_diff
    vol = np.std(residuals) / np.sqrt(dt)
    
    return theta, vol

def calculate_edge(z_current, theta, vol, b_target, b_stop, gain, loss):
    """
    Calculate edge using mean-reversion probability formula.
    
    Edge = P(target)*Gain - P(stop)*Loss
    
    where P(target|z_t) = [exp(a*z_t) - exp(a*b_stop)] / [exp(a*b_target) - exp(a*b_stop)]
    and a = 2*theta / vol^2
    """
    # Calculate parameter a
    a = 2 * theta / (vol ** 2)
    
    # Calculate probability of hitting target before stop
    try:
        numerator = np.exp(a * z_current) - np.exp(a * b_stop)
        denominator = np.exp(a * b_target) - np.exp(a * b_stop)
        
        # Prevent division by zero
        if abs(denominator) < 1e-10:
            p_target = 0.5
        else:
            p_target = numerator / denominator
            p_target = np.clip(p_target, 0, 1)
    except:
        p_target = 0.5
    
    p_stop = 1 - p_target
    
    # Calculate edge
    edge = p_target * gain - p_stop * loss
    
    return edge, p_target, p_stop

def main():
    # Load data
    print("Loading data...")
    prices_df = pd.read_csv("data/sp500_prices_clean.csv")
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.set_index('date')
    
    pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
    
    print(f"Analyzing {len(pairs_df)} cointegrated pairs...")
    
    # Storage for results
    results = []
    
    # Track filtered pairs
    skipped_inside_target = 0
    skipped_outside_stop = 0
    
    for idx, row in pairs_df.iterrows():
        stock1 = row['stock1']
        stock2 = row['stock2']
        beta = row['hedge_ratio']
        
        try:
            # Calculate spread
            spread = prices_df[stock1] - beta * prices_df[stock2]
            
            # Standardize to z-score
            mu = spread.mean()
            sigma = spread.std()
            z_spread = (spread - mu) / sigma
            
            # Current z-score (most recent)
            z_current = z_spread.iloc[-1]
            
            # FILTER: Skip pairs already inside take-profit zone or outside stop-loss
            if abs(z_current) < TAKE_PROFIT_Z:
                # Already inside the take-profit zone (too close to mean)
                skipped_inside_target += 1
                continue
            
            if abs(z_current) > STOP_LOSS_Z:
                # Already outside the stop-loss (too far from mean)
                skipped_outside_stop += 1
                continue
            
            # Estimate OU parameters from the spread (not z-score)
            theta, vol = estimate_ou_parameters(spread)
            
            # FIXED: Define barriers for MEAN REVERSION
            # Normalize to work with absolute values to ensure symmetry
            abs_z = abs(z_current)
            
            # For the probability calculation, work in normalized space
            # where we're always moving from abs_z toward TAKE_PROFIT_Z
            # This ensures symmetric treatment of positive and negative z-scores
            
            # Barriers in normalized (absolute) space
            b_target_abs = TAKE_PROFIT_Z  # Closer to zero
            b_stop_abs = STOP_LOSS_Z      # Further from zero
            
            # Calculate edge using normalized values
            gain = abs_z - b_target_abs  # Distance to target
            loss = b_stop_abs - abs_z     # Distance to stop
            
            # Calculate probability using NORMALIZED position
            # We're at abs_z, moving toward b_target_abs, with stop at b_stop_abs
            edge, p_target, p_stop = calculate_edge(
                abs_z, theta, vol, b_target_abs, b_stop_abs, gain, loss
            )
            
            # Determine signal based on original sign
            if z_current > 0:
                signal = "SHORT_SPREAD"   # Short stock1, Long stock2
            else:
                signal = "LONG_SPREAD"     # Long stock1, Short stock2
            
            results.append({
                'stock1': stock1,
                'stock2': stock2,
                'z_score': z_current,
                'theta': theta,
                'volatility': vol,
                'edge': edge,
                'p_target': p_target,
                'p_stop': p_stop,
                'signal': signal,
                'gain': gain,
                'loss': loss,
                'hedge_ratio': beta,
                'pvalue': row['pvalue']
            })
            
        except Exception as e:
            print(f"  Error with {stock1}/{stock2}: {e}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by edge (highest first)
    results_df = results_df.sort_values('edge', ascending=False)
    
    # Save results
    results_df.to_csv("data/pair_edges.csv", index=False)
    
    print(f"\nCalculated edges for {len(results_df)} pairs")
    print(f"Filtered out:")
    print(f"  - {skipped_inside_target} pairs already inside take-profit zone (|z| < {TAKE_PROFIT_Z})")
    print(f"  - {skipped_outside_stop} pairs already outside stop-loss (|z| > {STOP_LOSS_Z})")
    print(f"Saved to data/pair_edges.csv")
    
    # Display top 10 by edge
    print("\nTop 10 pairs by edge:")
    display_cols = ['stock1', 'stock2', 'z_score', 'edge', 'p_target', 'signal']
    print(results_df[display_cols].head(10).to_string(index=False))
    
    # Summary statistics
    print(f"\nEdge statistics:")
    print(f"  Mean edge: {results_df['edge'].mean():.4f}")
    print(f"  Median edge: {results_df['edge'].median():.4f}")
    print(f"  Positive edges: {(results_df['edge'] > 0).sum()} / {len(results_df)}")

if __name__ == "__main__":
    main()