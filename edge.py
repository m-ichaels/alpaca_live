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
            
            # Estimate OU parameters from the spread (not z-score)
            theta, vol = estimate_ou_parameters(spread)
            
            # Define barriers
            if z_current > 0:
                b_target = TAKE_PROFIT_Z
                b_stop = STOP_LOSS_Z
                signal = "LONG_SPREAD"
            else:
                b_target = -TAKE_PROFIT_Z
                b_stop = -STOP_LOSS_Z
                signal = "SHORT_SPREAD"
            
            # Calculate gain/loss in z-score terms
            gain = abs(z_current - b_target)
            loss = abs(z_current - b_stop)
            
            edge, p_target, p_stop = calculate_edge(
                z_current, theta, vol, b_target, b_stop, gain, loss
            )
            
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