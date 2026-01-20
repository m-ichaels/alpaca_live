import pandas as pd
import numpy as np
from scipy.integrate import quad

# Load data
df_pairs = pd.read_csv("data/pairs_with_tp.csv")

print(f"Calculating edge for {len(df_pairs)} pairs")

results = []

for idx, row in df_pairs.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    z_current = row['z_score']
    half_life = row['half_life']
    spread_std = row['spread_std']
    stop_loss = row['stop_loss']
    take_profit = row['take_profit']
    
    print(f"Processing {idx+1}/{len(df_pairs)}: {stock1}-{stock2}")
    
    # OU process parameters
    mu = 0  # Long-term mean
    lambda_param = np.log(2) / half_life  # Mean reversion speed from half-life
    sigma = spread_std
    
    # Calculate probability of hitting take profit before stop loss
    # Using the integral formula from the document
    def integrand(z):
        return np.exp((lambda_param / (sigma**2)) * (z - mu)**2)
    
    try:
        # Numerator: integral from current z to stop loss
        numerator, _ = quad(integrand, z_current, stop_loss)
        
        # Denominator: integral from take profit to stop loss
        denominator, _ = quad(integrand, take_profit, stop_loss)
        
        # Probability of hitting take profit first
        prob_win = numerator / denominator
        
    except Exception as e:
        print(f"  Warning: Integration failed for {stock1}-{stock2}, using fallback")
        # Fallback: simple linear approximation
        total_distance = abs(stop_loss - take_profit)
        distance_to_tp = abs(z_current - take_profit)
        prob_win = 1 - (distance_to_tp / total_distance) if total_distance > 0 else 0.5
    
    # Ensure probability is in valid range
    prob_win = np.clip(prob_win, 0, 1)
    
    # Calculate profit and loss magnitudes
    tp_magnitude = abs(take_profit - z_current)
    sl_magnitude = abs(stop_loss - z_current)
    
    # Calculate edge using expected value formula
    # Edge = P * |TP - x0| - (1-P) * |SL - x0|
    edge = (prob_win * tp_magnitude) - ((1 - prob_win) * sl_magnitude)
    
    results.append({
        'stock1': stock1,
        'stock2': stock2,
        'hedge_ratio': row['hedge_ratio'],
        'z_score': z_current,
        'half_life': half_life,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'spread_mean': row['spread_mean'],
        'spread_std': spread_std,
        'prob_win': prob_win,
        'tp_magnitude': tp_magnitude,
        'sl_magnitude': sl_magnitude,
        'edge': edge
    })

print(f"\n{'='*80}")
print(f"Edge calculation complete")
print(f"{'='*80}")

if results:
    results_df = pd.DataFrame(results)
    
    # Sort by edge (highest first)
    results_df = results_df.sort_values('edge', ascending=False)
    
    # Save
    results_df.to_csv("data/pairs_with_edge.csv", index=False)
    
    print(f"\nTop 10 pairs by edge:")
    print(results_df[['stock1', 'stock2', 'z_score', 'prob_win', 'edge']].head(10))
    
    print(f"\nSummary statistics:")
    print(f"  Average edge: {results_df['edge'].mean():.4f}")
    print(f"  Positive edge pairs: {len(results_df[results_df['edge'] > 0])}")
    print(f"  Negative edge pairs: {len(results_df[results_df['edge'] <= 0])}")
    print(f"  Average win probability: {results_df['prob_win'].mean():.4f}")
    
    print(f"\nSaved to data/pairs_with_edge.csv")
else:
    print("\nNo pairs to calculate edge for")
    pd.DataFrame().to_csv("data/pairs_with_edge.csv", index=False)