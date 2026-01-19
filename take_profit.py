import pandas as pd
import numpy as np

# Load data
df_acceptable = pd.read_csv("data/acceptable_pairs.csv")

print(f"Calculating take profit levels for {len(df_acceptable)} pairs")

results = []

for idx, row in df_acceptable.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    z_current = row['z_score']
    half_life = row['half_life']
    spread_std = row['spread_std']
    ema_current = row['stop_loss']  # This is the current EMA value
    
    # Progress indicator
    print(f"Processing {idx+1}/{len(df_acceptable)}: {stock1}-{stock2}")
    
    # OU process parameters
    mu = 0  # Long-term mean (spread should revert to 0)
    lambda_param = np.log(2) / half_life  # Mean reversion speed
    sigma = spread_std
    
    # Monte Carlo simulation parameters
    n_simulations = 10000
    n_steps = int(3 * half_life)  # Simulate for 3x half-life
    dt = 1  # Daily time step
    alpha = 1 - np.exp(-lambda_param * dt)  # EMA smoothing factor
    
    # Store all simulation paths
    all_paths = np.zeros((n_simulations, n_steps))
    
    # Run Monte Carlo simulations
    for sim in range(n_simulations):
        z = z_current
        
        for step in range(n_steps):
            # OU process: dX_t = lambda(mu - X_t)dt + sigma*dW_t
            dW = np.random.normal(0, np.sqrt(dt))
            dz = lambda_param * (mu - z) * dt + sigma * dW
            z = z + dz
            
            all_paths[sim, step] = z
    
    # Calculate average path across all simulations
    average_path = np.mean(all_paths, axis=0)
    
    # Calculate EMA of the average path
    ema_path = np.zeros(n_steps)
    ema = ema_current
    
    for step in range(n_steps):
        ema = alpha * average_path[step] + (1 - alpha) * ema
        ema_path[step] = ema
    
    # Find where average z-score path crosses EMA (take profit point)
    take_profit = None
    for step in range(n_steps):
        # Check if z-score crosses EMA (gets closer to zero than EMA)
        if z_current > 0:  # Positive z-score, expecting to revert down
            if average_path[step] <= ema_path[step]:
                take_profit = average_path[step]
                break
        else:  # Negative z-score, expecting to revert up
            if average_path[step] >= ema_path[step]:
                take_profit = average_path[step]
                break
    
    # If no crossing found, use final value
    if take_profit is None:
        take_profit = average_path[-1]
    
    results.append({
        'stock1': stock1,
        'stock2': stock2,
        'hedge_ratio': row['hedge_ratio'],
        'z_score': z_current,
        'half_life': half_life,
        'stop_loss': ema_current,
        'take_profit': take_profit,
        'spread_mean': row['spread_mean'],
        'spread_std': spread_std
    })

print(f"\n{'='*80}")
print(f"Calculated take profit for all pairs")
print(f"{'='*80}")

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/pairs_with_tp.csv", index=False)
    
    print(f"\nTop 10 pairs by z-score magnitude:")
    print(results_df[['stock1', 'stock2', 'z_score', 'stop_loss', 'take_profit']].head(10))
    
    print(f"\nSummary statistics:")
    print(f"  Average take profit distance: {abs(results_df['take_profit'] - results_df['z_score']).mean():.4f}")
    print(f"  Average stop loss distance: {abs(results_df['stop_loss'] - results_df['z_score']).mean():.4f}")
    
    print(f"\nSaved to data/pairs_with_tp.csv")
else:
    print("\nNo pairs to calculate take profit for")
    pd.DataFrame().to_csv("data/pairs_with_tp.csv", index=False)