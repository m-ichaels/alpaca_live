import pandas as pd
import numpy as np

# Load data
df_prices = pd.read_csv("data/sp500_prices_clean.csv")
df_prices['date'] = pd.to_datetime(df_prices['date'])
df_prices = df_prices.set_index('date')

df_pairs = pd.read_csv("data/cointegrated_pairs.csv")

print(f"Analyzing {len(df_pairs)} cointegrated pairs for entry signals")

tradeable_pairs = []

for idx, row in df_pairs.iterrows():
    stock1, stock2, beta = row['stock1'], row['stock2'], row['hedge_ratio']
    
    try:
        # Get prices
        y = df_prices[stock1].values
        x = df_prices[stock2].values
        
        # Calculate spread
        spread = y - beta * x
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # Calculate z-score time series
        z_scores = (spread - spread_mean) / spread_std
        
        # Calculate half-life via OU process
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        lambda_param = np.polyfit(spread_lag, spread_diff, 1)[0]
        half_life = -np.log(2) / lambda_param if lambda_param < 0 else np.inf
        
        # Skip if half-life invalid
        if half_life <= 0 or half_life > 60:
            continue
        
        # Calculate EMA of z-scores using half-life
        alpha = 1 - np.exp(-half_life * 1)  # dt = 1 day
        ema_z = np.zeros(len(z_scores))
        ema_z[0] = z_scores[0]
        
        for i in range(1, len(z_scores)):
            ema_z[i] = alpha * z_scores[i] + (1 - alpha) * ema_z[i-1]
        
        # Current z-score and EMA
        current_z = z_scores[-1]
        current_ema = ema_z[-1]
        
        # Entry condition: current z-score < EMA (mean reverting)
        if abs(current_z) < abs(current_ema):
            tradeable_pairs.append({
                'stock1': stock1,
                'stock2': stock2,
                'hedge_ratio': beta,
                'z_score': current_z,
                'half_life': half_life,
                'stop_loss': current_ema,  # EMA serves as stop loss
                'spread_mean': spread_mean,
                'spread_std': spread_std
            })
    
    except Exception:
        continue

print(f"\nFound {len(tradeable_pairs)} pairs below moving average (acceptable to trade)")

if tradeable_pairs:
    results_df = pd.DataFrame(tradeable_pairs)
    results_df = results_df.sort_values('z_score', key=abs)
    results_df.to_csv("data/acceptable_pairs.csv", index=False)
    
    print(f"\nTop 10 pairs by z-score magnitude:")
    print(results_df[['stock1', 'stock2', 'z_score', 'half_life', 'stop_loss']].head(10))
else:
    print("\nNo pairs currently acceptable to trade")
    pd.DataFrame().to_csv("data/acceptable_pairs.csv", index=False)