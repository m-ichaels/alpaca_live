import pandas as pd

# Load data
df = pd.read_csv("data/sp500_prices.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"Original: {len(df)} rows, {len(df.columns)} stocks")

# Remove stocks with < 240 days of data
valid_stocks = df.columns[df.notna().sum() >= 240]
df_clean = df[valid_stocks].copy()

print(f"After filtering: {len(df_clean)} rows, {len(df_clean.columns)} stocks")
print(f"Removed {len(df.columns) - len(df_clean.columns)} stocks")

# Interpolate missing values
df_clean = df_clean.interpolate(method='linear', limit_direction='both')

# Check remaining nulls
null_counts = df_clean.isnull().sum()
if null_counts.sum() > 0:
    print(f"\nRemaining nulls after interpolation:")
    print(null_counts[null_counts > 0])

# Save
df_clean = df_clean.reset_index()
df_clean.to_csv("data/sp500_prices_clean.csv", index=False)
print(f"\nSaved to data/sp500_prices_clean.csv")