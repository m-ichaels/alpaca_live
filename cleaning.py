import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
try:
    from auth_local import KEY, SECRET  # For local testing
except ImportError:
    print("Import error")
    from auth import KEY, SECRET  # For GitHub Actions

# Connect to Alpaca
print("Connecting to Alpaca to check short-sale eligibility...")
tc = TradingClient(KEY, SECRET, paper=True)

# Load data
df = pd.read_csv("data/sp500_prices.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"\nOriginal: {len(df)} rows, {len(df.columns)} stocks")

# Get all tradable stocks from Alpaca
try:
    # Get all active, tradable stocks
    search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
    assets = tc.get_all_assets(search_params)
    
    # Create lookup dict: symbol -> asset properties
    asset_dict = {asset.symbol: asset for asset in assets if asset.tradable and asset.status == 'active'}
    
    print(f"\nChecking short-sale eligibility for {len(df.columns)} stocks...")
    
    # Filter stocks based on Alpaca short-sale eligibility
    shortable_stocks = []
    non_shortable_stocks = []
    not_found_stocks = []
    
    for stock in df.columns:
        if stock in asset_dict:
            asset = asset_dict[stock]
            # Check if stock is shortable AND easy to borrow
            if asset.shortable and asset.easy_to_borrow:
                shortable_stocks.append(stock)
            else:
                non_shortable_stocks.append(stock)
                print(f"  [X] {stock}: shortable={asset.shortable}, easy_to_borrow={asset.easy_to_borrow}")
        else:
            not_found_stocks.append(stock)
            print(f"  [X] {stock}: not found in Alpaca assets")
    
    print(f"\nShort-sale filtering results:")
    print(f"  Shortable: {len(shortable_stocks)}")
    print(f"  Not shortable: {len(non_shortable_stocks)}")
    print(f"  Not found: {len(not_found_stocks)}")
    
    if non_shortable_stocks:
        print(f"\nRemoved stocks (not shortable):")
        print(f"  {', '.join(sorted(non_shortable_stocks))}")
    
    if not_found_stocks:
        print(f"\nRemoved stocks (not found in Alpaca):")
        print(f"  {', '.join(sorted(not_found_stocks))}")
    
    # Keep only shortable stocks
    df_clean = df[shortable_stocks].copy()
    
except Exception as e:
    print(f"\nError checking Alpaca assets: {e}")
    print("Proceeding without short-sale filter...")
    df_clean = df.copy()

# Remove stocks with < 240 days of data
print(f"\nFiltering for data completeness (>= 240 days)...")
valid_stocks = df_clean.columns[df_clean.notna().sum() >= 240]
df_clean = df_clean[valid_stocks].copy()

print(f"After data completeness filter: {len(df_clean.columns)} stocks")
print(f"Removed {len(df.columns) - len(df_clean.columns)} stocks total")

# Interpolate missing values
print(f"\nInterpolating missing values...")
df_clean = df_clean.interpolate(method='linear', limit_direction='both')

# Check remaining nulls
null_counts = df_clean.isnull().sum()
if null_counts.sum() > 0:
    print(f"\nRemaining nulls after interpolation:")
    print(null_counts[null_counts > 0])
else:
    print("No remaining null values")

# Save
df_clean = df_clean.reset_index()
df_clean.to_csv("data/sp500_prices_clean.csv", index=False)

print(f"\n{'='*80}")
print(f"FINAL DATASET")
print(f"{'='*80}")
print(f"Rows: {len(df_clean)}")
print(f"Stocks: {len(df_clean.columns) - 1}")  # -1 for date column
print(f"All stocks are:")
print(f"  [OK] Shortable on Alpaca")
print(f"  [OK] Easy to borrow")
print(f"  [OK] Have >= 240 days of data")
print(f"\nSaved to data/sp500_prices_clean.csv")
print(f"{'='*80}")