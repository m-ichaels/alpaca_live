import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from auth import KEY, SECRET

# Connect to Alpaca with IEX feed
client = StockHistoricalDataClient(KEY, SECRET)

# Load tickers
tickers_df = pd.read_csv("data/sp500_stocks.csv")
tickers = tickers_df["Symbol"].tolist()

all_data = {}

print(f"Fetching data for {len(tickers)} stocks...")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

for i, ticker in enumerate(tickers, 1):
    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed='iex'  # Use IEX feed for free tier
        )
        
        bars = client.get_stock_bars(request)
        
        if bars.df is not None and not bars.df.empty:
            df = bars.df.reset_index()
            df = df[['timestamp', 'close']].tail(250)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            all_data[ticker] = df['close']
            print(f"{i}/{len(tickers)}: {ticker} - {len(df)} days")
        else:
            print(f"{i}/{len(tickers)}: {ticker} - No data")
            
    except Exception as e:
        print(f"{i}/{len(tickers)}: {ticker} - Error: {e}")

# Combine into wide format
if all_data:
    final_df = pd.DataFrame(all_data)
    final_df.index.name = 'date'
    final_df = final_df.reset_index()
    final_df.to_csv("data/sp500_prices.csv", index=False)
    print(f"\nSaved {len(final_df)} rows x {len(final_df.columns)-1} stocks to data/sp500_prices.csv")
else:
    print("No data retrieved")