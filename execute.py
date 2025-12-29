import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET

# Connect to Alpaca Paper Trading
trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

# Load entry signals and eigenvectors
signals_df = pd.read_csv("data/ou_entry_scores.csv")
groups_df = pd.read_csv("data/cointegrated_groups.csv")
signals_df = signals_df.merge(groups_df[['group', 'eigenvector']], on='group', how='left')
signals_df = signals_df[signals_df['signal'].isin(['BUY', 'SELL'])]
signals_df = signals_df.sort_values('entry_score', ascending=False)

print(f"Processing {len(signals_df)} trading signals")

# Get current positions
positions = trading_client.get_all_positions()
current_holdings = {p.symbol: float(p.qty) for p in positions}

# Get open orders
open_orders = trading_client.get_orders()
pending_symbols = {o.symbol for o in open_orders}

print(f"\nCurrent positions: {len(current_holdings)}")
print(f"Open orders: {len(pending_symbols)}")

executed_trades = []

for idx, row in signals_df.iterrows():
    tickers = row['group'].split(',')
    weights = [float(x) for x in row['eigenvector'].split(',')]
    signal = row['signal']
    
    # Normalize weights to sum to 1 (absolute value)
    weights = np.array(weights)
    weights = weights / np.sum(np.abs(weights))
    
    print(f"\n--- Group: {row['group'][:50]}... ---")
    print(f"Signal: {signal}, Z-score: {row['z_score']:.2f}, Entry score: {row['entry_score']:.2f}")
    
    # Check for partial fills
    has_position = any(ticker in current_holdings for ticker in tickers)
    
    if has_position:
        conflicting = [t for t in tickers if t in current_holdings]
        print(f"Skipping - existing positions: {', '.join(conflicting)}")
        continue
    
    # Check pending orders
    pending_in_group = [t for t in tickers if t in pending_symbols]
    
    if len(pending_in_group) == len(tickers):
        print(f"Skipping - all stocks have pending orders")
        continue
    elif len(pending_in_group) > 0:
        print(f"Partial pending orders detected: {', '.join(pending_in_group)}")
        print(f"Will place orders for remaining: {', '.join([t for t in tickers if t not in pending_symbols])}")
        tickers_to_trade = [t for t in tickers if t not in pending_symbols]
        weights_to_trade = [w for t, w in zip(tickers, weights) if t not in pending_symbols]
    else:
        tickers_to_trade = tickers
        weights_to_trade = weights
    
    # Calculate position sizes ($1,000 per group)
    capital_per_group = 1000
    
    for ticker, weight in zip(tickers_to_trade, weights_to_trade):
        try:
            # Get current price
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = data_client.get_stock_latest_quote(quote_request)[ticker]
            
            # Use bid/ask midpoint
            price = (quote.bid_price + quote.ask_price) / 2
            
            if price is None or price <= 0:
                print(f"  {ticker}: No valid price available")
                continue
            
            # Calculate shares based on weight
            dollar_amount = capital_per_group * abs(weight)
            shares = int(dollar_amount / price)
            
            if shares == 0:
                continue
            
            # Determine action
            if signal == 'BUY':
                side = OrderSide.BUY if weight > 0 else OrderSide.SELL
            else:  # SELL signal
                side = OrderSide.SELL if weight > 0 else OrderSide.BUY
            
            # Place order
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            order = trading_client.submit_order(order_data)
            
            executed_trades.append({
                'group': row['group'],
                'ticker': ticker,
                'action': side.value,
                'shares': shares,
                'price': price,
                'weight': weight,
                'signal': signal
            })
            
            print(f"  {ticker}: {side.value} {shares} shares @ ${price:.2f}")
            
        except Exception as e:
            print(f"  {ticker}: Error - {e}")

# Save executed trades
if executed_trades:
    trades_df = pd.DataFrame(executed_trades)
    trades_df.to_csv("data/executed_trades.csv", index=False)
    print(f"\nExecuted {len(executed_trades)} trades across {len(signals_df)} groups")
else:
    print("\nNo trades executed")