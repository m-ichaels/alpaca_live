import pandas as pd
import os
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET

# Connect to Alpaca Paper Trading
trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

TRACKER_FILE = "data/open_pairs.csv"
HISTORY_FILE = "data/trade_history.csv"

def get_latest_price(symbol):
    """Get latest price for a symbol"""
    try:
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(request)
        return float(quote[symbol].ask_price)
    except:
        return None

def calculate_exit_z_score(stock1, stock2, beta):
    """
    Calculate current z-score for a pair using THE EXACT SAME beta from entry.
    This ensures consistency between entry and exit calculations.
    """
    try:
        # Load historical data
        prices_df = pd.read_csv("data/sp500_prices_clean.csv")
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
        
        # Get current prices
        p1 = get_latest_price(stock1)
        p2 = get_latest_price(stock2)
        
        if p1 is None or p2 is None:
            return None
        
        # Calculate current spread using THE EXACT SAME beta from entry
        current_spread = p1 - (beta * p2)
        
        # Get historical spread statistics using THE EXACT SAME beta
        hist_spread = prices_df[stock1] - (beta * prices_df[stock2])
        mu, sigma = hist_spread.mean(), hist_spread.std()
        
        # Calculate z-score
        current_z = (current_spread - mu) / sigma
        return current_z
        
    except Exception as e:
        print(f"    Warning: Could not calculate z-score - {e}")
        return None

def log_liquidated_trades():
    """
    Log all open trades as liquidated and REMOVE them from tracker.
    Uses stored hedge ratio from tracker for consistency.
    """
    try:
        if not os.path.exists(TRACKER_FILE):
            print("  No tracker file found")
            return
        
        tracker = pd.read_csv(TRACKER_FILE)
        
        # Check if hedge_ratio column exists
        if 'hedge_ratio' not in tracker.columns:
            print("  ERROR: tracker missing hedge_ratio column!")
            print("  Please delete data/open_pairs.csv and recreate it by running the pipeline.")
            return
        
        open_pairs = tracker[tracker['status'] == 'open'].copy()
        
        if len(open_pairs) == 0:
            print("  No open pairs to log")
            return
        
        print(f"\n  Logging {len(open_pairs)} liquidated trades to history...")
        
        pairs_to_remove = []
        
        for idx, row in open_pairs.iterrows():
            stock1, stock2 = row['stock1'], row['stock2']
            
            # USE THE STORED HEDGE RATIO FROM TRACKER (exact same as entry!)
            beta = row['hedge_ratio']
            
            print(f"    Processing {stock1}/{stock2} (beta={beta:.4f})...")
            
            exit_z = calculate_exit_z_score(stock1, stock2, beta)
            
            # Update trade history
            if os.path.exists(HISTORY_FILE):
                history = pd.read_csv(HISTORY_FILE)
                
                # Find the open trade
                mask = (
                    ((history['stock1'] == stock1) & (history['stock2'] == stock2)) |
                    ((history['stock1'] == stock2) & (history['stock2'] == stock1))
                ) & (history['exit_date'].isna())
                
                if mask.any():
                    history.loc[mask, 'exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    history.loc[mask, 'exit_z'] = exit_z if exit_z is not None else history.loc[mask, 'entry_z']
                    history.loc[mask, 'exit_reason'] = 'MANUAL LIQUIDATION'
                    history.loc[mask, 'win'] = None  # Neutral - not counted in win rate
                    
                    history.to_csv(HISTORY_FILE, index=False)
                    z_str = f"entry_z={row['z_score']:.2f}, exit_z={exit_z:.2f}" if exit_z is not None else "z=unknown"
                    print(f"      ✓ Logged to history ({z_str})")
                else:
                    print(f"      ! Trade not found in history")
            
            # Mark this pair for removal
            pairs_to_remove.append(idx)
        
        # REMOVE closed pairs from tracker (don't just mark as closed)
        tracker = tracker.drop(pairs_to_remove)
        tracker.to_csv(TRACKER_FILE, index=False)
        print(f"\n  ✓ Removed {len(pairs_to_remove)} pairs from tracker")
        print(f"  ✓ Remaining open pairs: {len(tracker[tracker['status'] == 'open'])}")
        
    except Exception as e:
        print(f"  ERROR logging liquidated trades: {e}")
        import traceback
        traceback.print_exc()

# Main liquidation logic
print("=" * 80)
print("MANUAL LIQUIDATION")
print("=" * 80)

print("\nCanceling all open orders...")
try:
    trading_client.cancel_orders()
    print("✓ All open orders canceled")
except Exception as e:
    print(f"✗ Error canceling orders: {e}")

print("\nLiquidating all positions...")
try:
    positions = trading_client.get_all_positions()
    
    if not positions:
        print("✓ No positions to liquidate")
    else:
        for position in positions:
            symbol = position.symbol
            qty = abs(float(position.qty))
            side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY
            
            try:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                order = trading_client.submit_order(order_data)
                print(f"  ✓ {symbol}: {side.value} {qty} shares")
            except Exception as e:
                print(f"  ✗ {symbol}: Error - {e}")
        
        print(f"\n✓ Liquidation complete - closed {len(positions)} positions")
except Exception as e:
    print(f"✗ Error liquidating positions: {e}")

# Log all liquidated trades and REMOVE from tracker
print("\nUpdating trade records...")
log_liquidated_trades()

print("\n" + "=" * 80)
print("Liquidation complete. Manual exits logged as neutral (not counted in win rate).")
print("All closed pairs removed from tracker.")
print("=" * 80)