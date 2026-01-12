import pandas as pd
import os
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
try:
    from auth_local import KEY, SECRET  # For local testing
except ImportError:
    print("Import error")
    from auth import KEY, SECRET  # For GitHub Actions

# Connect to Alpaca Paper Trading
trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

TRACKER_FILE = "data/open_pairs.csv"
HISTORY_FILE = "data/trade_history.csv"

def get_position_side(symbol, positions_dict):
    """
    Check if we have an open position and return 'LONG' or 'SHORT'
    Uses pre-fetched positions dict for efficiency
    """
    if symbol in positions_dict:
        qty = float(positions_dict[symbol].qty)
        return 'LONG' if qty > 0 else 'SHORT'
    return None

def get_exit_price(symbol, positions_dict):
    """
    Get the correct exit price for a position:
    - If LONG position: use BID (we're selling)
    - If SHORT position: use ASK (we're buying to cover)
    - If no position: use last trade price as fallback
    """
    try:
        position_side = get_position_side(symbol, positions_dict)
        
        quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(quote_request)
        
        if symbol not in quote:
            return None
        
        if position_side == 'LONG':
            # Exiting a long = selling = use BID price
            price = quote[symbol].bid_price
        elif position_side == 'SHORT':
            # Exiting a short = buying to cover = use ASK price
            price = quote[symbol].ask_price
        else:
            # No position - fall back to last trade price or mid price
            try:
                trade_request = StockLatestTradeRequest(symbol_or_symbols=symbol)
                trade = data_client.get_stock_latest_trade(trade_request)
                if symbol in trade and trade[symbol].price:
                    price = trade[symbol].price
                else:
                    # Use mid price as last resort
                    if quote[symbol].bid_price and quote[symbol].ask_price:
                        price = (float(quote[symbol].bid_price) + float(quote[symbol].ask_price)) / 2
                    else:
                        price = quote[symbol].ask_price or quote[symbol].bid_price
            except:
                # Use mid price as last resort
                if quote[symbol].bid_price and quote[symbol].ask_price:
                    price = (float(quote[symbol].bid_price) + float(quote[symbol].ask_price)) / 2
                else:
                    price = quote[symbol].ask_price or quote[symbol].bid_price
        
        if price and float(price) > 0:
            return float(price)
        
        return None
        
    except Exception as e:
        print(f"      Warning: Could not get exit price for {symbol}: {e}")
        return None

def calculate_exit_z_score(stock1, stock2, beta, entry_z, positions_dict):
    """
    Calculate current z-score using THE EXACT SAME statistics as entry.
    
    Uses the correct bid/ask prices based on position direction:
    - LONG position: use BID (we're selling)
    - SHORT position: use ASK (we're buying to cover)
    - No position: use last trade or mid price
    """
    try:
        # Load historical data (same as entry calculation)
        prices_df = pd.read_csv("data/sp500_prices_clean.csv")
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
        
        # Calculate historical spread using the SAME beta
        historical_spread = prices_df[stock1] - (beta * prices_df[stock2])
        
        # Get statistics from HISTORICAL data only (frozen at time of entry)
        mu = historical_spread.mean()
        sigma = historical_spread.std()
        
        # Get CURRENT exit prices (bid for longs, ask for shorts, or fallback)
        current_p1 = get_exit_price(stock1, positions_dict)
        current_p2 = get_exit_price(stock2, positions_dict)
        
        if current_p1 is None or current_p2 is None:
            print(f"      ERROR: Could not get valid prices ({stock1}={current_p1}, {stock2}={current_p2})")
            return None
        
        if current_p1 <= 0 or current_p2 <= 0:
            print(f"      ERROR: Invalid prices ({stock1}={current_p1}, {stock2}={current_p2})")
            return None
        
        # Calculate CURRENT spread using same beta
        current_spread = current_p1 - (beta * current_p2)
        
        # Calculate z-score: how many standard deviations is current spread from historical mean?
        current_z = (current_spread - mu) / sigma
        
        print(f"      Entry Z: {entry_z:.2f}, Exit Z: {current_z:.2f}, Change: {current_z - entry_z:.2f}")
        print(f"      Prices: {stock1}=${current_p1:.2f}, {stock2}=${current_p2:.2f}")
        print(f"      Spread: current={current_spread:.2f}, μ={mu:.2f}, σ={sigma:.2f}")
        
        return current_z
        
    except Exception as e:
        print(f"      Warning: Could not calculate z-score - {e}")
        import traceback
        traceback.print_exc()
        return None

def log_liquidated_trades(positions_before_liquidation):
    """
    Log all open trades as liquidated and REMOVE them from tracker.
    Uses stored hedge ratio from tracker for consistency.
    Takes positions snapshot from BEFORE liquidation.
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
        
        # Create positions dict for efficient lookup
        positions_dict = {pos.symbol: pos for pos in positions_before_liquidation}
        
        pairs_to_remove = []
        
        for idx, row in open_pairs.iterrows():
            stock1, stock2 = row['stock1'], row['stock2']
            
            # USE THE STORED HEDGE RATIO FROM TRACKER (exact same as entry!)
            beta = row['hedge_ratio']
            entry_z = row['z_score']
            
            print(f"\n    Processing {stock1}/{stock2} (beta={beta:.4f})...")
            
            exit_z = calculate_exit_z_score(stock1, stock2, beta, entry_z, positions_dict)
            
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
                    z_str = f"entry_z={entry_z:.2f}, exit_z={exit_z:.2f}" if exit_z is not None else "z=unknown"
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

# GET POSITIONS SNAPSHOT BEFORE LIQUIDATING
print("\nGetting positions snapshot...")
try:
    positions_before = trading_client.get_all_positions()
    print(f"✓ Captured {len(positions_before)} positions")
except Exception as e:
    print(f"✗ Error getting positions: {e}")
    positions_before = []

print("\nLiquidating all positions...")
try:
    if not positions_before:
        print("✓ No positions to liquidate")
    else:
        for position in positions_before:
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
        
        print(f"\n✓ Liquidation complete - closed {len(positions_before)} positions")
except Exception as e:
    print(f"✗ Error liquidating positions: {e}")

# Log all liquidated trades and REMOVE from tracker
# Pass the positions snapshot from BEFORE liquidation
print("\nUpdating trade records...")
log_liquidated_trades(positions_before)

print("\n" + "=" * 80)
print("Liquidation complete. Manual exits logged as neutral (not counted in win rate).")
print("All closed pairs removed from tracker.")
print("=" * 80)