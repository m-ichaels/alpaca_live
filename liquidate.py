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
    """Calculate current z-score for a pair"""
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
        
        # Calculate current spread
        current_spread = p1 - (beta * p2)
        
        # Get historical spread statistics
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
    Log all open trades as liquidated (neutral outcome - not counted as win or loss)
    """
    try:
        if not os.path.exists(TRACKER_FILE):
            return
        
        tracker = pd.read_csv(TRACKER_FILE)
        open_pairs = tracker[tracker['status'] == 'open']
        
        if len(open_pairs) == 0:
            return
        
        # Load pairs data for hedge ratios
        try:
            pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
        except FileNotFoundError:
            print("  Warning: cointegrated_pairs.csv not found, cannot calculate exit z-scores")
            pairs_df = None
        
        print("\n  Logging liquidated trades to history...")
        
        for _, row in open_pairs.iterrows():
            stock1, stock2 = row['stock1'], row['stock2']
            
            # Get hedge ratio and calculate exit z-score
            exit_z = None
            if pairs_df is not None:
                pair_info = pairs_df[
                    ((pairs_df['stock1'] == stock1) & (pairs_df['stock2'] == stock2)) |
                    ((pairs_df['stock1'] == stock2) & (pairs_df['stock2'] == stock1))
                ]
                
                if len(pair_info) > 0:
                    beta = pair_info.iloc[0]['hedge_ratio']
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
                    z_str = f"z={exit_z:.2f}" if exit_z is not None else "z=unknown"
                    print(f"    ✓ Logged {stock1}/{stock2} ({z_str})")
            
            # Mark as closed in tracker
            tracker.loc[
                ((tracker['stock1'] == stock1) & (tracker['stock2'] == stock2)) |
                ((tracker['stock1'] == stock2) & (tracker['stock2'] == stock1)),
                'status'
            ] = 'closed'
            tracker.loc[
                ((tracker['stock1'] == stock1) & (tracker['stock2'] == stock2)) |
                ((tracker['stock1'] == stock2) & (tracker['stock2'] == stock1)),
                'exit_date'
            ] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save updated tracker
        tracker.to_csv(TRACKER_FILE, index=False)
        print(f"  ✓ Marked {len(open_pairs)} pairs as closed in tracker")
        
    except Exception as e:
        print(f"  Warning: Could not log liquidated trades - {e}")

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

# Log all liquidated trades BEFORE clearing tracker
print("\nUpdating trade records...")
log_liquidated_trades()

print("\n" + "=" * 80)
print("Liquidation complete. Manual exits logged as neutral (not counted in win rate).")
print("=" * 80)