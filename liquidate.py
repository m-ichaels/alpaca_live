import pandas as pd
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from auth import KEY, SECRET

# Connect to Alpaca Paper Trading
trading_client = TradingClient(KEY, SECRET, paper=True)

TRACKER_FILE = "data/open_pairs.csv"

print("Canceling all open orders...")
try:
    trading_client.cancel_orders()
    print("✓ All open orders canceled")
except Exception as e:
    print(f"Error canceling orders: {e}")

print("\nLiquidating all positions...")
try:
    positions = trading_client.get_all_positions()
    
    if not positions:
        print("No positions to liquidate")
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
                print(f"✓ {symbol}: {side.value} {qty} shares")
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
        
        print(f"\nLiquidation complete - closed {len(positions)} positions")
except Exception as e:
    print(f"Error liquidating positions: {e}")

print("\nClearing tracker file...")
try:
    if os.path.exists(TRACKER_FILE):
        tracker = pd.DataFrame(columns=[
            'stock1', 'stock2', 'signal', 'z_score', 
            'capital_allocation', 'entry_date', 
            'order1_id', 'order2_id', 'status', 'exit_date'
        ])
        tracker.to_csv(TRACKER_FILE, index=False)
        print("✓ Tracker file cleared")
except Exception as e:
    print(f"Error clearing tracker: {e}")