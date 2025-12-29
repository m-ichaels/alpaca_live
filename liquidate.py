from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from auth import KEY, SECRET

# Connect to Alpaca Paper Trading
trading_client = TradingClient(KEY, SECRET, paper=True)

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
            
            # Determine side: if position is long (positive), sell; if short (negative), buy
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