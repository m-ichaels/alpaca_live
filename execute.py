import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from auth import KEY, SECRET

# Connect to Alpaca
trading_client = TradingClient(KEY, SECRET, paper=True)

# Load Kelly-sized signals (already checked for feasibility)
try:
    signals_df = pd.read_csv("data/sized_signals.csv")
    print("Using Kelly-sized signals from sizing.py")
except FileNotFoundError:
    print("ERROR: sized_signals.csv not found. Run sizing.py first!")
    exit(1)

print(f"Processing {len(signals_df)} pre-validated trading signals")
print(f"Expected Capital Deployment: ${signals_df['allocated_capital'].sum():,.2f}\n")

# Get current positions (double-check)
positions = trading_client.get_all_positions()
current_holdings = {p.symbol for p in positions}

executed_trades = []
skipped_trades = []

for idx, row in signals_df.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    signal = row['signal']
    
    print(f"\n--- Pair {idx + 1}/{len(signals_df)}: {stock1}/{stock2} ---")
    print(f"Signal: {signal}, Z-score: {row['z_score']:.2f}")
    print(f"Allocated Capital: ${row['allocated_capital']:,.2f}")
    print(f"Planned: {int(row['shares1'])} {stock1} @ ${row['price1']:.2f}, "
          f"{int(row['shares2'])} {stock2} @ ${row['price2']:.2f}")
    
    # Skip if positions exist (shouldn't happen if sizing.py ran recently)
    if stock1 in current_holdings or stock2 in current_holdings:
        print(f"⚠ Skipping - existing positions detected")
        skipped_trades.append({
            'stock1': stock1,
            'stock2': stock2,
            'reason': 'existing_positions',
            'allocated_capital': row['allocated_capital']
        })
        continue
    
    # Use pre-calculated shares from sizing.py
    shares1 = int(row['shares1'])
    shares2 = int(row['shares2'])
    
    # Determine order sides based on signal
    if signal == 'BUY':
        side1 = OrderSide.BUY
        side2 = OrderSide.SELL
    else:
        side1 = OrderSide.SELL
        side2 = OrderSide.BUY
    
    try:
        # Place first order
        order1 = trading_client.submit_order(MarketOrderRequest(
            symbol=stock1,
            qty=shares1,
            side=side1,
            time_in_force=TimeInForce.DAY
        ))
        
        # Place second order
        order2 = trading_client.submit_order(MarketOrderRequest(
            symbol=stock2,
            qty=shares2,
            side=side2,
            time_in_force=TimeInForce.DAY
        ))
        
        executed_trades.append({
            'stock1': stock1,
            'stock2': stock2,
            'action1': side1.value,
            'shares1': shares1,
            'price1': row['price1'],
            'action2': side2.value,
            'shares2': shares2,
            'price2': row['price2'],
            'signal': signal,
            'z_score': row['z_score'],
            'allocated_capital': row['allocated_capital'],
            'kelly_fraction': row['kelly_fraction'],
            'order1_id': order1.id,
            'order2_id': order2.id
        })
        
        print(f"✓ Executed:")
        print(f"  {stock1}: {side1.value} {shares1} @ ${row['price1']:.2f} (${shares1 * row['price1']:,.2f})")
        print(f"  {stock2}: {side2.value} {shares2} @ ${row['price2']:.2f} (${shares2 * row['price2']:,.2f})")
        
    except Exception as e:
        print(f"✗ Error executing trade: {e}")
        skipped_trades.append({
            'stock1': stock1,
            'stock2': stock2,
            'reason': str(e),
            'allocated_capital': row['allocated_capital']
        })

# Save results
print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)

if executed_trades:
    trades_df = pd.DataFrame(executed_trades)
    trades_df.to_csv("data/executed_trades.csv", index=False)
    
    print(f"✓ Successfully Executed: {len(executed_trades)} pairs")
    print(f"  Total Capital Deployed: ${trades_df['allocated_capital'].sum():,.2f}")
    print(f"  Average Position Size: ${trades_df['allocated_capital'].mean():,.2f}")
    print(f"  Total Kelly Fraction: {trades_df['kelly_fraction'].sum():.2%}")
else:
    print("✗ No trades executed")

if skipped_trades:
    skipped_df = pd.DataFrame(skipped_trades)
    skipped_df.to_csv("data/skipped_trades.csv", index=False)
    
    print(f"\n⚠ Skipped: {len(skipped_trades)} pairs")
    print(f"  Lost Capital: ${skipped_df['allocated_capital'].sum():,.2f}")
    print("\nSkipped trades saved to data/skipped_trades.csv")

print("=" * 80)

# Final efficiency metrics
if executed_trades:
    expected_total = signals_df['allocated_capital'].sum()
    actual_total = trades_df['allocated_capital'].sum()
    efficiency = (actual_total / expected_total) * 100 if expected_total > 0 else 0
    
    print(f"\nExecution Efficiency: {efficiency:.1f}%")
    print(f"Expected: ${expected_total:,.2f}")
    print(f"Achieved: ${actual_total:,.2f}")