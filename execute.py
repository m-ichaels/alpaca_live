import pandas as pd
import os
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from auth import KEY, SECRET

# ============================================================================
# PAIR TRACKER FUNCTIONS (built into execute.py)
# ============================================================================

TRACKER_FILE = "data/open_pairs.csv"

def initialize_tracker():
    """Initialize tracker file with proper columns if it doesn't exist or is empty"""
    try:
        # Check if file exists and has content
        if os.path.exists(TRACKER_FILE):
            try:
                tracker = pd.read_csv(TRACKER_FILE)
                if len(tracker.columns) > 0:
                    return  # File is valid, nothing to do
            except pd.errors.EmptyDataError:
                pass  # File is empty, will recreate
        
        # Create new tracker with proper columns
        tracker = pd.DataFrame(columns=[
            'stock1', 'stock2', 'signal', 'z_score', 
            'capital_allocation', 'entry_date', 
            'order1_id', 'order2_id', 'status', 'exit_date'
        ])
        tracker.to_csv(TRACKER_FILE, index=False)
        print(f"✓ Initialized tracker file: {TRACKER_FILE}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize tracker - {e}")

def add_open_pair(stock1, stock2, signal, z_score, capital_allocation, order1_id, order2_id):
    """Record a newly opened pair position"""
    try:
        # Ensure tracker is initialized
        initialize_tracker()
        
        tracker = pd.read_csv(TRACKER_FILE)
        
        new_entry = pd.DataFrame([{
            'stock1': stock1,
            'stock2': stock2,
            'signal': signal,
            'z_score': z_score,
            'capital_allocation': capital_allocation,
            'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'order1_id': order1_id,
            'order2_id': order2_id,
            'status': 'open',
            'exit_date': ''
        }])
        
        tracker = pd.concat([tracker, new_entry], ignore_index=True)
        tracker.to_csv(TRACKER_FILE, index=False)
        
        print(f"  ✓ Tracked in open_pairs.csv")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not track pair - {e}")


def reconcile_with_alpaca(trading_client):
    """Reconcile tracker with actual Alpaca positions"""
    try:
        # Ensure tracker is initialized
        initialize_tracker()
        
        tracker = pd.read_csv(TRACKER_FILE)
        
        if len(tracker) == 0:
            print("  ✓ No open pairs to reconcile\n")
            return
        
        open_tracker = tracker[tracker['status'] == 'open']
        
        if len(open_tracker) == 0:
            print("  ✓ No open pairs to reconcile\n")
            return
        
        # Get current Alpaca positions
        positions = trading_client.get_all_positions()
        current_holdings = {p.symbol for p in positions}
        
        print("\n--- Reconciling tracker with Alpaca positions ---")
        
        changes_made = False
        for idx, row in open_tracker.iterrows():
            stock1, stock2 = row['stock1'], row['stock2']
            
            # If either leg is missing from Alpaca, close the pair in tracker
            if stock1 not in current_holdings or stock2 not in current_holdings:
                print(f"  ⚠️  Pair {stock1}/{stock2} incomplete - closing in tracker")
                tracker.loc[idx, 'status'] = 'closed'
                tracker.loc[idx, 'exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                changes_made = True
        
        if changes_made:
            tracker.to_csv(TRACKER_FILE, index=False)
            print("  ✓ Reconciliation complete\n")
        else:
            print("  ✓ All tracked pairs match Alpaca\n")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not reconcile - {e}\n")

# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Initialize tracker file
initialize_tracker()

# Connect to Alpaca
trading_client = TradingClient(KEY, SECRET, paper=True)

# Reconcile tracker with actual positions first
print("Reconciling tracker with Alpaca positions...")
reconcile_with_alpaca(trading_client)

# Load Kelly-sized signals (already checked for feasibility)
try:
    signals_df = pd.read_csv("data/sized_signals.csv")
    print("Using Kelly-sized signals from sizing.py")
except FileNotFoundError:
    print("ERROR: sized_signals.csv not found. Run sizing.py first!")
    exit(1)

print(f"Processing {len(signals_df)} pre-validated trading signals")
print(f"Expected Capital Deployment: ${signals_df['capital_allocation'].sum():,.2f}\n")

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
    print(f"Allocated Capital: ${row['capital_allocation']:,.2f}")
    print(f"Planned: {int(row['shares1'])} {stock1} @ ${row['price1']:.2f}, "
          f"{int(row['shares2'])} {stock2} @ ${row['price2']:.2f}")
    
    # Skip if positions exist (shouldn't happen if sizing.py ran recently)
    if stock1 in current_holdings or stock2 in current_holdings:
        print(f"⚠️  Skipping - existing positions detected")
        skipped_trades.append({
            'stock1': stock1,
            'stock2': stock2,
            'reason': 'existing_positions',
            'capital_allocation': row['capital_allocation']
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
        
        print(f"✓ Executed:")
        print(f"  {stock1}: {side1.value} {shares1} @ ${row['price1']:.2f} (${shares1 * row['price1']:,.2f})")
        print(f"  {stock2}: {side2.value} {shares2} @ ${row['price2']:.2f} (${shares2 * row['price2']:,.2f})")
        
        # Track this pair as open
        add_open_pair(
            stock1=stock1,
            stock2=stock2,
            signal=signal,
            z_score=row['z_score'],
            capital_allocation=row['capital_allocation'],
            order1_id=order1.id,
            order2_id=order2.id
        )
        
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
            'capital_allocation': row['capital_allocation'],
            'kelly_fraction': row['kelly_fraction'],
            'order1_id': order1.id,
            'order2_id': order2.id
        })
        
    except Exception as e:
        print(f"✗ Error executing trade: {e}")
        skipped_trades.append({
            'stock1': stock1,
            'stock2': stock2,
            'reason': str(e),
            'capital_allocation': row['capital_allocation']
        })

# Save results
print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)

if executed_trades:
    trades_df = pd.DataFrame(executed_trades)
    trades_df.to_csv("data/executed_trades.csv", index=False)
    
    print(f"✓ Successfully Executed: {len(executed_trades)} pairs")
    print(f"  Total Capital Deployed: ${trades_df['capital_allocation'].sum():,.2f}")
    print(f"  Average Position Size: ${trades_df['capital_allocation'].mean():,.2f}")
    print(f"  Total Kelly Fraction: {trades_df['kelly_fraction'].sum():.2%}")
else:
    print("✗ No trades executed")

if skipped_trades:
    skipped_df = pd.DataFrame(skipped_trades)
    skipped_df.to_csv("data/skipped_trades.csv", index=False)
    
    print(f"\n⚠️  Skipped: {len(skipped_trades)} pairs")
    print(f"  Lost Capital: ${skipped_df['capital_allocation'].sum():,.2f}")
    print("\nSkipped trades saved to data/skipped_trades.csv")

print("=" * 80)

# Final efficiency metrics
if executed_trades:
    expected_total = signals_df['capital_allocation'].sum()
    actual_total = trades_df['capital_allocation'].sum()
    efficiency = (actual_total / expected_total) * 100 if expected_total > 0 else 0
    
    print(f"\nExecution Efficiency: {efficiency:.1f}%")
    print(f"Expected: ${expected_total:,.2f}")
    print(f"Achieved: ${actual_total:,.2f}")