import pandas as pd
import time
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
try:
    from auth_local import KEY, SECRET
except ImportError:
    from auth import KEY, SECRET

# Setup Alpaca
tc = TradingClient(KEY, SECRET, paper=True)

TRACKER_FILE = "data/open_pairs.csv"
HISTORY_FILE = "data/trade_history.csv"

def initialize_tracker():
    """Initialize the open pairs tracker file if it doesn't exist"""
    if not os.path.exists(TRACKER_FILE):
        pd.DataFrame(columns=[
            'stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio',
            'shares1', 'shares2', 'capital_allocation', 'entry_date',
            'entry_price1', 'entry_price2', 'status', 'edge', 'p_target'
        ]).to_csv(TRACKER_FILE, index=False)

def initialize_history():
    """Initialize the trade history file if it doesn't exist"""
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=[
            'stock1', 'stock2', 'signal', 'z_score', 'entry_date',
            'entry_price1', 'entry_price2', 'capital_allocation',
            'exit_date', 'exit_z', 'exit_reason', 'win', 'edge'
        ]).to_csv(HISTORY_FILE, index=False)

def add_pair_to_tracker(pair_data):
    """Add a newly opened pair to the tracker"""
    try:
        # Load existing tracker
        if os.path.exists(TRACKER_FILE):
            tracker = pd.read_csv(TRACKER_FILE)
        else:
            tracker = pd.DataFrame()
        
        # Add new pair
        new_row = pd.DataFrame([pair_data])
        tracker = pd.concat([tracker, new_row], ignore_index=True)
        
        # Save
        tracker.to_csv(TRACKER_FILE, index=False)
        print(f"  [TRACKER] Added {pair_data['stock1']}/{pair_data['stock2']} to open_pairs.csv")
        
    except Exception as e:
        print(f"  [!] Warning: Could not add pair to tracker - {e}")

def add_trade_to_history(pair_data):
    """Add a newly opened trade to history for win rate tracking"""
    try:
        # Load existing history
        if os.path.exists(HISTORY_FILE):
            history = pd.read_csv(HISTORY_FILE)
        else:
            history = pd.DataFrame()
        
        # Add new trade (exit fields will be filled by tp_sl.py)
        new_row = pd.DataFrame([{
            'stock1': pair_data['stock1'],
            'stock2': pair_data['stock2'],
            'signal': pair_data['signal'],
            'z_score': pair_data['z_score'],
            'entry_date': pair_data['entry_date'],
            'entry_price1': pair_data['entry_price1'],
            'entry_price2': pair_data['entry_price2'],
            'capital_allocation': pair_data['capital_allocation'],
            'edge': pair_data.get('edge', None),
            'exit_date': None,
            'exit_z': None,
            'exit_reason': None,
            'win': None
        }])
        
        history = pd.concat([history, new_row], ignore_index=True)
        
        # Save
        history.to_csv(HISTORY_FILE, index=False)
        print(f"  [HISTORY] Added {pair_data['stock1']}/{pair_data['stock2']} to trade_history.csv")
        
    except Exception as e:
        print(f"  [!] Warning: Could not add trade to history - {e}")

def remove_closed_pairs_from_tracker(symbol):
    """Remove pairs from tracker if positions were closed outside this script"""
    try:
        if not os.path.exists(TRACKER_FILE):
            return
        
        tracker = pd.read_csv(TRACKER_FILE)
        
        # Find pairs involving this symbol
        mask = ((tracker['stock1'] == symbol) | (tracker['stock2'] == symbol)) & (tracker['status'] == 'open')
        
        if mask.any():
            tracker = tracker[~mask]
            tracker.to_csv(TRACKER_FILE, index=False)
            print(f"  [TRACKER] Removed pairs involving {symbol} from tracker")
            
    except Exception as e:
        print(f"  [!] Warning: Could not update tracker - {e}")

print("=" * 80)
print("PORTFOLIO EXECUTION - Transitioning to Recommended Portfolio")
print("=" * 80)

# Initialize tracking files
import os
os.makedirs("data", exist_ok=True)
initialize_tracker()
initialize_history()

# Load portfolio orders from reconciliation
try:
    orders_df = pd.read_csv("data/portfolio_orders.csv")
    print(f"\nLoaded {len(orders_df)} required actions from portfolio_orders.csv")
except FileNotFoundError:
    print("\nERROR: portfolio_orders.csv not found. Run comparison.py first.")
    exit()

if len(orders_df) == 0:
    print("\nNo actions required - portfolio already aligned!")
    exit()

# Load kelly positions for pair metadata
try:
    kelly_df = pd.read_csv("data/kelly_positions.csv")
    kelly_dict = {}
    for _, row in kelly_df.iterrows():
        kelly_dict[f"{row['stock1']}/{row['stock2']}"] = row.to_dict()
except FileNotFoundError:
    print("\nWARNING: kelly_positions.csv not found. Tracker will have limited data.")
    kelly_dict = {}

# Get current account state
acct = tc.get_account()
equity = float(acct.equity)
cash = float(acct.cash)

print(f"\nAccount Status:")
print(f"  Equity: ${equity:,.2f}")
print(f"  Cash Available: ${cash:,.2f}")

# Sort by priority to execute in correct order
orders_df = orders_df.sort_values(['priority', 'edge'], ascending=[True, False])

# Track execution results
executed = []
failed = []
skipped = []

# PRIORITY 0: Cancel Orders
print("\n" + "=" * 80)
print("PHASE 1: Canceling Orders")
print("=" * 80)

cancel_orders = orders_df[orders_df['action'] == 'CANCEL_ORDER']

if len(cancel_orders) > 0:
    for _, order in cancel_orders.iterrows():
        try:
            tc.cancel_order_by_id(order['order_id'])
            print(f"[OK] Canceled {order['symbol']} - {order['side']} {order['qty']} shares")
            executed.append({
                'phase': 'Cancel',
                'symbol': order['symbol'],
                'action': 'CANCEL_ORDER',
                'side': order['side'],
                'qty': order['qty'],
                'status': 'success',
                'order_id': order['order_id']
            })
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"[X] Failed to cancel {order['symbol']}: {e}")
            failed.append({
                'phase': 'Cancel',
                'symbol': order['symbol'],
                'action': 'CANCEL_ORDER',
                'error': str(e)
            })
else:
    print("No orders to cancel")

# PRIORITY 1: Close/Reduce Positions
print("\n" + "=" * 80)
print("PHASE 2: Closing/Reducing Positions")
print("=" * 80)

close_reduce_orders = orders_df[orders_df['priority'] == 1]

if len(close_reduce_orders) > 0:
    for _, order in close_reduce_orders.iterrows():
        symbol = order['symbol']
        qty = int(order['qty'])
        side_str = order['side']
        action = order['action']
        
        # Convert side string to OrderSide enum
        side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
        
        try:
            # Place market order
            submitted_order = tc.submit_order(MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            ))
            
            print(f"[OK] {action} {symbol} - {side_str.upper()} {qty} shares (Order ID: {submitted_order.id})")
            
            # Remove from tracker if closing
            if action == 'CLOSE':
                remove_closed_pairs_from_tracker(symbol)
            
            executed.append({
                'phase': 'Close/Reduce',
                'symbol': symbol,
                'action': action,
                'side': side_str,
                'qty': qty,
                'status': 'success',
                'order_id': submitted_order.id,
                'pair': order.get('pair', 'N/A')
            })
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"[X] Failed to {action} {symbol}: {e}")
            failed.append({
                'phase': 'Close/Reduce',
                'symbol': symbol,
                'action': action,
                'side': side_str,
                'qty': qty,
                'error': str(e)
            })
else:
    print("No positions to close or reduce")

# Wait for orders to settle
if len(close_reduce_orders) > 0:
    print("\nWaiting 5 seconds for orders to settle...")
    time.sleep(5)
    
    # Update cash after closing positions
    acct = tc.get_account()
    cash = float(acct.cash)
    print(f"Updated cash available: ${cash:,.2f}")

# PRIORITY 2: Open/Add Positions
print("\n" + "=" * 80)
print("PHASE 3: Opening/Adding Positions")
print("=" * 80)

open_add_orders = orders_df[orders_df['priority'] == 2].sort_values('edge', ascending=False)

if len(open_add_orders) > 0:
    # Group by pair to execute legs together
    pairs = {}
    for _, order in open_add_orders.iterrows():
        pair = order.get('pair', 'N/A')
        if pair not in pairs:
            pairs[pair] = []
        pairs[pair].append(order)
    
    # Execute each pair
    for pair_name, pair_orders in pairs.items():
        if pair_name == 'N/A':
            # Execute standalone orders (not tracking these as pairs)
            for order in pair_orders:
                symbol = order['symbol']
                qty = int(order['qty'])
                side_str = order['side']
                action = order['action']
                
                # Check cash availability
                estimated_cost = qty * order.get('target_price', 0)
                if estimated_cost > cash * 1.1:  # 10% buffer for slippage
                    print(f"[!] Skipping {symbol} - insufficient cash (need ~${estimated_cost:,.0f}, have ${cash:,.2f})")
                    skipped.append({
                        'phase': 'Open/Add',
                        'symbol': symbol,
                        'action': action,
                        'reason': 'insufficient_cash'
                    })
                    continue
                
                side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
                
                try:
                    submitted_order = tc.submit_order(MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    ))
                    
                    print(f"[OK] {action} {symbol} - {side_str.upper()} {qty} shares (Order ID: {submitted_order.id})")
                    
                    executed.append({
                        'phase': 'Open/Add',
                        'symbol': symbol,
                        'action': action,
                        'side': side_str,
                        'qty': qty,
                        'status': 'success',
                        'order_id': submitted_order.id,
                        'pair': pair_name
                    })
                    
                    # Update available cash estimate
                    cash -= estimated_cost if side_str == 'buy' else -estimated_cost
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"[X] Failed to {action} {symbol}: {e}")
                    failed.append({
                        'phase': 'Open/Add',
                        'symbol': symbol,
                        'action': action,
                        'error': str(e)
                    })
        else:
            # Execute pair legs together
            print(f"\nExecuting pair: {pair_name}")
            
            # Calculate total capital needed for pair
            total_cost = sum(order['qty'] * order.get('target_price', 0) for order in pair_orders)
            
            if total_cost > cash * 1.1:
                print(f"[!] Skipping pair {pair_name} - insufficient cash (need ~${total_cost:,.0f}, have ${cash:,.2f})")
                for order in pair_orders:
                    skipped.append({
                        'phase': 'Open/Add',
                        'symbol': order['symbol'],
                        'action': order['action'],
                        'pair': pair_name,
                        'reason': 'insufficient_cash'
                    })
                continue
            
            # Execute both legs
            pair_success = True
            pair_orders_submitted = []
            executed_prices = {}
            
            for order in pair_orders:
                symbol = order['symbol']
                qty = int(order['qty'])
                side_str = order['side']
                action = order['action']
                
                side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
                
                try:
                    submitted_order = tc.submit_order(MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    ))
                    
                    print(f"  [OK] {symbol} - {side_str.upper()} {qty} shares @ ~${order.get('target_price', 0):.2f}")
                    
                    # Store executed price for tracking
                    executed_prices[symbol] = order.get('target_price', 0)
                    
                    pair_orders_submitted.append({
                        'phase': 'Open/Add',
                        'symbol': symbol,
                        'action': action,
                        'side': side_str,
                        'qty': qty,
                        'status': 'success',
                        'order_id': submitted_order.id,
                        'pair': pair_name,
                        'edge': order.get('edge', 0)
                    })
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  [X] Failed {symbol}: {e}")
                    pair_success = False
                    failed.append({
                        'phase': 'Open/Add',
                        'symbol': symbol,
                        'action': action,
                        'pair': pair_name,
                        'error': str(e)
                    })
                    break
            
            if pair_success:
                executed.extend(pair_orders_submitted)
                cash -= total_cost
                print(f"  [OK] Pair {pair_name} executed successfully (Edge: {pair_orders[0].get('edge', 0):.4f})")
                
                # Add to tracker if this is a new pair opening
                if action == 'OPEN' and pair_name in kelly_dict:
                    kelly_data = kelly_dict[pair_name]
                    
                    # Determine stocks and entry prices
                    stock1 = kelly_data['stock1']
                    stock2 = kelly_data['stock2']
                    
                    tracker_data = {
                        'stock1': stock1,
                        'stock2': stock2,
                        'signal': kelly_data['signal'],
                        'z_score': kelly_data['z_score'],
                        'hedge_ratio': kelly_data['hedge_ratio'],
                        'shares1': kelly_data['shares1'],
                        'shares2': kelly_data['shares2'],
                        'capital_allocation': kelly_data['capital_allocation'],
                        'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price1': executed_prices.get(stock1, kelly_data['price1']),
                        'entry_price2': executed_prices.get(stock2, kelly_data['price2']),
                        'status': 'open',
                        'edge': kelly_data.get('edge', 0),
                        'p_target': kelly_data.get('p_target', 0)
                    }
                    
                    add_pair_to_tracker(tracker_data)
                    add_trade_to_history(tracker_data)
            else:
                print(f"  [X] Pair {pair_name} execution failed - may need manual cleanup")
else:
    print("No positions to open or add")

# Save execution results
print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)

if executed:
    executed_df = pd.DataFrame(executed)
    executed_df.to_csv("data/execution_log.csv", index=False)
    
    print(f"\n[OK] Successfully Executed: {len(executed)} actions")
    print(f"    Phase 1 (Cancel): {len(executed_df[executed_df['phase'] == 'Cancel'])}")
    print(f"    Phase 2 (Close/Reduce): {len(executed_df[executed_df['phase'] == 'Close/Reduce'])}")
    print(f"    Phase 3 (Open/Add): {len(executed_df[executed_df['phase'] == 'Open/Add'])}")
    
    # Show unique pairs traded
    pairs_executed = executed_df[executed_df['pair'] != 'N/A']['pair'].unique()
    if len(pairs_executed) > 0:
        print(f"\n    Pairs executed: {len(pairs_executed)}")
        for pair in pairs_executed:
            pair_data = executed_df[executed_df['pair'] == pair]
            if len(pair_data) > 0 and 'edge' in pair_data.columns:
                edge = pair_data['edge'].iloc[0]
                print(f"      - {pair} (Edge: {edge:.4f})")
            else:
                print(f"      - {pair}")

if failed:
    failed_df = pd.DataFrame(failed)
    failed_df.to_csv("data/execution_failures.csv", index=False)
    
    print(f"\n[X] Failed: {len(failed)} actions")
    for _, fail in failed_df.iterrows():
        print(f"    - {fail['symbol']} ({fail['action']}): {fail.get('error', 'Unknown error')}")

if skipped:
    skipped_df = pd.DataFrame(skipped)
    skipped_df.to_csv("data/execution_skipped.csv", index=False)
    
    print(f"\n[!] Skipped: {len(skipped)} actions")
    for reason in skipped_df['reason'].unique():
        count = len(skipped_df[skipped_df['reason'] == reason])
        print(f"    - {reason}: {count} actions")

# Final account status
acct = tc.get_account()
final_equity = float(acct.equity)
final_cash = float(acct.cash)

print(f"\n" + "-" * 80)
print("Final Account Status:")
print(f"  Equity: ${final_equity:,.2f} (change: ${final_equity - equity:,.2f})")
print(f"  Cash: ${final_cash:,.2f} (change: ${final_cash - cash:,.2f})")

# Get final positions
final_positions = tc.get_all_positions()
print(f"  Total Positions: {len(final_positions)}")

# Show open pairs being tracked
if os.path.exists(TRACKER_FILE):
    tracker = pd.read_csv(TRACKER_FILE)
    open_tracked = tracker[tracker['status'] == 'open']
    print(f"  Open Pairs Tracked: {len(open_tracked)}")

print("=" * 80)

if executed:
    print(f"\nExecution log saved to: data/execution_log.csv")
if failed:
    print(f"Failures saved to: data/execution_failures.csv")
if skipped:
    print(f"Skipped actions saved to: data/execution_skipped.csv")

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print(f"Open pairs tracker: {TRACKER_FILE}")
print(f"Trade history: {HISTORY_FILE}")
print("=" * 80)