import pandas as pd
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
try:
    from auth_local import KEY, SECRET
except ImportError:
    from auth import KEY, SECRET

# Setup Alpaca
tc = TradingClient(KEY, SECRET, paper=True)

print("=" * 80)
print("PORTFOLIO EXECUTION - Transitioning to Recommended Portfolio")
print("=" * 80)

# Load portfolio orders from reconciliation
try:
    orders_df = pd.read_csv("data/portfolio_orders.csv")
    print(f"\nLoaded {len(orders_df)} required actions from portfolio_orders.csv")
except FileNotFoundError:
    print("\nERROR: portfolio_orders.csv not found. Run portfolio_reconciliation.py first.")
    exit()

if len(orders_df) == 0:
    print("\nNo actions required - portfolio already aligned!")
    exit()

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
            # Execute standalone orders
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

print("=" * 80)

if executed:
    print(f"\nExecution log saved to: data/execution_log.csv")
if failed:
    print(f"Failures saved to: data/execution_failures.csv")
if skipped:
    print(f"Skipped actions saved to: data/execution_skipped.csv")

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)