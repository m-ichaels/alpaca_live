import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus, QueryOrderStatus
try:
    from auth_local import KEY, SECRET  # For local testing
except ImportError:
    print("Import error")
    from auth import KEY, SECRET  # For GitHub Actions

# Setup Alpaca
tc = TradingClient(KEY, SECRET, paper=True)

print("=" * 80)
print("PORTFOLIO RECONCILIATION")
print("=" * 80)

# Get current account state
acct = tc.get_account()
equity = float(acct.equity)
cash = float(acct.cash)

print(f"\nAccount Status:")
print(f"  Equity: ${equity:,.2f}")
print(f"  Cash: ${cash:,.2f}")

# Get current positions
current_positions = tc.get_all_positions()
positions_dict = {p.symbol: {
    'qty': int(p.qty),
    'side': p.side,
    'market_value': float(p.market_value),
    'avg_entry_price': float(p.avg_entry_price),
    'current_price': float(p.current_price)
} for p in current_positions}

print(f"\nCurrent Positions: {len(positions_dict)}")
for symbol, pos in positions_dict.items():
    print(f"  {symbol}: {pos['qty']} shares @ ${pos['current_price']:.2f} = ${pos['market_value']:,.2f}")

# Get open orders
open_orders_request = GetOrdersRequest(
    status=QueryOrderStatus.OPEN
)
open_orders = tc.get_orders(filter=open_orders_request)
orders_dict = {}

for order in open_orders:
    symbol = order.symbol
    if symbol not in orders_dict:
        orders_dict[symbol] = []
    orders_dict[symbol].append({
        'id': order.id,
        'qty': int(order.qty) if order.qty else 0,
        'side': order.side.value,
        'type': order.type.value,
        'status': order.status.value
    })

print(f"\nOpen Orders: {len(orders_dict)} symbols")
for symbol, orders in orders_dict.items():
    for order in orders:
        print(f"  {symbol}: {order['side']} {order['qty']} shares ({order['status']})")

# Load recommended portfolio
try:
    recommended_df = pd.read_csv("data/kelly_positions.csv")
    print(f"\nRecommended Portfolio: {len(recommended_df)} pairs")
except FileNotFoundError:
    print("\nERROR: kelly_positions.csv not found. Run portfolio_kelly.py first.")
    exit()

# Build target positions from recommended portfolio with NETTING
# Track all position requirements by symbol
symbol_exposures = {}  # symbol -> {'long': qty, 'short': qty, 'pairs': [...]}

for _, row in recommended_df.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    signal = row['signal']
    shares1 = int(row['shares1'])
    shares2 = int(row['shares2'])
    pair_name = f"{stock1}/{stock2}"
    
    # Initialize symbols if not seen
    if stock1 not in symbol_exposures:
        symbol_exposures[stock1] = {'long': 0, 'short': 0, 'pairs': [], 'price': row['price1']}
    if stock2 not in symbol_exposures:
        symbol_exposures[stock2] = {'long': 0, 'short': 0, 'pairs': [], 'price': row['price2']}
    
    # Add exposure based on signal
    if signal == 'LONG_SPREAD':
        # Long stock1, Short stock2
        symbol_exposures[stock1]['long'] += shares1
        symbol_exposures[stock1]['pairs'].append({'pair': pair_name, 'side': 'long', 'qty': shares1, 'edge': row['edge']})
        
        symbol_exposures[stock2]['short'] += shares2
        symbol_exposures[stock2]['pairs'].append({'pair': pair_name, 'side': 'short', 'qty': shares2, 'edge': row['edge']})
    else:  # SHORT_SPREAD
        # Short stock1, Long stock2
        symbol_exposures[stock1]['short'] += shares1
        symbol_exposures[stock1]['pairs'].append({'pair': pair_name, 'side': 'short', 'qty': shares1, 'edge': row['edge']})
        
        symbol_exposures[stock2]['long'] += shares2
        symbol_exposures[stock2]['pairs'].append({'pair': pair_name, 'side': 'long', 'qty': shares2, 'edge': row['edge']})

# Net the positions
target_positions = {}

for symbol, exposure in symbol_exposures.items():
    long_qty = exposure['long']
    short_qty = exposure['short']
    
    # Net the position
    net_qty = long_qty - short_qty
    
    if net_qty > 0:
        # Net long position
        target_positions[symbol] = {
            'qty': net_qty,
            'side': 'long',
            'pair': ', '.join([p['pair'] for p in exposure['pairs']]),
            'signal': f"NET LONG (L:{long_qty} - S:{short_qty})",
            'edge': max([p['edge'] for p in exposure['pairs']]),  # Use best edge
            'capital': net_qty * exposure['price'],
            'price': exposure['price'],
            'pairs_detail': exposure['pairs']
        }
    elif net_qty < 0:
        # Net short position
        target_positions[symbol] = {
            'qty': abs(net_qty),
            'side': 'short',
            'pair': ', '.join([p['pair'] for p in exposure['pairs']]),
            'signal': f"NET SHORT (L:{long_qty} - S:{short_qty})",
            'edge': max([p['edge'] for p in exposure['pairs']]),
            'capital': abs(net_qty) * exposure['price'],
            'price': exposure['price'],
            'pairs_detail': exposure['pairs']
        }
    # If net_qty == 0, no position needed (perfectly hedged)

print(f"\nTarget Positions (NETTED): {len(target_positions)} stocks across {len(recommended_df)} pairs")

# Show netting summary
netted_symbols = [s for s, e in symbol_exposures.items() if e['long'] > 0 and e['short'] > 0]
if netted_symbols:
    print(f"\nNetting applied to {len(netted_symbols)} symbols:")
    for symbol in netted_symbols:
        exp = symbol_exposures[symbol]
        net = exp['long'] - exp['short']
        side = 'LONG' if net > 0 else 'SHORT' if net < 0 else 'FLAT'
        print(f"  {symbol}: L{exp['long']} - S{exp['short']} = {side} {abs(net)}")
        for p in exp['pairs']:
            print(f"    - {p['pair']}: {p['side']} {p['qty']}")

# Calculate required actions
actions = []

# 1. Check positions to close (in current but not in target)
for symbol in positions_dict.keys():
    if symbol not in target_positions:
        current_qty = positions_dict[symbol]['qty']
        current_side = positions_dict[symbol]['side']
        
        # Need to close this position
        action_side = 'sell' if current_side == 'long' else 'buy'
        actions.append({
            'symbol': symbol,
            'action': 'CLOSE',
            'side': action_side,
            'qty': current_qty,
            'reason': 'Not in recommended portfolio',
            'current_value': positions_dict[symbol]['market_value'],
            'pair': 'N/A',
            'edge': 0,
            'priority': 1  # Close first
        })

# 2. Check orders and handle wash trade prevention
# We need to cancel ALL existing orders for symbols where we'll be trading
# This prevents wash trade errors
for symbol in set(list(orders_dict.keys()) + list(target_positions.keys())):
    if symbol not in orders_dict:
        continue
        
    # Determine if we need to cancel existing orders for this symbol
    should_cancel_all = False
    
    if symbol in target_positions:
        target = target_positions[symbol]
        expected_side = 'buy' if target['side'] == 'long' else 'sell'
        
        # Check if ANY existing order conflicts with our intended action
        for order in orders_dict[symbol]:
            # If there's an order on the opposite side, we MUST cancel it to avoid wash trade
            if order['side'] != expected_side:
                should_cancel_all = True
                break
    else:
        # Symbol not in target - cancel all orders for it
        should_cancel_all = True
    
    if should_cancel_all:
        for order in orders_dict[symbol]:
            actions.append({
                'symbol': symbol,
                'action': 'CANCEL_ORDER',
                'order_id': order['id'],
                'side': order['side'],
                'qty': order['qty'],
                'reason': f"Prevent wash trade conflict (need clean slate for {symbol})",
                'current_value': 0,
                'pair': target_positions.get(symbol, {}).get('pair', 'N/A'),
                'edge': target_positions.get(symbol, {}).get('edge', 0),
                'priority': 0  # Cancel orders first
            })

# 3. Check positions to adjust or open
for symbol, target in target_positions.items():
    target_qty = target['qty']
    target_side = target['side']
    
    # Calculate current effective position
    current_qty = 0
    current_side = None
    
    if symbol in positions_dict:
        current_qty = positions_dict[symbol]['qty']
        current_side = positions_dict[symbol]['side']
    
    # NOTE: We're NOT counting pending orders since we're canceling all conflicting ones above
    # This ensures clean execution without wash trade issues
    
    # Determine needed action
    action_side = 'buy' if target_side == 'long' else 'sell'
    
    if current_side is None:
        # No current position - need to open
        actions.append({
            'symbol': symbol,
            'action': 'OPEN',
            'side': action_side,
            'qty': target_qty,
            'reason': f'Open new {target_side} position',
            'current_value': 0,
            'pair': target['pair'],
            'edge': target['edge'],
            'priority': 2,  # Open after closing
            'target_price': target['price']
        })
    elif current_side == target_side:
        # Same side - check if qty needs adjustment
        qty_diff = target_qty - current_qty
        
        if abs(qty_diff) >= 1:  # At least 1 share difference
            if qty_diff > 0:
                # Need to add shares
                actions.append({
                    'symbol': symbol,
                    'action': 'ADD',
                    'side': action_side,
                    'qty': abs(qty_diff),
                    'reason': f'Increase {target_side} position',
                    'current_value': positions_dict[symbol]['market_value'],
                    'pair': target['pair'],
                    'edge': target['edge'],
                    'priority': 2,
                    'target_price': target['price']
                })
            else:
                # Need to reduce shares
                reduce_side = 'sell' if target_side == 'long' else 'buy'
                actions.append({
                    'symbol': symbol,
                    'action': 'REDUCE',
                    'side': reduce_side,
                    'qty': abs(qty_diff),
                    'reason': f'Reduce {target_side} position',
                    'current_value': positions_dict[symbol]['market_value'],
                    'pair': target['pair'],
                    'edge': target['edge'],
                    'priority': 1
                })
    else:
        # Different side - need to close and reopen
        # First close current position
        close_side = 'sell' if current_side == 'long' else 'buy'
        actions.append({
            'symbol': symbol,
            'action': 'CLOSE',
            'side': close_side,
            'qty': current_qty,
            'reason': f'Close {current_side} to flip to {target_side}',
            'current_value': positions_dict[symbol]['market_value'],
            'pair': target['pair'],
            'edge': target['edge'],
            'priority': 1
        })
        
        # Then open new position
        actions.append({
            'symbol': symbol,
            'action': 'OPEN',
            'side': action_side,
            'qty': target_qty,
            'reason': f'Open {target_side} after flip',
            'current_value': 0,
            'pair': target['pair'],
            'edge': target['edge'],
            'priority': 2,
            'target_price': target['price']
        })

# Sort actions by priority
actions_df = pd.DataFrame(actions)

if len(actions_df) == 0:
    print("\n" + "=" * 80)
    print("NO ACTIONS NEEDED - Portfolio already aligned with recommendations")
    print("=" * 80)
    pd.DataFrame().to_csv("data/portfolio_orders.csv", index=False)
    exit()

actions_df = actions_df.sort_values(['priority', 'edge'], ascending=[True, False])

# Save to CSV
actions_df.to_csv("data/portfolio_orders.csv", index=False)

# Display summary
print("\n" + "=" * 80)
print("REQUIRED ACTIONS")
print("=" * 80)

print(f"\nTotal actions: {len(actions_df)}")
print(f"  Cancel orders: {len(actions_df[actions_df['action'] == 'CANCEL_ORDER'])}")
print(f"  Close positions: {len(actions_df[actions_df['action'] == 'CLOSE'])}")
print(f"  Reduce positions: {len(actions_df[actions_df['action'] == 'REDUCE'])}")
print(f"  Add to positions: {len(actions_df[actions_df['action'] == 'ADD'])}")
print(f"  Open new positions: {len(actions_df[actions_df['action'] == 'OPEN'])}")

# Calculate capital impact
close_value = actions_df[actions_df['action'].isin(['CLOSE', 'REDUCE'])]['current_value'].sum()
open_value = actions_df[actions_df['action'].isin(['OPEN', 'ADD'])].apply(
    lambda x: x['qty'] * x.get('target_price', 0) if 'target_price' in x else 0, axis=1
).sum()

print(f"\nCapital Impact:")
print(f"  Closing/Reducing: ${close_value:,.2f}")
print(f"  Opening/Adding: ${open_value:,.2f}")
print(f"  Net change: ${open_value - close_value:,.2f}")
print(f"  Required cash: ${max(0, open_value - close_value - cash):,.2f}")

# Display actions by priority
print("\n" + "-" * 80)
print("PRIORITY 0: Cancel Orders (Wash Trade Prevention)")
print("-" * 80)
cancel_actions = actions_df[actions_df['action'] == 'CANCEL_ORDER']
if len(cancel_actions) > 0:
    for _, action in cancel_actions.iterrows():
        print(f"  {action['symbol']}: Cancel {action['side']} order for {action['qty']} shares")
        print(f"    Reason: {action['reason']}")
else:
    print("  None")

print("\n" + "-" * 80)
print("PRIORITY 1: Close/Reduce Positions")
print("-" * 80)
close_actions = actions_df[actions_df['priority'] == 1]
if len(close_actions) > 0:
    for _, action in close_actions.iterrows():
        print(f"  {action['symbol']} ({action['pair']}): {action['action']} - {action['side']} {action['qty']} shares")
        print(f"    Reason: {action['reason']}")
        if action['current_value'] > 0:
            print(f"    Current value: ${action['current_value']:,.2f}")
else:
    print("  None")

print("\n" + "-" * 80)
print("PRIORITY 2: Open/Add Positions")
print("-" * 80)
open_actions = actions_df[actions_df['priority'] == 2].sort_values('edge', ascending=False)
if len(open_actions) > 0:
    for _, action in open_actions.iterrows():
        print(f"  {action['symbol']} ({action['pair']}): {action['action']} - {action['side']} {action['qty']} shares")
        print(f"    Reason: {action['reason']}")
        print(f"    Edge: {action['edge']:.4f}")
        if 'target_price' in action:
            print(f"    Target price: ${action['target_price']:.2f}, Est. value: ${action['qty'] * action['target_price']:,.2f}")
else:
    print("  None")

print("\n" + "=" * 80)
print(f"Saved detailed actions to data/portfolio_orders.csv")
print("=" * 80)

# Create execution summary grouped by pairs
print("\n" + "=" * 80)
print("EXECUTION SUMMARY BY PAIR")
print("=" * 80)

pair_summary = {}
for _, action in actions_df.iterrows():
    pair = action.get('pair', 'N/A')
    if pair not in pair_summary:
        pair_summary[pair] = {
            'actions': [],
            'edge': action.get('edge', 0)
        }
    pair_summary[pair]['actions'].append(action)

for pair, data in sorted(pair_summary.items(), key=lambda x: x[1]['edge'], reverse=True):
    if pair == 'N/A':
        continue
    print(f"\n{pair} (Edge: {data['edge']:.4f}):")
    for action in data['actions']:
        symbol = action['symbol']
        act = action['action']
        side = action['side']
        qty = action['qty']
        print(f"  {symbol}: {act} - {side} {qty} shares")

print("\n" + "=" * 80)
print("WASH TRADE PREVENTION SUMMARY")
print("=" * 80)
print(f"Total orders to cancel: {len(cancel_actions)}")
print("This ensures clean execution without opposite-side order conflicts.")
print("All conflicting orders will be canceled before new positions are opened.")
print("=" * 80)