import pandas as pd
import numpy as np
import os
import importlib.util
from scipy.optimize import linprog
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET

# Load TP/SL parameters from tp_sl.py
spec = importlib.util.spec_from_file_location("tp_sl", "tp_sl.py")
tp_sl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp_sl_module)

TAKE_PROFIT_Z = tp_sl_module.TAKE_PROFIT_Z
STOP_LOSS_Z = tp_sl_module.STOP_LOSS_Z

# Connect to Alpaca
trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

# Load signals and historical prices for spread statistics
signals_df = pd.read_csv("data/entry_signals.csv")
prices_df = pd.read_csv("data/sp500_prices_clean.csv")
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df = prices_df.set_index('date')

# Get account equity
account = trading_client.get_account()
total_equity = float(account.equity)

print(f"Total Account Equity: ${total_equity:,.2f}")
print(f"Processing {len(signals_df)} signals for Kelly-optimal sizing\n")

# Get current positions to exclude from allocation
positions = trading_client.get_all_positions()
current_holdings = {p.symbol for p in positions}

def calculate_adaptive_win_probability(z_score):
    """
    Calculate win probability using historical trade data when available,
    falling back to theoretical baseline.
    
    NOTE: Only uses trades with definitive outcomes (win=0 or win=1).
    Manual liquidations (win=None) are excluded from win rate calculations.
    """
    theoretical = 0.51
    
    if os.path.exists("data/trade_history.csv"):
        try:
            history = pd.read_csv("data/trade_history.csv")
            
            completed = history[
                (history['exit_date'].notna()) & 
                (history['win'].notna())
            ]
            
            if len(completed) >= 10:
                empirical_win_rate = completed['win'].mean()
                blended = 0.6 * empirical_win_rate + 0.4 * theoretical
                print(f"  Using blended win prob: "
                      f"{blended:.1%} (empirical: {empirical_win_rate:.1%}, "
                      f"n={len(completed)} definitive outcomes)")
                return blended
                    
        except Exception as e:
            print(f"  Warning: Could not load trade history - {e}")
    
    return theoretical

def calculate_kelly_parameters(z_score, spread_mean, spread_std):
    """
    Calculate Kelly based on actual TP/SL levels from tp_sl.py
    """
    current_z = z_score
    abs_z = abs(current_z)
    
    if current_z > 0:
        profit_distance = abs(current_z - TAKE_PROFIT_Z)
        loss_distance = abs(STOP_LOSS_Z - current_z)
    else:
        profit_distance = abs(current_z + TAKE_PROFIT_Z)
        loss_distance = abs(-STOP_LOSS_Z - current_z)
    
    expected_profit = profit_distance * spread_std
    expected_loss = loss_distance * spread_std
    
    win_loss_ratio = expected_profit / expected_loss if expected_loss > 0 else 1.0
    win_prob = calculate_adaptive_win_probability(z_score)
    
    loss_prob = 1 - win_prob
    kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
    
    # Conservative 1/3 Kelly
    kelly_fraction = max(0, kelly_fraction) * 0.33
    
    # Cap at 8% per pair
    kelly_fraction = min(kelly_fraction, 0.08)
    
    return kelly_fraction, win_prob, win_loss_ratio

def find_feasible_allocations(stock1, stock2, beta, price1, price2, 
                               min_pct=0.01, max_pct=0.08, equity=100000,
                               max_hedge_error=5.0):
    """
    Find all capital allocations that can be executed with integer shares
    and acceptable hedge error.
    
    Returns list of dicts with: shares1, shares2, capital, pct, hedge_error
    """
    feasible = []
    min_capital = min_pct * equity
    max_capital = max_pct * equity
    
    # Calculate reasonable range for shares1
    # Min: enough to reach min_capital
    min_shares1 = max(1, int(min_capital / (price1 * (1 + 1/abs(beta)))))
    # Max: don't exceed max_capital
    max_shares1 = int(max_capital / price1) + 10  # +10 buffer
    
    for shares1 in range(min_shares1, min(max_shares1, 500)):  # cap at 500 to avoid infinite loops
        # Calculate required shares2 for hedge
        target_shares2 = (shares1 * price1) / (abs(beta) * price2)
        
        # Try both floor and ceil to find best hedge
        for shares2 in [int(target_shares2), int(target_shares2) + 1]:
            if shares2 < 1:
                continue
            
            # Check hedge error
            value1 = shares1 * price1
            value2 = shares2 * price2
            actual_ratio = value1 / value2
            hedge_error = abs(actual_ratio - abs(beta)) / abs(beta) * 100
            
            if hedge_error <= max_hedge_error:
                total_capital = value1 + value2
                pct = total_capital / equity
                
                if min_pct <= pct <= max_pct:
                    # Check if we already have a very similar allocation
                    is_duplicate = any(
                        abs(f['capital'] - total_capital) < 10  # within $10
                        for f in feasible
                    )
                    
                    if not is_duplicate:
                        feasible.append({
                            'shares1': shares1,
                            'shares2': shares2,
                            'capital': total_capital,
                            'pct': pct,
                            'hedge_error': hedge_error
                        })
    
    return feasible

def get_prices(stock1, stock2):
    """Get current prices for a stock pair."""
    try:
        quotes = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[stock1, stock2])
        )
        
        price1 = (quotes[stock1].bid_price + quotes[stock1].ask_price) / 2
        price2 = (quotes[stock2].bid_price + quotes[stock2].ask_price) / 2
        
        if price1 <= 0 or price2 <= 0:
            return None, None
        
        return price1, price2
        
    except Exception as e:
        return None, None

# Display trade history stats if available
if os.path.exists("data/trade_history.csv"):
    try:
        history = pd.read_csv("data/trade_history.csv")
        completed = history[history['exit_date'].notna()]
        definitive = completed[completed['win'].notna()]
        liquidated = completed[completed['win'].isna()]
        
        if len(completed) > 0:
            print("\n" + "=" * 80)
            print("TRADE HISTORY SUMMARY")
            print("=" * 80)
            print(f"Total completed trades: {len(completed)}")
            print(f"  Definitive outcomes (used for win rate): {len(definitive)}")
            if len(definitive) > 0:
                print(f"    Wins: {definitive['win'].sum():.0f}")
                print(f"    Losses: {(definitive['win'] == 0).sum():.0f}")
                print(f"    Win rate: {definitive['win'].mean():.1%}")
            print(f"  Manual liquidations (neutral, excluded): {len(liquidated)}")
            print("=" * 80 + "\n")
    except Exception as e:
        print(f"Warning: Could not load trade history stats - {e}\n")

# ============================================================================
# PASS 1: Calculate raw Kelly fractions for each pair
# ============================================================================
print("PASS 1: Calculating raw Kelly fractions...")
print("=" * 80)

raw_kelly_data = []

for idx, row in signals_df.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    
    # Skip if already have positions
    if stock1 in current_holdings or stock2 in current_holdings:
        continue
    
    z_score = row['z_score']
    beta = row['hedge_ratio']
    abs_z = abs(z_score)
    
    # Calculate spread statistics from historical data
    spread = prices_df[stock1] - beta * prices_df[stock2]
    spread_mean = spread.mean()
    spread_std = spread.std()
    
    # Calculate Kelly using actual TP/SL levels
    kelly_fraction, win_prob, win_loss_ratio = calculate_kelly_parameters(
        z_score, spread_mean, spread_std
    )
    
    raw_kelly_data.append({
        'stock1': row['stock1'],
        'stock2': row['stock2'],
        'signal': row['signal'],
        'z_score': row['z_score'],
        'hedge_ratio': row['hedge_ratio'],
        'win_prob': win_prob,
        'win_loss_ratio': win_loss_ratio,
        'raw_kelly_fraction': kelly_fraction,
        'priority': abs_z
    })

kelly_df = pd.DataFrame(raw_kelly_data)
kelly_df = kelly_df.sort_values('priority', ascending=False).reset_index(drop=True)

print(f"Total pairs: {len(kelly_df)}")
print(f"Raw Kelly sum: {kelly_df['raw_kelly_fraction'].sum():.2%}")

# CRITICAL: Filter out pairs with zero or negative Kelly
positive_kelly_df = kelly_df[kelly_df['raw_kelly_fraction'] > 0.001].copy()
zero_kelly_count = len(kelly_df) - len(positive_kelly_df)

if zero_kelly_count > 0:
    print(f"\n⚠ Filtered out {zero_kelly_count} pairs with zero/negative Kelly (no positive expectancy)")

if len(positive_kelly_df) == 0:
    print("\n❌ No pairs with positive Kelly found - nothing to trade!")
    pd.DataFrame(columns=['stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio', 
                          'kelly_fraction', 'capital_allocation', 'shares1', 'shares2',
                          'price1', 'price2', 'win_prob', 'win_loss_ratio']).to_csv("data/sized_signals.csv", index=False)
    exit(0)

kelly_df = positive_kelly_df
print(f"Pairs with positive Kelly: {len(kelly_df)}")
print(f"Positive Kelly sum: {kelly_df['raw_kelly_fraction'].sum():.2%}\n")

# ============================================================================
# PASS 2: Find all feasible integer allocations for each pair
# ============================================================================
print("PASS 2: Finding feasible integer allocations...")
print("=" * 80)

enriched_kelly_data = []

for idx, row in kelly_df.iterrows():
    # Get current prices
    p1, p2 = get_prices(row['stock1'], row['stock2'])
    
    if p1 is None or p2 is None:
        print(f"[X] {row['stock1']}/{row['stock2']}: Could not get prices")
        continue
    
    # Find all feasible allocations for this pair
    feasible = find_feasible_allocations(
        row['stock1'], row['stock2'], row['hedge_ratio'],
        p1, p2,
        min_pct=0.01,  # minimum 1% of account
        max_pct=0.08,  # maximum 8% of account
        equity=total_equity,
        max_hedge_error=5.0
    )
    
    if len(feasible) == 0:
        print(f"[X] {row['stock1']}/{row['stock2']}: No feasible integer allocations found")
        continue
    
    # Store pair with all its feasible allocations
    enriched_kelly_data.append({
        **row.to_dict(),
        'price1': p1,
        'price2': p2,
        'feasible_allocations': feasible,
        'min_capital': min(f['capital'] for f in feasible),
        'max_capital': max(f['capital'] for f in feasible),
        'min_fraction': min(f['pct'] for f in feasible),
        'max_fraction': max(f['pct'] for f in feasible),
        'num_feasible': len(feasible)
    })
    
    print(f"[✓] {row['stock1']}/{row['stock2']}: {len(feasible)} feasible allocations "
          f"(${min(f['capital'] for f in feasible):,.0f} - ${max(f['capital'] for f in feasible):,.0f})")

enriched_df = pd.DataFrame(enriched_kelly_data)

if len(enriched_df) == 0:
    print("\n❌ No feasible pairs found!")
    pd.DataFrame(columns=['stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio', 
                          'kelly_fraction', 'capital_allocation', 'shares1', 'shares2',
                          'price1', 'price2', 'win_prob', 'win_loss_ratio']).to_csv("data/sized_signals.csv", index=False)
    exit(0)

print(f"\n✓ Feasible pairs: {len(enriched_df)}/{len(kelly_df)}")
print(f"Feasible Kelly sum: {enriched_df['raw_kelly_fraction'].sum():.2%}\n")

# ============================================================================
# PASS 3: Calculate target allocations (continuous Kelly-optimal)
# ============================================================================
print("PASS 3: Calculating continuous Kelly-optimal targets...")
print("=" * 80)

raw_kelly_sum = enriched_df['raw_kelly_fraction'].sum()
target_fraction = 0.95  # 95% of equity
target_capital = target_fraction * total_equity

# Calculate unconstrained scaling factor
unconstrained_scale = target_fraction / raw_kelly_sum

print(f"Target allocation: {target_fraction:.1%} (${target_capital:,.2f})")
print(f"Unconstrained scale: {unconstrained_scale:.4f}\n")

# Calculate ideal continuous targets
enriched_df['target_fraction'] = enriched_df['raw_kelly_fraction'] * unconstrained_scale
enriched_df['target_capital'] = enriched_df['target_fraction'] * total_equity

# Show targets
for idx, row in enriched_df.iterrows():
    print(f"[TARGET] {row['stock1']}/{row['stock2']}: "
          f"${row['target_capital']:,.2f} ({row['target_fraction']:.2%})")

# ============================================================================
# PASS 4: Snap to nearest feasible integer allocations
# ============================================================================
print("\n" + "=" * 80)
print("PASS 4: Snapping to nearest feasible integer allocations...")
print("=" * 80)

snapped_allocations = []

for idx, row in enriched_df.iterrows():
    target = row['target_capital']
    feasible = row['feasible_allocations']
    
    # Find closest feasible allocation to target
    best = min(feasible, key=lambda x: abs(x['capital'] - target))
    
    snap_error = abs(best['capital'] - target) / target * 100
    
    snapped_allocations.append({
        **row.to_dict(),
        'snapped_shares1': best['shares1'],
        'snapped_shares2': best['shares2'],
        'snapped_capital': best['capital'],
        'snapped_fraction': best['pct'],
        'snapped_hedge_error': best['hedge_error'],
        'snap_error_pct': snap_error
    })
    
    print(f"[SNAP] {row['stock1']}/{row['stock2']}: "
          f"${target:,.2f} → ${best['capital']:,.2f} "
          f"(error: {snap_error:.1f}%)")

snapped_df = pd.DataFrame(snapped_allocations)

# Check if total snapped allocation exceeds target
total_snapped = snapped_df['snapped_capital'].sum()
total_snapped_pct = total_snapped / total_equity

print(f"\nTotal snapped allocation: ${total_snapped:,.2f} ({total_snapped_pct:.1%})")
print(f"Target was: ${target_capital:,.2f} ({target_fraction:.1%})")

# ============================================================================
# PASS 5: Optimize for diversification using Linear Programming
# ============================================================================
if total_snapped > target_capital:
    print("\n" + "=" * 80)
    print("PASS 5: Over-allocated - optimizing for maximum diversification...")
    print("=" * 80)
    print(f"Need to reduce by: ${total_snapped - target_capital:,.2f}\n")
    
    # ========================================================================
    # Mixed Integer Programming (MIP) for optimal selection
    # 
    # For each pair, choose ONE allocation from its feasible set (or skip it)
    # Objective: Maximize number of pairs included + minimize Kelly deviation
    # Constraint: Total capital <= target_capital
    # ========================================================================
    
    # Build decision variables: one binary var per (pair, allocation) combination
    variables = []
    pair_indices = []
    
    for pair_idx, row in snapped_df.iterrows():
        for alloc_idx, alloc in enumerate(row['feasible_allocations']):
            variables.append({
                'pair_idx': pair_idx,
                'pair_name': f"{row['stock1']}/{row['stock2']}",
                'alloc_idx': alloc_idx,
                'capital': alloc['capital'],
                'kelly': row['raw_kelly_fraction'],
                'target': row['target_capital'],
                'allocation': alloc
            })
        pair_indices.append(pair_idx)
    
    n_vars = len(variables)
    print(f"Decision variables: {n_vars}")
    print(f"Pairs: {len(pair_indices)}\n")
    
    # Objective: maximize diversification (number of pairs) while minimizing deviation from Kelly
    # Negative coefficients because linprog minimizes (we want to maximize pairs)
    # Give each allocation from same pair equal weight so we maximize PAIRS not allocations
    c = np.zeros(n_vars)
    for i, var in enumerate(variables):
        n_allocations_for_pair = len(snapped_df.loc[var['pair_idx']]['feasible_allocations'])
        c[i] = -1.0 / n_allocations_for_pair  # Equal weight per pair
    
    # Constraint 1: Total capital <= target
    A_capital = np.array([[var['capital'] for var in variables]])
    b_capital = np.array([target_capital])
    
    # Constraint 2: At most one allocation per pair
    A_pairs = []
    b_pairs = []
    
    for pair_idx in pair_indices:
        row = np.zeros(n_vars)
        for i, var in enumerate(variables):
            if var['pair_idx'] == pair_idx:
                row[i] = 1
        A_pairs.append(row)
        b_pairs.append(1)  # At most 1 allocation per pair
    
    A_ub = np.vstack([A_capital, np.array(A_pairs)])
    b_ub = np.concatenate([b_capital, np.array(b_pairs)])
    
    # Bounds: binary variables (0 or 1) - LP relaxation
    bounds = [(0, 1) for _ in range(n_vars)]
    
    # Solve LP relaxation
    print("Solving linear programming relaxation...")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        print(f"❌ Optimization failed: {result.message}")
        print("Falling back to snapped allocations with smallest Kelly removed...\n")
        
        # Emergency fallback: remove smallest Kelly pairs until we fit
        final_allocations = []
        sorted_pairs = snapped_df.sort_values('raw_kelly_fraction', ascending=False)
        remaining_budget = target_capital
        
        for idx, row in sorted_pairs.iterrows():
            if row['snapped_capital'] <= remaining_budget:
                final_allocations.append({
                    'stock1': row['stock1'],
                    'stock2': row['stock2'],
                    'signal': row['signal'],
                    'z_score': row['z_score'],
                    'hedge_ratio': row['hedge_ratio'],
                    'win_prob': row['win_prob'],
                    'win_loss_ratio': row['win_loss_ratio'],
                    'raw_kelly_fraction': row['raw_kelly_fraction'],
                    'target_capital': row['target_capital'],
                    'shares1': row['snapped_shares1'],
                    'shares2': row['snapped_shares2'],
                    'capital_allocation': row['snapped_capital'],
                    'kelly_fraction': row['snapped_fraction'],
                    'price1': row['price1'],
                    'price2': row['price2'],
                    'hedge_error': row['snapped_hedge_error']
                })
                remaining_budget -= row['snapped_capital']
                print(f"[INCLUDED] {row['stock1']}/{row['stock2']}: ${row['snapped_capital']:,.2f}")
            else:
                print(f"[SKIPPED] {row['stock1']}/{row['stock2']}: ${row['snapped_capital']:,.2f} exceeds budget")
        
        final_df = pd.DataFrame(final_allocations)
    else:
        print("✓ Optimization successful\n")
        
        # Round LP solution to binary (greedy rounding based on LP weights)
        selected = []
        used_pairs = set()
        remaining_budget = target_capital
        
        # Sort by LP solution value (higher = more likely to be selected)
        candidates = [(i, result.x[i], variables[i]) for i in range(n_vars)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print("Greedy rounding to integer solution...\n")
        
        for i, weight, var in candidates:
            if var['pair_idx'] in used_pairs:
                continue
            
            if var['capital'] <= remaining_budget:
                selected.append(var)
                used_pairs.add(var['pair_idx'])
                remaining_budget -= var['capital']
        
        print(f"Selected {len(selected)}/{len(snapped_df)} pairs\n")
        
        final_allocations = []
        for var in selected:
            row = snapped_df.loc[var['pair_idx']]
            alloc = var['allocation']
            
            final_allocations.append({
                'stock1': row['stock1'],
                'stock2': row['stock2'],
                'signal': row['signal'],
                'z_score': row['z_score'],
                'hedge_ratio': row['hedge_ratio'],
                'win_prob': row['win_prob'],
                'win_loss_ratio': row['win_loss_ratio'],
                'raw_kelly_fraction': row['raw_kelly_fraction'],
                'target_capital': row['target_capital'],
                'shares1': alloc['shares1'],
                'shares2': alloc['shares2'],
                'capital_allocation': alloc['capital'],
                'kelly_fraction': alloc['pct'],
                'price1': row['price1'],
                'price2': row['price2'],
                'hedge_error': alloc['hedge_error']
            })
            
            deviation = (alloc['capital'] - row['target_capital']) / row['target_capital'] * 100
            print(f"[SELECTED] {row['stock1']}/{row['stock2']}: "
                  f"Target ${row['target_capital']:,.2f} → ${alloc['capital']:,.2f} ({deviation:+.1f}%)")
        
        # Check if we have remaining budget to add more pairs
        total_allocated = sum(p['capital_allocation'] for p in final_allocations)
        remaining = target_capital - total_allocated
        
        if remaining > total_equity * 0.02:  # More than 2% remaining
            print(f"\n💡 Remaining budget: ${remaining:,.2f} - attempting to include more pairs...\n")
            
            # Try to add pairs that weren't selected
            unselected_pairs = [idx for idx in pair_indices if idx not in used_pairs]
            
            for pair_idx in unselected_pairs:
                row = snapped_df.loc[pair_idx]
                feasible = row['feasible_allocations']
                
                # Find largest allocation that fits
                valid = [f for f in feasible if f['capital'] <= remaining]
                
                if len(valid) > 0:
                    best = max(valid, key=lambda x: x['capital'])
                    
                    final_allocations.append({
                        'stock1': row['stock1'],
                        'stock2': row['stock2'],
                        'signal': row['signal'],
                        'z_score': row['z_score'],
                        'hedge_ratio': row['hedge_ratio'],
                        'win_prob': row['win_prob'],
                        'win_loss_ratio': row['win_loss_ratio'],
                        'raw_kelly_fraction': row['raw_kelly_fraction'],
                        'target_capital': row['target_capital'],
                        'shares1': best['shares1'],
                        'shares2': best['shares2'],
                        'capital_allocation': best['capital'],
                        'kelly_fraction': best['pct'],
                        'price1': row['price1'],
                        'price2': row['price2'],
                        'hedge_error': best['hedge_error']
                    })
                    
                    remaining -= best['capital']
                    print(f"[ADDED] {row['stock1']}/{row['stock2']}: ${best['capital']:,.2f}")
        
        final_df = pd.DataFrame(final_allocations)
    
else:
    # Under-allocated or perfectly allocated - use snapped allocations
    print("\n" + "=" * 80)
    print("PASS 5: Allocation within budget - using snapped values")
    print("=" * 80)
    
    final_df = pd.DataFrame([{
        'stock1': row['stock1'],
        'stock2': row['stock2'],
        'signal': row['signal'],
        'z_score': row['z_score'],
        'hedge_ratio': row['hedge_ratio'],
        'win_prob': row['win_prob'],
        'win_loss_ratio': row['win_loss_ratio'],
        'raw_kelly_fraction': row['raw_kelly_fraction'],
        'target_capital': row['target_capital'],
        'shares1': row['snapped_shares1'],
        'shares2': row['snapped_shares2'],
        'capital_allocation': row['snapped_capital'],
        'kelly_fraction': row['snapped_fraction'],
        'price1': row['price1'],
        'price2': row['price2'],
        'hedge_error': row['snapped_hedge_error']
    } for idx, row in snapped_df.iterrows()])

# ============================================================================
# OUTPUT: Save results and display summary
# ============================================================================
if len(final_df) > 0:
    final_df = final_df.sort_values('z_score', key=abs, ascending=False)
    
    # Save sized signals
    output_cols = ['stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio', 
                   'kelly_fraction', 'capital_allocation', 'shares1', 'shares2',
                   'price1', 'price2', 'win_prob', 'win_loss_ratio']
    
    final_df[output_cols].to_csv("data/sized_signals.csv", index=False)
    
    # Display all positions
    print("\n" + "=" * 80)
    print(f"FINAL ALLOCATION - All {len(final_df)} Positions:")
    print("=" * 80)
    for idx, row in final_df.iterrows():
        val1 = row['shares1'] * row['price1']
        val2 = row['shares2'] * row['price2']
        actual_ratio = val1 / val2
        
        print(f"\n✓ {row['stock1']}/{row['stock2']}:")
        print(f"  Z-score: {row['z_score']:6.2f}, Signal: {row['signal']}")
        print(f"  Raw Kelly: {row['raw_kelly_fraction']:.2%}, Target: ${row['target_capital']:,.2f}, Final: ${row['capital_allocation']:,.2f} ({row['kelly_fraction']:.2%})")
        print(f"  Trade: {row['shares1']} {row['stock1']} @ ${row['price1']:.2f} = ${val1:,.2f}")
        print(f"        {row['shares2']} {row['stock2']} @ ${row['price2']:.2f} = ${val2:,.2f}")
        print(f"  Hedge Ratio: {actual_ratio:.3f} (target: {abs(row['hedge_ratio']):.3f}, error: {row['hedge_error']:.1f}%)")
    
    # Summary
    actual_allocated = final_df['capital_allocation'].sum()
    actual_fraction = actual_allocated / total_equity
    target_allocation = target_fraction * total_equity
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Pairs Executed: {len(final_df)}")
    print(f"Total Pairs Available: {len(enriched_df)}")
    
    skipped = len(enriched_df) - len(final_df)
    if skipped > 0:
        print(f"  Skipped due to budget: {skipped}")
    
    print(f"\nCapital Allocation:")
    print(f"  Target: ${target_allocation:,.2f} ({target_fraction:.1%} of equity)")
    print(f"  Actual: ${actual_allocated:,.2f} ({actual_fraction:.1%} of equity)")
    print(f"  Efficiency: {actual_allocated / target_allocation:.1%}")
    
    print(f"\nPortfolio Metrics:")
    print(f"  Total Kelly Fraction: {final_df['kelly_fraction'].sum():.2%}")
    print(f"  Average position size: {actual_fraction / len(final_df):.2%}")
    print(f"  Largest position: {final_df['kelly_fraction'].max():.2%}")
    print(f"  Smallest position: {final_df['kelly_fraction'].min():.2%}")
    
    print(f"\nHedge Accuracy:")
    print(f"  Average error: {final_df['hedge_error'].mean():.2f}%")
    print(f"  Max error: {final_df['hedge_error'].max():.2f}%")
    print(f"  Pairs with error > 3%: {(final_df['hedge_error'] > 3).sum()}")
    
    # Allocation precision
    target_vs_actual = final_df.apply(
        lambda row: abs(row['capital_allocation'] - row['target_capital']) / row['target_capital'] * 100,
        axis=1
    )
    print(f"\nAllocation Precision:")
    print(f"  Average deviation from target: {target_vs_actual.mean():.1f}%")
    print(f"  Max deviation from target: {target_vs_actual.max():.1f}%")
    
    print(f"\n✓ Sized signals saved to data/sized_signals.csv")
else:
    print("\n❌ No executable pairs found - saving empty CSV")
    pd.DataFrame(columns=['stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio', 
                          'kelly_fraction', 'capital_allocation', 'shares1', 'shares2',
                          'price1', 'price2', 'win_prob', 'win_loss_ratio']).to_csv("data/sized_signals.csv", index=False)