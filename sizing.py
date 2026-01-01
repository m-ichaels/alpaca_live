import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET
from scipy.stats import norm

# Connect to Alpaca
trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

# Load signals
signals_df = pd.read_csv("data/entry_signals.csv")

# Get account equity
account = trading_client.get_account()
total_equity = float(account.equity)

print(f"Total Account Equity: ${total_equity:,.2f}")
print(f"Processing {len(signals_df)} signals for Kelly sizing\n")

# Get current positions to exclude from allocation
positions = trading_client.get_all_positions()
current_holdings = {p.symbol for p in positions}

def estimate_win_probability(z_score):
    """
    More realistic win probability based on statistical properties
    Uses normal distribution CDF - probability that z-score will revert
    """
    abs_z = abs(z_score)
    
    # For z > threshold, probability it reverts (goes back toward mean)
    # This is more conservative than the original formula
    if abs_z < 1.5:
        return 0.55  # Weak signal
    elif abs_z < 2.0:
        return 0.65  # Moderate signal
    elif abs_z < 2.5:
        return 0.72  # Strong signal
    elif abs_z < 3.0:
        return 0.77  # Very strong signal
    else:
        return 0.80  # Extreme signal (cap at 80%)

def estimate_win_loss_ratio(z_score, volatility_percentile=50):
    """
    Expected profit/loss ratio based on mean reversion distance
    Higher z-scores suggest more reversion potential
    """
    abs_z = abs(z_score)
    
    # Base ratio increases with z-score (more room to revert)
    base_ratio = 1.0 + (abs_z - 1.5) * 0.4
    
    # Cap between reasonable bounds
    return max(1.2, min(2.5, base_ratio))

def check_trade_feasibility(stock1, stock2, beta, capital, signal):
    """Check if we can execute trades with given capital"""
    try:
        quotes = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[stock1, stock2])
        )
        
        price1 = (quotes[stock1].bid_price + quotes[stock1].ask_price) / 2
        price2 = (quotes[stock2].bid_price + quotes[stock2].ask_price) / 2
        
        # Skip if prices are invalid
        if price1 <= 0 or price2 <= 0:
            return False, None, None, None, None, None
        
        # Calculate shares with hedge ratio consideration
        dollars_stock2 = capital / (1 + beta)
        dollars_stock1 = capital - dollars_stock2
        
        shares1 = int(dollars_stock1 / price1)
        shares2 = int(dollars_stock2 / price2)
        
        # Need at least 1 share of each
        if shares1 < 1 or shares2 < 1:
            return False, None, None, None, None, None
        
        # Calculate actual capital used (integer shares)
        actual_capital = shares1 * price1 + shares2 * price2
        
        return True, shares1, shares2, price1, price2, actual_capital
    
    except Exception as e:
        return False, None, None, None, None, None

# Calculate raw Kelly fractions for each pair
raw_kelly_data = []

for idx, row in signals_df.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    
    # Skip if already have positions
    if stock1 in current_holdings or stock2 in current_holdings:
        continue
    
    z_score = row['z_score']
    abs_z = abs(z_score)
    
    # Improved Kelly parameters
    win_prob = estimate_win_probability(z_score)
    win_loss_ratio = estimate_win_loss_ratio(z_score)
    
    # Kelly formula: f = (p * b - q) / b
    # where p = win_prob, q = 1-p, b = win/loss ratio
    loss_prob = 1 - win_prob
    kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
    
    # Apply 1/3 Kelly for conservative sizing (less aggressive than 1/4)
    kelly_fraction = max(0, kelly_fraction) * 0.33
    
    # Cap at 8% per pair for diversification
    kelly_fraction = min(kelly_fraction, 0.08)
    
    raw_kelly_data.append({
        'stock1': row['stock1'],
        'stock2': row['stock2'],
        'signal': row['signal'],
        'z_score': row['z_score'],
        'hedge_ratio': row['hedge_ratio'],
        'win_prob': win_prob,
        'win_loss_ratio': win_loss_ratio,
        'raw_kelly_fraction': kelly_fraction,
        'priority': abs_z  # Use z-score as priority
    })

# Convert to DataFrame and sort by priority
kelly_df = pd.DataFrame(raw_kelly_data)
kelly_df = kelly_df.sort_values('priority', ascending=False).reset_index(drop=True)

print(f"Raw Kelly Total: {kelly_df['raw_kelly_fraction'].sum():.2%}\n")

# Pre-filter: Check all pairs for feasibility BEFORE allocation
print("Pre-filtering for feasibility...")
print("=" * 80)

feasible_pairs = []
infeasible_pairs = []

for idx, row in kelly_df.iterrows():
    # Test with a reasonable amount to see if trade is even possible
    test_capital = row['raw_kelly_fraction'] * total_equity
    
    feasible, s1, s2, p1, p2, actual_cap = check_trade_feasibility(
        row['stock1'], row['stock2'], row['hedge_ratio'], 
        test_capital, row['signal']
    )
    
    if feasible:
        feasible_pairs.append({
            **row.to_dict(),
            'min_capital': actual_cap,
            'price1': p1,
            'price2': p2
        })
    else:
        infeasible_pairs.append(row.to_dict())

print(f"Feasible pairs: {len(feasible_pairs)}/{len(kelly_df)}")
print(f"Infeasible pairs: {len(infeasible_pairs)}\n")

if not feasible_pairs:
    print("No feasible pairs found!")
    exit(1)

# Recalculate Kelly fractions based only on feasible pairs
feasible_df = pd.DataFrame(feasible_pairs)
total_feasible_kelly = feasible_df['raw_kelly_fraction'].sum()

print(f"Feasible Kelly Total: {total_feasible_kelly:.2%}")

# Target allocation (95% of equity)
target_total_allocation = 0.95 * total_equity

# Smart allocation: allocate proportionally with single-pass optimization
print("\nOptimal Allocation (Single Pass):")
print("=" * 80)

allocated_pairs = []
allocated_capital = 0

# If total Kelly < 95%, scale up proportionally
# If total Kelly > 95%, scale down proportionally
scaling_factor = min(target_total_allocation / (total_feasible_kelly * total_equity), 1.0)

print(f"Scaling factor: {scaling_factor:.4f}\n")

for idx, row in feasible_df.iterrows():
    # Calculate target allocation
    target_kelly = row['raw_kelly_fraction'] * scaling_factor
    target_capital = target_kelly * total_equity
    
    # Check feasibility at this allocation level
    feasible, s1, s2, p1, p2, actual_cap = check_trade_feasibility(
        row['stock1'], row['stock2'], row['hedge_ratio'], 
        target_capital, row['signal']
    )
    
    if feasible:
        allocated_pairs.append({
            **row,
            'allocated_capital': actual_cap,
            'shares1': s1,
            'shares2': s2,
            'price1': p1,
            'price2': p2,
            'kelly_fraction': actual_cap / total_equity
        })
        allocated_capital += actual_cap
        print(f"✓ {row['stock1']:6s}/{row['stock2']:6s}: ${actual_cap:>9,.2f} ({actual_cap/total_equity:>5.2%})")
    else:
        print(f"✗ {row['stock1']:6s}/{row['stock2']:6s}: Failed at ${target_capital:>9,.2f}")

# Second pass: Redistribute remaining capital if significant
remaining_capital = target_total_allocation - allocated_capital

if remaining_capital > 500 and allocated_pairs:
    print(f"\nRedistribution Pass - ${remaining_capital:,.2f} remaining:")
    print("=" * 80)
    
    # Sort by priority for redistribution
    allocated_pairs.sort(key=lambda x: x['priority'], reverse=True)
    
    for pair in allocated_pairs:
        if remaining_capital < 100:
            break
        
        # Check if we can add more to this pair (respect 8% cap)
        current_kelly = pair['kelly_fraction']
        max_additional_kelly = 0.08 - current_kelly
        
        if max_additional_kelly < 0.005:  # Less than 0.5% room
            continue
        
        # Try adding capital
        additional = min(remaining_capital * 0.1, max_additional_kelly * total_equity)
        new_total = pair['allocated_capital'] + additional
        
        feasible, s1, s2, p1, p2, actual_cap = check_trade_feasibility(
            pair['stock1'], pair['stock2'], pair['hedge_ratio'],
            new_total, pair['signal']
        )
        
        if feasible and actual_cap > pair['allocated_capital']:
            added = actual_cap - pair['allocated_capital']
            remaining_capital -= added
            allocated_capital += added
            
            pair['allocated_capital'] = actual_cap
            pair['shares1'] = s1
            pair['shares2'] = s2
            pair['kelly_fraction'] = actual_cap / total_equity
            
            print(f"  + {pair['stock1']}/{pair['stock2']}: +${added:,.2f} → ${actual_cap:,.2f}")

# Create final DataFrame
final_df = pd.DataFrame(allocated_pairs)

if len(final_df) > 0:
    final_df = final_df.sort_values('priority', ascending=False)
    
    # Display top positions
    print("\n" + "=" * 80)
    print("FINAL ALLOCATION - Top 20 Positions:")
    print("=" * 80)
    for idx, row in final_df.head(20).iterrows():
        print(f"{row['stock1']}/{row['stock2']}:")
        print(f"  Z-score: {row['z_score']:6.2f}, Signal: {row['signal']}")
        print(f"  Win Prob: {row['win_prob']:.1%}, Win/Loss: {row['win_loss_ratio']:.2f}x")
        print(f"  Kelly: {row['kelly_fraction']:.2%} (Raw: {row['raw_kelly_fraction']:.2%})")
        print(f"  Capital: ${row['allocated_capital']:,.2f}")
        print(f"  Trade: {row['shares1']} {row['stock1']} @ ${row['price1']:.2f}, "
              f"{row['shares2']} {row['stock2']} @ ${row['price2']:.2f}\n")
    
    # Save sized signals
    output_cols = ['stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio', 
                   'kelly_fraction', 'capital_allocation', 'shares1', 'shares2',
                   'price1', 'price2', 'win_prob', 'win_loss_ratio']
    
    final_df['capital_allocation'] = final_df['allocated_capital']
    final_df[output_cols].to_csv("data/sized_signals.csv", index=False)
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Pairs: {len(final_df)}")
    print(f"Infeasible Pairs: {len(infeasible_pairs)}")
    print(f"Total Capital Allocated: ${final_df['allocated_capital'].sum():,.2f}")
    print(f"Target Allocation: ${target_total_allocation:,.2f}")
    print(f"Allocation Efficiency: {final_df['allocated_capital'].sum() / target_total_allocation:.2%}")
    print(f"Portfolio Utilization: {final_df['allocated_capital'].sum() / total_equity:.2%}")
    print(f"\nPosition Size Statistics:")
    print(f"  Average: ${final_df['allocated_capital'].mean():,.2f}")
    print(f"  Median:  ${final_df['allocated_capital'].median():,.2f}")
    print(f"  Largest: ${final_df['allocated_capital'].max():,.2f}")
    print(f"  Smallest: ${final_df['allocated_capital'].min():,.2f}")
    print(f"\nKelly Statistics:")
    print(f"  Max Kelly Fraction: {final_df['kelly_fraction'].max():.2%}")
    print(f"  Total Kelly Fraction: {final_df['kelly_fraction'].sum():.2%}")
    print(f"  Avg Win Probability: {final_df['win_prob'].mean():.1%}")
    print(f"  Avg Win/Loss Ratio: {final_df['win_loss_ratio'].mean():.2f}x")
    print(f"\nSized signals saved to data/sized_signals.csv")
else:
    print("\nNo feasible pairs found!")