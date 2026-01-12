import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest
    ALPACA_OPTIONS_AVAILABLE = True
except ImportError:
    ALPACA_OPTIONS_AVAILABLE = False
    print("WARNING: Alpaca options data client not available")
    print("Install with: pip install alpaca-py[options]")

try:
    from auth_local import KEY, SECRET  # For local testing
except ImportError:
    print("Import error")
    from auth import KEY, SECRET  # For GitHub Actions

# ============================================================================
# CONFIGURATION
# ============================================================================

# Liquidity thresholds
MIN_OPEN_INTEREST = 100        # Minimum open interest per option
MIN_VOLUME = 20                # Minimum daily volume per option
MAX_BID_ASK_SPREAD_PCT = 5.0   # Max bid-ask spread as % of mid price
MIN_STRIKE_DENSITY = 3         # Min number of strikes within 5% of current price

# Expiration preferences
MIN_DTE = 25                   # Minimum days to expiration
MAX_DTE = 65                   # Maximum days to expiration
TARGET_DTE = 45                # Target days to expiration

# Strike selection
ATM_TOLERANCE_PCT = 2.0        # How far from ATM to look for strikes (%)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_price(symbol, data_client):
    """Get current stock price"""
    try:
        quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(quote_request)
        
        if symbol in quote:
            bid = quote[symbol].bid_price
            ask = quote[symbol].ask_price
            if bid and ask and bid > 0 and ask > 0:
                return (float(bid) + float(ask)) / 2
        return None
    except Exception as e:
        return None

def check_options_liquidity_mock(symbol, current_price, target_dte=TARGET_DTE):
    """
    Mock function that estimates options feasibility based on stock characteristics.
    
    This is a fallback when Alpaca options API is not available.
    Real implementation would query actual options chains.
    """
    
    # Heuristic: High price, high volume stocks usually have liquid options
    # This is a ROUGH approximation - real data is always better
    
    result = {
        'symbol': symbol,
        'current_price': current_price,
        'feasible': False,
        'reason': 'Unknown',
        'call_bid': None,
        'call_ask': None,
        'put_bid': None,
        'put_ask': None,
        'strike': None,
        'expiration': None,
        'dte': None,
        'call_oi': None,
        'put_oi': None,
        'call_volume': None,
        'put_volume': None,
        'estimated': True  # Flag that this is estimated, not real data
    }
    
    # Estimate based on price (stocks with higher prices tend to have more liquid options)
    # This is NOT accurate but gives a rough filter
    if current_price < 20:
        result['reason'] = 'Price too low (< $20) - typically illiquid options'
        return result
    
    # For common S&P 500 stocks, we can make educated guesses
    # Major stocks that typically have very liquid options
    liquid_tickers = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
        'BRK.B', 'V', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV',
        'KO', 'PEP', 'AVGO', 'COST', 'MCD', 'CSCO', 'ACN', 'ADBE', 'NKE',
        'TMO', 'ABT', 'DIS', 'CRM', 'ORCL', 'INTC', 'AMD', 'NFLX', 'CMCSA',
        'XOM', 'BA', 'GE', 'CAT', 'UNH', 'DHR', 'VZ', 'TXN', 'PM', 'UPS',
        'HON', 'IBM', 'QCOM', 'AMGN', 'NEE', 'RTX', 'SBUX', 'LOW', 'SPGI',
        'GS', 'MS', 'BLK', 'AXP', 'C', 'USB', 'PNC', 'TFC', 'BAC', 'WFC'
    }
    
    if symbol in liquid_tickers and current_price >= 30:
        result['feasible'] = True
        result['reason'] = 'High-liquidity ticker (estimated)'
        result['strike'] = round(current_price * 2) / 2  # Round to nearest $0.50
        result['expiration'] = (datetime.now() + timedelta(days=target_dte)).strftime('%Y-%m-%d')
        result['dte'] = target_dte
        
        # Estimate spreads (very rough)
        estimated_spread_pct = 2.0 if symbol in list(liquid_tickers)[:10] else 3.5
        result['call_bid'] = current_price * 0.03  # Rough estimate
        result['call_ask'] = result['call_bid'] * (1 + estimated_spread_pct / 100)
        result['put_bid'] = current_price * 0.03
        result['put_ask'] = result['put_bid'] * (1 + estimated_spread_pct / 100)
        
        # Estimate open interest
        result['call_oi'] = 500 if symbol in list(liquid_tickers)[:20] else 200
        result['put_oi'] = 500 if symbol in list(liquid_tickers)[:20] else 200
        result['call_volume'] = 100 if symbol in list(liquid_tickers)[:20] else 40
        result['put_volume'] = 100 if symbol in list(liquid_tickers)[:20] else 40
        
        return result
    
    elif symbol in liquid_tickers:
        result['reason'] = 'Known ticker but price < $30 - lower liquidity expected'
        return result
    
    else:
        result['reason'] = 'Not in high-liquidity list - would need real options data'
        return result

def check_options_liquidity_real(symbol, current_price, option_client, target_dte=TARGET_DTE):
    """
    Real implementation using Alpaca Options API.
    
    Checks actual options chain data for liquidity.
    """
    
    result = {
        'symbol': symbol,
        'current_price': current_price,
        'feasible': False,
        'reason': 'Unknown',
        'call_bid': None,
        'call_ask': None,
        'put_bid': None,
        'put_ask': None,
        'strike': None,
        'expiration': None,
        'dte': None,
        'call_oi': None,
        'put_oi': None,
        'call_volume': None,
        'put_volume': None,
        'estimated': False
    }
    
    try:
        # Request options chain
        chain_request = OptionChainRequest(
            underlying_symbol=symbol,
            expiration_date_gte=(datetime.now() + timedelta(days=MIN_DTE)).strftime('%Y-%m-%d'),
            expiration_date_lte=(datetime.now() + timedelta(days=MAX_DTE)).strftime('%Y-%m-%d')
        )
        
        chain = option_client.get_option_chain(chain_request)
        
        if not chain or len(chain) == 0:
            result['reason'] = 'No options chain data available'
            return result
        
        # Find ATM strike (closest to current price)
        chain_df = pd.DataFrame([
            {
                'strike': opt.strike_price,
                'expiration': opt.expiration_date,
                'type': opt.option_type,
                'bid': opt.bid_price,
                'ask': opt.ask_price,
                'volume': opt.volume,
                'open_interest': opt.open_interest
            }
            for opt in chain
        ])
        
        # Calculate DTE for each expiration
        chain_df['dte'] = chain_df['expiration'].apply(
            lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days
        )
        
        # Find expiration closest to target DTE
        best_expiration = chain_df.iloc[(chain_df['dte'] - target_dte).abs().argsort()[:1]]['expiration'].values[0]
        best_dte = chain_df[chain_df['expiration'] == best_expiration]['dte'].values[0]
        
        # Filter to that expiration
        exp_chain = chain_df[chain_df['expiration'] == best_expiration].copy()
        
        # Find ATM strike
        exp_chain['distance_from_atm'] = abs(exp_chain['strike'] - current_price)
        atm_strike = exp_chain.iloc[exp_chain['distance_from_atm'].argsort()[:1]]['strike'].values[0]
        
        # Get call and put at ATM strike
        call = exp_chain[(exp_chain['strike'] == atm_strike) & (exp_chain['type'] == 'call')]
        put = exp_chain[(exp_chain['strike'] == atm_strike) & (exp_chain['type'] == 'put')]
        
        if len(call) == 0 or len(put) == 0:
            result['reason'] = 'Missing call or put at ATM strike'
            return result
        
        call = call.iloc[0]
        put = put.iloc[0]
        
        # Check liquidity criteria
        call_spread_pct = (call['ask'] - call['bid']) / ((call['ask'] + call['bid']) / 2) * 100
        put_spread_pct = (put['ask'] - put['bid']) / ((put['ask'] + put['bid']) / 2) * 100
        
        failures = []
        
        if call['open_interest'] < MIN_OPEN_INTEREST:
            failures.append(f"Call OI too low ({call['open_interest']})")
        
        if put['open_interest'] < MIN_OPEN_INTEREST:
            failures.append(f"Put OI too low ({put['open_interest']})")
        
        if call['volume'] < MIN_VOLUME:
            failures.append(f"Call volume too low ({call['volume']})")
        
        if put['volume'] < MIN_VOLUME:
            failures.append(f"Put volume too low ({put['volume']})")
        
        if call_spread_pct > MAX_BID_ASK_SPREAD_PCT:
            failures.append(f"Call spread too wide ({call_spread_pct:.1f}%)")
        
        if put_spread_pct > MAX_BID_ASK_SPREAD_PCT:
            failures.append(f"Put spread too wide ({put_spread_pct:.1f}%)")
        
        if len(failures) > 0:
            result['reason'] = '; '.join(failures)
            return result
        
        # All checks passed!
        result['feasible'] = True
        result['reason'] = 'Passes all liquidity checks'
        result['strike'] = atm_strike
        result['expiration'] = best_expiration
        result['dte'] = best_dte
        result['call_bid'] = call['bid']
        result['call_ask'] = call['ask']
        result['put_bid'] = put['bid']
        result['put_ask'] = put['ask']
        result['call_oi'] = call['open_interest']
        result['put_oi'] = put['open_interest']
        result['call_volume'] = call['volume']
        result['put_volume'] = put['volume']
        
        return result
        
    except Exception as e:
        result['reason'] = f'Error checking options: {str(e)}'
        return result

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("OPTIONS FEASIBILITY CHECKER")
    print("=" * 80)
    print(f"Checking which pairs have liquid options for synthetic positions")
    print(f"\nCriteria:")
    print(f"  Min Open Interest: {MIN_OPEN_INTEREST}")
    print(f"  Min Volume: {MIN_VOLUME}")
    print(f"  Max Bid-Ask Spread: {MAX_BID_ASK_SPREAD_PCT}%")
    print(f"  Target DTE: {TARGET_DTE} ({MIN_DTE}-{MAX_DTE} range)")
    print("=" * 80 + "\n")
    
    # Initialize clients
    trading_client = TradingClient(KEY, SECRET, paper=True)
    data_client = StockHistoricalDataClient(KEY, SECRET)
    
    # Try to initialize options client
    option_client = None
    use_real_data = False
    
    if ALPACA_OPTIONS_AVAILABLE:
        try:
            option_client = OptionHistoricalDataClient(KEY, SECRET)
            use_real_data = True
            print("✓ Using real options data from Alpaca API\n")
        except Exception as e:
            print("⚠️ Could not initialize options client: {e}")
            print("Falling back to heuristic estimation\n")
    else:
        print("⚠️ Alpaca options library not available")
        print("Using heuristic estimation (less accurate)\n")
    
    # Load entry signals
    try:
        signals_df = pd.read_csv("data/entry_signals.csv")
    except FileNotFoundError:
        print("ERROR: entry_signals.csv not found. Run get_entry_criteria.py first!")
        return
    
    print(f"Checking options liquidity for {len(signals_df)} pairs...\n")
    
    # Get unique stocks
    stocks = set(signals_df['stock1'].tolist() + signals_df['stock2'].tolist())
    print(f"Unique stocks to check: {len(stocks)}\n")
    
    # Check each stock
    results = []
    
    for idx, symbol in enumerate(sorted(stocks), 1):
        print(f"[{idx}/{len(stocks)}] Checking {symbol}...", end=" ")
        
        # Get current price
        current_price = get_current_price(symbol, data_client)
        
        if current_price is None:
            print("âŒ No price data")
            results.append({
                'symbol': symbol,
                'current_price': None,
                'feasible': False,
                'reason': 'No price data available',
                'estimated': True
            })
            continue
        
        # Check options liquidity
        if use_real_data and option_client:
            result = check_options_liquidity_real(symbol, current_price, option_client)
        else:
            result = check_options_liquidity_mock(symbol, current_price)
        
        results.append(result)
        
        if result['feasible']:
            print(f"✓ ${current_price:.2f} - Liquid options available")
        else:
            print(f"✗ ${current_price:.2f} - {result['reason']}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Analyze pairs
    print("\n" + "=" * 80)
    print("PAIR FEASIBILITY ANALYSIS")
    print("=" * 80 + "\n")
    
    pair_results = []
    
    for idx, row in signals_df.iterrows():
        stock1 = row['stock1']
        stock2 = row['stock2']
        
        s1_result = results_df[results_df['symbol'] == stock1].iloc[0]
        s2_result = results_df[results_df['symbol'] == stock2].iloc[0]
        
        both_feasible = s1_result['feasible'] and s2_result['feasible']
        
        pair_results.append({
            'stock1': stock1,
            'stock2': stock2,
            'signal': row['signal'],
            'z_score': row['z_score'],
            'hedge_ratio': row['hedge_ratio'],
            'stock1_feasible': s1_result['feasible'],
            'stock2_feasible': s2_result['feasible'],
            'pair_feasible': both_feasible,
            'stock1_reason': s1_result['reason'],
            'stock2_reason': s2_result['reason'],
            'stock1_strike': s1_result.get('strike'),
            'stock2_strike': s2_result.get('strike'),
            'stock1_expiration': s1_result.get('expiration'),
            'stock2_expiration': s2_result.get('expiration'),
            'estimated': s1_result.get('estimated', True) or s2_result.get('estimated', True)
        })
    
    pairs_df = pd.DataFrame(pair_results)
    
    # Summary statistics
    feasible_pairs = pairs_df[pairs_df['pair_feasible'] == True]
    partial_pairs = pairs_df[(pairs_df['stock1_feasible'] == True) | (pairs_df['stock2_feasible'] == True)]
    
    print(f"Total pairs: {len(pairs_df)}")
    print(f"Both stocks feasible: {len(feasible_pairs)} ({len(feasible_pairs)/len(pairs_df)*100:.1f}%)")
    print(f"At least one stock feasible: {len(partial_pairs)} ({len(partial_pairs)/len(pairs_df)*100:.1f}%)")
    print(f"Neither stock feasible: {len(pairs_df) - len(partial_pairs)}")
    
    # Show feasible pairs
    if len(feasible_pairs) > 0:
        print(f"\n✓ FEASIBLE PAIRS ({len(feasible_pairs)}):")
        print("-" * 80)
        for idx, row in feasible_pairs.iterrows():
            print(f"{row['stock1']}/{row['stock2']}: Z={row['z_score']:.2f}, Signal={row['signal']}")
            if not row['estimated']:
                s1 = results_df[results_df['symbol'] == row['stock1']].iloc[0]
                s2 = results_df[results_df['symbol'] == row['stock2']].iloc[0]
                print(f"  {row['stock1']}: Strike ${s1['strike']:.2f}, DTE {s1['dte']}, OI {s1['call_oi']}/{s1['put_oi']}")
                print(f"  {row['stock2']}: Strike ${s2['strike']:.2f}, DTE {s2['dte']}, OI {s2['call_oi']}/{s2['put_oi']}")
    
    # Show problematic pairs
    problematic = pairs_df[pairs_df['pair_feasible'] == False]
    if len(problematic) > 0:
        print(f"\n✗ PROBLEMATIC PAIRS ({len(problematic)}):")
        print("-" * 80)
        for idx, row in problematic.head(10).iterrows():
            print(f"{row['stock1']}/{row['stock2']}:")
            if not row['stock1_feasible']:
                print(f"  {row['stock1']}: {row['stock1_reason']}")
            if not row['stock2_feasible']:
                print(f"  {row['stock2']}: {row['stock2_reason']}")
    
    # Save results
    results_df.to_csv("data/options_liquidity_stocks.csv", index=False)
    pairs_df.to_csv("data/options_liquidity_pairs.csv", index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Stock results saved to: data/options_liquidity_stocks.csv")
    print(f"Pair results saved to: data/options_liquidity_pairs.csv")
    
    if not use_real_data:
        print("\n⚠️ IMPORTANT: Results are ESTIMATED, not based on real options data!")
        print("For production use, you should:")
        print("  1. Install: pip install alpaca-py[options]")
        print("  2. Verify your Alpaca account has options data access")
        print("  3. Re-run this script to get real liquidity data")
    
    print("\nNext steps:")
    if len(feasible_pairs) > 0:
        print(f"  ✓ {len(feasible_pairs)} pairs ready for synthetic trading")
        print("  Run synthetic_sizing.py to calculate position sizes")
    else:
        print("  ✗ No feasible pairs found")
        print("  Consider:")
        print("    - Lowering liquidity thresholds")
        print("    - Focusing on more liquid stocks")
        print("    - Using stock positions for illiquid names")
    
    print("=" * 80)

if __name__ == "__main__":
    main()