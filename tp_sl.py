import pandas as pd
import numpy as np
import os
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET

# Configuration
TAKE_PROFIT_Z = 0.5    # Close when spread reverts near mean
STOP_LOSS_Z = 4.0      # Close if divergence becomes extreme
TRACKER_FILE = "data/open_pairs.csv"

trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

def get_latest_price(symbol):
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data_client.get_stock_latest_quote(request)
    return float(quote[symbol].ask_price)

def close_pair_in_tracker(stock1, stock2):
    """Mark a pair as closed in the tracker"""
    try:
        if not os.path.exists(TRACKER_FILE):
            print(f"  ⚠️  No tracker file found")
            return False
        
        tracker = pd.read_csv(TRACKER_FILE)
        
        # Find the pair (check both orderings)
        mask = (
            ((tracker['stock1'] == stock1) & (tracker['stock2'] == stock2)) |
            ((tracker['stock1'] == stock2) & (tracker['stock2'] == stock1))
        ) & (tracker['status'] == 'open')
        
        if mask.any():
            tracker.loc[mask, 'status'] = 'closed'
            tracker.loc[mask, 'exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            tracker.to_csv(TRACKER_FILE, index=False)
            print(f"  ✓ Pair closed in tracker")
            return True
        else:
            print(f"  ⚠️  Pair not found in tracker")
            return False
            
    except Exception as e:
        print(f"  ⚠️  Warning: Could not close pair in tracker - {e}")
        return False

def manage_open_trades():
    # 1. Load tracked open pairs
    if not os.path.exists(TRACKER_FILE):
        print("No tracker file found. No open pairs to manage.")
        return
    
    tracker = pd.read_csv(TRACKER_FILE)
    open_pairs_df = tracker[tracker['status'] == 'open']
    
    if len(open_pairs_df) == 0:
        print("No open pairs to manage.")
        return
    
    # 2. Load historical data for z-score calculation
    try:
        prices_df = pd.read_csv("data/sp500_prices_clean.csv")
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
        
        pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
    except FileNotFoundError:
        print("Required data files not found. Skipping management.")
        return

    print(f"Checking {len(open_pairs_df)} tracked open pairs...")
    print("=" * 80)

    # 3. Iterate through tracked pairs
    for _, row in open_pairs_df.iterrows():
        s1, s2 = row['stock1'], row['stock2']
        
        # Get hedge ratio from cointegrated pairs
        pair_info = pairs_df[
            ((pairs_df['stock1'] == s1) & (pairs_df['stock2'] == s2)) |
            ((pairs_df['stock1'] == s2) & (pairs_df['stock2'] == s1))
        ]
        
        if len(pair_info) == 0:
            print(f"⚠️  {s1}/{s2} - No cointegration data found, skipping")
            continue
        
        beta = pair_info.iloc[0]['hedge_ratio']
        
        # Calculate CURRENT spread and Z-score
        try:
            p1 = get_latest_price(s1)
            p2 = get_latest_price(s2)
            current_spread = p1 - (beta * p2)
            
            # Use historical stats for consistency
            hist_spread = prices_df[s1] - (beta * prices_df[s2])
            mu, sigma = hist_spread.mean(), hist_spread.std()
            current_z = (current_spread - mu) / sigma
            
            print(f"\n{s1}/{s2}")
            print(f"  Entry Z: {row['z_score']:.2f}, Current Z: {current_z:.2f}")
            print(f"  Signal: {row['signal']}, Capital: ${row['capital_allocation']:,.2f}")
            
            # 4. Exit Logic
            should_close = False
            close_reason = ""
            
            # Take Profit: Spread has reverted toward the mean
            if abs(current_z) <= TAKE_PROFIT_Z:
                should_close = True
                close_reason = "TARGET REACHED"
                print(f"  >>> {close_reason} - Closing for profit")
            
            # Stop Loss: Spread has moved even further away
            elif abs(current_z) >= STOP_LOSS_Z:
                should_close = True
                close_reason = "STOP LOSS HIT"
                print(f"  >>> {close_reason} - Closing to limit risk")
            
            if should_close:
                try:
                    trading_client.close_position(s1)
                    trading_client.close_position(s2)
                    close_pair_in_tracker(s1, s2)
                    print(f"  ✓ Positions closed in Alpaca and tracker updated")
                except Exception as e:
                    print(f"  ✗ Error closing positions: {e}")
            else:
                print(f"  → Holding (within thresholds)")
                
        except Exception as e:
            print(f"\n{s1}/{s2}")
            print(f"  ✗ Error checking pair: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    manage_open_trades()