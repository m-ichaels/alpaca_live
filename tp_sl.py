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
HISTORY_FILE = "data/trade_history.csv"

trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

def get_latest_price(symbol):
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data_client.get_stock_latest_quote(request)
    return float(quote[symbol].ask_price)

def remove_pair_from_tracker(stock1, stock2):
    """REMOVE a pair from the tracker (not just mark as closed)"""
    try:
        if not os.path.exists(TRACKER_FILE):
            print(f"  [!] No tracker file found")
            return False
        
        tracker = pd.read_csv(TRACKER_FILE)
        
        # Find the pair (check both orderings)
        mask = (
            ((tracker['stock1'] == stock1) & (tracker['stock2'] == stock2)) |
            ((tracker['stock1'] == stock2) & (tracker['stock2'] == stock1))
        ) & (tracker['status'] == 'open')
        
        if mask.any():
            # REMOVE the row instead of marking as closed
            tracker = tracker[~mask]
            tracker.to_csv(TRACKER_FILE, index=False)
            print(f"  [OK] Pair removed from tracker")
            return True
        else:
            print(f"  [!] Pair not found in tracker")
            return False
            
    except Exception as e:
        print(f"  [!] Warning: Could not remove pair from tracker - {e}")
        return False

def log_trade_outcome(stock1, stock2, exit_z, exit_reason):
    """Log trade outcome to history file for win rate calculation"""
    try:
        if not os.path.exists(HISTORY_FILE):
            print(f"  [!] No trade history file found")
            return
        
        history = pd.read_csv(HISTORY_FILE)
        
        # Find the open trade (check both orderings)
        mask = (
            ((history['stock1'] == stock1) & (history['stock2'] == stock2)) |
            ((history['stock1'] == stock2) & (history['stock2'] == stock1))
        ) & (history['exit_date'].isna())
        
        if mask.any():
            history.loc[mask, 'exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            history.loc[mask, 'exit_z'] = exit_z
            history.loc[mask, 'exit_reason'] = exit_reason
            
            # Determine win/loss: TARGET REACHED = win, STOP LOSS HIT = loss
            history.loc[mask, 'win'] = 1 if exit_reason == "TARGET REACHED" else 0
            
            history.to_csv(HISTORY_FILE, index=False)
            print(f"  [OK] Trade outcome logged to history")
            
            # Display updated win rate
            completed = history[history['exit_date'].notna()]
            definitive = completed[completed['win'].notna()]
            if len(definitive) >= 5:
                win_rate = definitive['win'].mean()
                print(f"  [STATS] Overall win rate: {win_rate:.1%} ({definitive['win'].sum():.0f}/{len(definitive)} wins)")
        else:
            print(f"  [!] Trade not found in history")
            
    except Exception as e:
        print(f"  [!] Warning: Could not log trade outcome - {e}")

def calculate_exit_z_score(stock1, stock2, beta):
    """
    Calculate current z-score using THE EXACT SAME beta from entry.
    This ensures consistency between entry and exit calculations.
    """
    try:
        # Load historical data
        prices_df = pd.read_csv("data/sp500_prices_clean.csv")
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
        
        # Get current prices
        p1 = get_latest_price(stock1)
        p2 = get_latest_price(stock2)
        
        if p1 is None or p2 is None:
            return None
        
        # Calculate current spread using THE EXACT SAME beta from entry
        current_spread = p1 - (beta * p2)
        
        # Get historical spread statistics using THE EXACT SAME beta
        hist_spread = prices_df[stock1] - (beta * prices_df[stock2])
        mu, sigma = hist_spread.mean(), hist_spread.std()
        
        # Calculate z-score
        current_z = (current_spread - mu) / sigma
        return current_z
        
    except Exception as e:
        print(f"    Warning: Could not calculate z-score - {e}")
        return None

def manage_open_trades():
    # 1. Load tracked open pairs
    if not os.path.exists(TRACKER_FILE):
        print("No tracker file found. No open pairs to manage.")
        return
    
    tracker = pd.read_csv(TRACKER_FILE)
    
    # Check if hedge_ratio column exists
    if 'hedge_ratio' not in tracker.columns:
        print("ERROR: tracker missing hedge_ratio column!")
        print("Please delete data/open_pairs.csv and recreate it by running the pipeline.")
        return
    
    open_pairs_df = tracker[tracker['status'] == 'open'].copy()
    
    if len(open_pairs_df) == 0:
        print("No open pairs to manage.")
        return
    
    # 2. Load historical data for z-score calculation
    try:
        prices_df = pd.read_csv("data/sp500_prices_clean.csv")
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
    except FileNotFoundError:
        print("Required data files not found. Skipping management.")
        return

    print(f"Checking {len(open_pairs_df)} tracked open pairs...")
    print("=" * 80)

    # 3. Iterate through tracked pairs
    for _, row in open_pairs_df.iterrows():
        s1, s2 = row['stock1'], row['stock2']
        
        # USE THE STORED HEDGE RATIO FROM TRACKER (exact same as entry!)
        beta = row['hedge_ratio']
        
        # Calculate CURRENT spread and Z-score
        try:
            current_z = calculate_exit_z_score(s1, s2, beta)
            
            if current_z is None:
                print(f"\n{s1}/{s2}")
                print(f"  [!] Could not get current prices, skipping")
                continue
            
            print(f"\n{s1}/{s2}")
            print(f"  Entry Z: {row['z_score']:.2f}, Current Z: {current_z:.2f}")
            print(f"  Signal: {row['signal']}, Capital: ${row['capital_allocation']:,.2f}")
            print(f"  Beta: {beta:.4f}")
            
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
                    remove_pair_from_tracker(s1, s2)  # REMOVE instead of mark closed
                    log_trade_outcome(s1, s2, current_z, close_reason)
                    print(f"  [OK] Positions closed in Alpaca and removed from tracker")
                except Exception as e:
                    print(f"  [X] Error closing positions: {e}")
            else:
                print(f"  -> Holding (within thresholds)")
                
        except Exception as e:
            print(f"\n{s1}/{s2}")
            print(f"  [X] Error checking pair: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    manage_open_trades()