import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, time
import pytz
import pandas_market_calendars as mcal
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
try:
    from auth_local import KEY, SECRET  # For local testing
except ImportError:
    print("Import error")
    from auth import KEY, SECRET  # For GitHub Actions

# Configuration
TAKE_PROFIT_Z = 0.5    # Close when spread reverts near mean
STOP_LOSS_Z = 3.0      # Close if divergence becomes extreme
TRACKER_FILE = "data/open_pairs.csv"
HISTORY_FILE = "data/trade_history.csv"

trading_client = TradingClient(KEY, SECRET, paper=True)
data_client = StockHistoricalDataClient(KEY, SECRET)

def should_run_tp_sl():
    """
    Determine if TP/SL should run based on market schedule.
    Returns True if market is currently open, False otherwise.
    """
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    today = now_et.date()
    current_time = now_et.time()
    
    print("\n" + "="*70)
    print("TP/SL MARKET CHECK")
    print("="*70)
    print(f"Current time: {now_et.strftime('%Y-%m-%d %I:%M:%S %p')} ET")
    
    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Check if market is open today
    schedule = nyse.schedule(start_date=today, end_date=today)
    
    if schedule.empty:
        print(f"Market is CLOSED today ({today})")
        print("="*70)
        return False
    
    # Get market open and close times for today
    market_open = schedule.iloc[0]['market_open'].tz_convert(et_tz).time()
    market_close = schedule.iloc[0]['market_close'].tz_convert(et_tz).time()
    
    print(f"Market hours: {market_open.strftime('%I:%M %p')} - {market_close.strftime('%I:%M %p')} ET")
    
    # Check if current time is within market hours
    if market_open <= current_time <= market_close:
        print("✓ Market is OPEN - Running TP/SL monitoring")
        print("="*70)
        return True
    else:
        print("✗ Market is CLOSED - Skipping TP/SL monitoring")
        print("="*70)
        return False

def get_position_side(symbol):
    """
    Check if we have an open position and return 'LONG' or 'SHORT'
    Returns None if no position exists
    """
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            if pos.symbol == symbol:
                qty = float(pos.qty)
                return 'LONG' if qty > 0 else 'SHORT'
        return None
    except:
        return None

def get_exit_price(symbol):
    """
    Get the correct exit price for a position:
    - If LONG position: use BID (we're selling)
    - If SHORT position: use ASK (we're buying to cover)
    """
    try:
        position_side = get_position_side(symbol)
        
        if position_side is None:
            print(f"      Warning: No position found for {symbol}")
            return None
        
        quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(quote_request)
        
        if symbol not in quote:
            return None
        
        if position_side == 'LONG':
            # Exiting a long = selling = use BID price
            price = quote[symbol].bid_price
            price_type = "bid"
        else:  # SHORT
            # Exiting a short = buying to cover = use ASK price
            price = quote[symbol].ask_price
            price_type = "ask"
        
        if price and float(price) > 0:
            return float(price)
        
        return None
        
    except Exception as e:
        print(f"      Warning: Could not get exit price for {symbol}: {e}")
        return None

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

def calculate_current_z_score(stock1, stock2, beta, entry_z):
    """
    Calculate current z-score using THE EXACT SAME statistics as entry.
    
    Uses the correct bid/ask prices based on position direction:
    - LONG position: use BID (we're selling)
    - SHORT position: use ASK (we're buying to cover)
    """
    try:
        # Load historical data (same as entry calculation)
        prices_df = pd.read_csv("data/sp500_prices_clean.csv")
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.set_index('date')
        
        # Calculate historical spread using the SAME beta
        historical_spread = prices_df[stock1] - (beta * prices_df[stock2])
        
        # Get statistics from HISTORICAL data only (frozen at time of entry)
        mu = historical_spread.mean()
        sigma = historical_spread.std()
        
        # Get CURRENT exit prices (bid for longs, ask for shorts)
        current_p1 = get_exit_price(stock1)
        current_p2 = get_exit_price(stock2)
        
        if current_p1 is None or current_p2 is None:
            return None
        
        if current_p1 <= 0 or current_p2 <= 0:
            return None
        
        # Calculate CURRENT spread using same beta
        current_spread = current_p1 - (beta * current_p2)
        
        # Calculate z-score: how many standard deviations is current spread from historical mean?
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
        entry_z = row['z_score']
        
        # Calculate CURRENT z-score using SAME statistics as entry
        try:
            current_z = calculate_current_z_score(s1, s2, beta, entry_z)
            
            if current_z is None:
                print(f"\n{s1}/{s2}")
                print(f"  [!] Could not get current prices, skipping")
                continue
            
            print(f"\n{s1}/{s2}")
            print(f"  Entry Z: {entry_z:.2f}, Current Z: {current_z:.2f}, Change: {current_z - entry_z:.2f}")
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

def main():
    """Main function with market hours check"""
    print("\n" + "="*70)
    print("TAKE PROFIT / STOP LOSS MONITOR")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if we should run based on market hours
    try:
        if not should_run_tp_sl():
            print("\nTP/SL monitoring skipped - market is closed")
            return 0
    except Exception as e:
        print(f"\nError checking market hours: {e}")
        print("Skipping TP/SL monitoring for safety")
        return 1
    
    # Run the monitoring
    try:
        manage_open_trades()
        print(f"\nFinished at: {datetime.now().strftime('%H:%M:%S')}")
        return 0
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())