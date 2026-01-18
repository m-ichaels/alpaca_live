import subprocess
import sys
import os
from datetime import datetime, time, timedelta
import pytz
import pandas_market_calendars as mcal

# Ensure output is visible immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def should_run_pipeline():
    """
    Determine if the pipeline should run based on market schedule.
    Returns True if we should run now (1 hour before close), False otherwise.
    Now properly handles DST transitions.
    """
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    today = now_et.date()
    current_time = now_et.time()
    
    print("\n" + "="*70)
    print("MARKET SCHEDULE CHECK")
    print("="*70)
    
    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Check today's schedule
    schedule = nyse.schedule(start_date=today, end_date=today)
    
    if schedule.empty:
        print(f"Market is CLOSED today ({today})")
        print("="*70)
        return False
    
    # Get market close time for today (already in ET timezone)
    market_close_dt = schedule.iloc[0]['market_close'].tz_convert(et_tz)
    market_close = market_close_dt.time()
    
    # Determine if early close (before 3 PM) or regular close (4 PM)
    is_early_close = market_close < time(15, 0)
    
    if is_early_close:
        # Early close at 1 PM ET -> run at 12 PM ET
        target_run_time = time(12, 0)
        window_end = time(12, 30)
        print(f"Early close day - Market closes at {market_close.strftime('%I:%M %p')} ET")
    else:
        # Regular close at 4 PM ET -> run at 3 PM ET
        target_run_time = time(15, 0)
        window_end = time(15, 30)
        print(f"Regular trading day - Market closes at {market_close.strftime('%I:%M %p')} ET")
    
    print(f"Target run time: {target_run_time.strftime('%I:%M %p')} ET")
    print(f"Current time: {current_time.strftime('%I:%M:%S %p')} ET")
    
    # Allow a 30-minute window for execution
    if target_run_time <= current_time <= window_end:
        print("✓ Within execution window - PROCEEDING WITH PIPELINE")
        print("="*70)
        return True
    else:
        print("✗ Outside execution window - SKIPPING PIPELINE")
        print("="*70)
        return False

def run_script(script_name, description):
    """Run a Python script and stream output in real-time."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"Running: {script_name}")
    print('='*70)
    sys.stdout.flush()
    
    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, '-u', script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                sys.stdout.flush()
        
        process.wait()
        return process.returncode == 0
            
    except Exception as e:
        print(f"[X] Unexpected error running {script_name}: {e}")
        return False

def main():
    """Execute the complete trading pipeline."""
    print("\n" + "="*70)
    print("STATISTICAL ARBITRAGE PIPELINE")
    print(f"System Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if we should run based on market schedule
    try:
        if not should_run_pipeline():
            print("\nPipeline execution skipped - not the correct time window")
            return 0
    except Exception as e:
        print(f"\nError checking market schedule: {e}")
        print("Skipping pipeline execution for safety")
        return 1
    
    os.makedirs("data", exist_ok=True)
    
    pipeline = [
        ("s&p500_stocks.py", "Step 1: Fetching S&P 500 ticker list"),
        ("data_getting.py", "Step 2: Downloading historical price data"),
        ("cleaning.py", "Step 3: Cleaning and preprocessing data"),
        ("pair_getting.py", "Step 4: Finding cointegrated pairs"),
        ("edge.py", "Step 5: Calculating entry signals and OU parameters"),
        ("correlations.py", "Step 6: Calculating pair correlations"),
        ("diversification.py", "Step 7: Kelly criterion position sizing"),
        ("portfolio_kelly.py", "Step 8: Executing trades based on sized signals"),
        ("comparison.py", "Step 9: Comparing ideal vs current portfolio"),
        ("execute.py", "Step 10: Executing trades via Alpaca API")
    ]
    
    results = {}
    for script, description in pipeline:
        success = run_script(script, description)
        results[script] = success
        if not success:
            print(f"\nPIPELINE STOPPED: {script} failed")
            break
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    for script, success in results.items():
        status = "[OK] SUCCESS" if success else "[X] FAILED"
        print(f"{status:15} - {script}")
    
    all_success = all(results.values())
    print("\n" + "="*70)
    print("RESULT: " + ("SUCCESS" if all_success else "FAILED"))
    print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70 + "\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())