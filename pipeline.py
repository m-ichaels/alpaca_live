import subprocess
import sys
import os
import time
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pytz

# Ensure output is visible in GitHub Action logs immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def check_market_and_wait():
    """
    Checks if NYSE is open. 
    If it's a trading day, waits until 1 hour before close.
    """
    nyse = mcal.get_calendar('NYSE')
    now_utc = datetime.now(pytz.utc)
    
    # Get schedule for today
    schedule = nyse.schedule(start_date=now_utc, end_date=now_utc)
    
    if schedule.empty:
        print(f"[{now_utc.strftime('%Y-%m-%d %H:%M:%S')}] NYSE is CLOSED today. Skipping.")
        return False

    # Get market close time for today (handles half-days automatically)
    market_close = schedule.iloc[0]['market_close'].to_pydatetime()
    target_time = market_close - timedelta(hours=1)
    
    wait_seconds = (target_time - now_utc).total_seconds()

    if wait_seconds > 0:
        print(f"Target: 1hr before close ({target_time.strftime('%H:%M:%S')} UTC)")
        print(f"Current: {now_utc.strftime('%H:%M:%S')} UTC")
        print(f"Waiting {round(wait_seconds / 60, 2)} minutes until execution...")
        time.sleep(wait_seconds)
    elif wait_seconds < -1800: # If we are more than 30 mins late
        print(f"Execution window missed. Target was {target_time.strftime('%H:%M:%S')} UTC")
        return False
    
    print(f"Ready to execute. Current time: {datetime.now(pytz.utc).strftime('%H:%M:%S')} UTC")
    return True

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def run_script(script_name, description):
    """Run a Python script and stream output in real-time."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"Running: {script_name}")
    print('='*70)
    sys.stdout.flush()
    
    try:
<<<<<<< HEAD
=======
        # Set environment to force unbuffered output
>>>>>>> 9e9798d47b88df72db715ebfdd10d4e6f4298a1c
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, '-u', script_name],
            stdout=subprocess.PIPE,
<<<<<<< HEAD
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
=======
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env  # Pass the modified environment
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                sys.stdout.flush()  # Force flush after each line
        
        process.wait()
        return_code = process.returncode
        
        if return_code == 0:
            print(f"[OK] {script_name} completed successfully")
            sys.stdout.flush()
            return True
        else:
            print(f"[X] {script_name} failed with exit code {return_code}")
            sys.stdout.flush()
            return False
            
    except Exception as e:
        print(f"[X] Unexpected error: {e}")
        sys.stdout.flush()
>>>>>>> 9e9798d47b88df72db715ebfdd10d4e6f4298a1c
        return False

def main():
    """Execute the complete trading pipeline."""
    print("\n" + "="*70)
    print("STATISTICAL ARBITRAGE PIPELINE")
    print(f"System Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if we should trade today and wait for the 1-hour-before-close mark
    if not check_market_and_wait():
        return 0

    os.makedirs("data", exist_ok=True)
    
    pipeline = [
        ("s&p500_stocks.py", "Step 1: Fetching S&P 500 ticker list"),
        ("data_getting.py", "Step 2: Downloading historical price data"),
        ("cleaning.py", "Step 3: Cleaning and preprocessing data"),
        ("pair_getting.py", "Step 4: Finding cointegrated pairs"),
<<<<<<< HEAD
        ("edge.py", "Step 5: Calculating entry signals and OU parameters"),
        ("correlations.py", "Step 6: Calculating pair correlations"),
        ("diversification.py", "Step 7: Kelly criterion position sizing"),
        ("portfolio_kelly.py", "Step 8: Executing trades based on sized signals"),
        ("comparison.py", "Step 9: Comparing ideal vs current portfolio"),
        ("execute.py", "Step 10: Executing trades via Alpaca API")
=======
        ("get_entry_criteria.py", "Step 5: Calculating entry signals and OU parameters"),
        ("correlations.py", "Step 6: Calculating pair correlations during trade periods"),
        ("sizing.py", "Step 7: Kelly criterion position sizing"),
        ("execute.py", "Step 8: Executing trades based on sized signals")
>>>>>>> 9e9798d47b88df72db715ebfdd10d4e6f4298a1c
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