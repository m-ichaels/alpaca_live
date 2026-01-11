import subprocess
import sys
import os
from datetime import datetime

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
        # Set environment to force unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, '-u', script_name],
            stdout=subprocess.PIPE,
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
        return False

def main():
    """Execute the complete trading pipeline."""
    print("\n" + "="*70)
    print("STATISTICAL ARBITRAGE PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Pipeline steps in order
    pipeline = [
        ("s&p500_stocks.py", "Step 1: Fetching S&P 500 ticker list"),
        ("data_getting.py", "Step 2: Downloading historical price data"),
        ("cleaning.py", "Step 3: Cleaning and preprocessing data"),
        ("pair_getting.py", "Step 4: Finding cointegrated pairs"),
        ("edge.py", "Step 5: Calculating entry signals and OU parameters"),
        ("correlations.py", "Step 6: Calculating pair correlations during trade periods"),
        ("diversification.py", "Step 7: Kelly criterion position sizing"),
        ("portfolio_kelly.py", "Step 8: Executing trades based on sized signals"),
        ("comparison.py", "Step 9: Comparing ideal vs current portfolio construction"),
        ("execute.py", "Step 10: Executing trades via Alpaca API")
    ]
    
    results = {}
    
    for script, description in pipeline:
        success = run_script(script, description)
        results[script] = success
        
        if not success:
            print(f"\n{'!'*70}")
            print(f"PIPELINE STOPPED: {script} failed")
            print(f"{'!'*70}")
            break
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    for script, success in results.items():
        status = "[OK] SUCCESS" if success else "[X] FAILED"
        print(f"{status:15} - {script}")
    
    all_success = all(results.values())
    
    print("\n" + "="*70)
    if all_success:
        print("PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("PIPELINE FAILED - See errors above")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)