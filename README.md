Okay, currently it runs as follows:

1. "s&p500_stocks.py", "Step 1: Fetching S&P 500 ticker list"
2. "data_getting.py", "Step 2: Downloading historical price data"
3. "cleaning.py", "Step 3: Cleaning and preprocessing data"
4. "pair_getting.py", "Step 4: Finding cointegrated pairs"
5. "acceptable_to_trade.py", "Step 5: seeing if they are below the moving average line (and finding stop loss+half life)"
6. "take_profit.py", "Step 6: applies monte-carlo simulation to estimate the stop loss"
