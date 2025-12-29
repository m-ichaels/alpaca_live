import requests
import pandas as pd
from io import StringIO
import os

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

tables = pd.read_html(StringIO(response.text))
tickers = tables[0]["Symbol"].str.replace(".", " ", regex=False).tolist()

print(f"Found {len(tickers)} S&P 500 stocks:")
print(tickers)

df = pd.DataFrame({"Symbol": tickers})
df.to_csv("data/sp500_stocks.csv", index=False)
print("\nSaved to data/sp500_stocks.csv")