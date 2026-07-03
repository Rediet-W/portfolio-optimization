import yfinance as yf
import pandas as pd
import os
import time

def fetch_data(tickers, start, end):
    # retry logic if the download fails due to locking
    for i in range(3):
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker')
            return data
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    return None

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    df = fetch_data(tickers, "2015-01-01", "2026-06-30")
    
    if df is not None:
        output_path = os.path.join("data", "processed", "market_data.csv")
        df.to_csv(output_path)
        print(f"Data saved successfully to {output_path}")
    else:
        print("Failed to download data after 3 attempts.")