import pandas as pd
import yfinance as yf
import time
import random

data_file_to_update = 'data_for_trading_platform_tw.csv'
target_date_to_update = "2021-04-01"
all_data = pd.read_csv(data_file_to_update) # also update tw data
tickers = list(all_data.Ticker.unique())

for ticker in tickers:
    current_max_date = all_data[all_data['Ticker']==ticker].Date.max()
    if current_max_date < target_date_to_update:
        try:
            new_data = yf.download(ticker, start=current_max_date, end=target_date_to_update)
            new_data['Ticker'] = ticker
            new_data['Date'] = new_data.index
            all_data = all_data.append(new_data[new_data['Date']>current_max_date], ignore_index=True)
        except:
            print("No such ticker")

    time.sleep(random.uniform(10,20))

all_data.to_csv(data_file_to_update, index=False, columns=['Date','Ticker','Open','High','Low','Close','Adj Close','Volume'])
