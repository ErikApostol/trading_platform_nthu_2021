import pandas as pd
from tickers_sorted_tw import * 
import yfinance as yf

tickers = list(zip(*asset_candidates))[0]
all_data = pd.DataFrame()
for ticker in tickers:
    try:
        new_data = yf.download(ticker, start="2010-01-01", end="2020-10-01")
        new_data['Ticker'] = ticker
        new_data['Date'] = new_data.index
        all_data = all_data.append(new_data, ignore_index=True)
        print(ticker, ' downloaded')
    except:
        print(ticker, ' failed')
        pass

all_data.to_csv('data_for_trading_platform_tw.csv', index=False, columns=['Date','Ticker','Open','High','Low','Close','Adj Close','Volume'])
