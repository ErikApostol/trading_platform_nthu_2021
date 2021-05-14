import pandas as pd
all_data = pd.read_csv('data_for_trading_platform.csv')
#all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')
        
import yfinance as yf
#from datetime import datetime

tickers = ['ARKF', 'ARKG', 'ARKK', 'ARKQ', 'ARKW']
for ticker in tickers:
    new_data = yf.download(ticker, start='2000-01-01', end="2021-01-01")
    new_data['Ticker'] = ticker
    new_data['Date'] = new_data.index
    all_data = all_data.append(new_data, ignore_index=True)

all_data.to_csv('data_for_trading_platform.csv', index=False, columns=['Date','Ticker','Open','High','Low','Close','Adj Close','Volume'])
