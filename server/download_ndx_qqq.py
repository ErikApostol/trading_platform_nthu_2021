import pandas as pd
all_data = pd.read_csv('data_for_trading_platform.csv')
#all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')
        
import yfinance as yf
#from datetime import datetime

tickers = ['AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'CDNS', 'CDW', 'CERN', 'CHKP', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DLTR', 'DOCU', 'DXCM', 'EA', 'EBAY', 'EXC', 'FAST', 'FB', 'FISV', 'FOX', 'FOXA', 'GILD', 'GOOG', 'GOOGL', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MTCH', 'MU', 'MXIM', 'NFLX', 'NTES', 'NVDA', 'NXPI', 'OKTA', 'ORLY', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PTON', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'SPLK', 'SWKS', 'TCOM', 'TEAM', 'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRSN', 'VRTX', 'WBA', 'WDAY', 'XEL', 'XLNX', 'ZM', 'QQQ']

for ticker in tickers:
    if ticker not in all_data['Ticker']:
        new_data = yf.download(ticker, start='2000-01-01', end="2021-04-01")
        new_data['Ticker'] = ticker
        new_data['Date'] = new_data.index
        all_data = all_data.append(new_data, ignore_index=True)

all_data.to_csv('data_for_trading_platform_new.csv', index=False, columns=['Date','Ticker','Open','High','Low','Close','Adj Close','Volume'])
