import pandas as pd
import json

data_all = pd.read_csv("data_for_trading_platform.csv")
data_tw = pd.read_csv("data_for_trading_platform_tw.csv")

ticker_all = data_all.groupby('Ticker')['Ticker'].tail(1)
price_all = data_all.groupby('Ticker')['Close'].tail(1)
ticker_tw = data_tw.groupby('Ticker')['Ticker'].tail(1)
price_tw = data_tw.groupby('Ticker')['Close'].tail(1)

data = {}

for i in range(ticker_all.size):
    data[ticker_all.values[i]] = price_all.values[i]
for i in range(ticker_tw.size):
    data[ticker_tw.values[i]] = price_tw.values[i]

all_data = json.dumps(data)

with open("latest_trading_data.txt", "w") as fp:
    fp.write(json.dumps(data))

