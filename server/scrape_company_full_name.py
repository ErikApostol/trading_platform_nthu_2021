import requests


def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


from tickers_for_trading_platform_202007_sorted import *
new_list = []
for stock in asset_candidates:
    y = get_symbol(stock[0])
    new_list.append((stock[0], y))

with open('tickers_sorted.py', 'w') as f:
    print('asset_candidates = ', new_list, file=f)
