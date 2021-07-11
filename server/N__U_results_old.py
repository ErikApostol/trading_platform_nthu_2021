import pickle
import psycopg2
import psycopg2.extras
import pandas as pd

#def dict_factory(cursor, row):
#    d = {}
#    for idx, col in enumerate(cursor.description):
#        d[col[0]] = row[idx]
#    return d

conn = psycopg2.connect(database='root', user='root')
cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
cur.execute("select * from strategy where competition='Service_NTHU_Spring_2021'")
sql_results = cur.fetchall()
cur.close()
conn.close()

Service_NTHU_Spring_2021_results = pd.DataFrame(columns=['strategy_id', 'author', 'sharpe_ratio', 'competition', 'quarters_of_data', 'assets'])

for record in sql_results:
  conn = psycopg2.connect(database='root', user='root')
  cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
  cur.execute("select * from assets_in_strategy where strategy_id=%s", (record['strategy_id'],))
  asset_results = cur.fetchall()
  assets = [ asset['asset_ticker'] for asset in asset_results ]
  cur.close()
  conn.close()
  
  if record['hist_returns']:
    Service_NTHU_Spring_2021_results.loc[len(Service_NTHU_Spring_2021_results)] = [ record['strategy_id'],
                                            record['author'],
                                            record['sharpe_ratio'],
                                            record['competition'],
                                            len(pickle.loads(record['hist_returns'])) + 2,
                                            assets ]
  else:
    Service_NTHU_Spring_2021_results.loc[len(Service_NTHU_Spring_2021_results)] = [ record['strategy_id'],
                                            record['author'],
                                            record['sharpe_ratio'],
                                            record['competition'],
                                            0,
                                            assets ]
                                            
Service_NTHU_Spring_2021_results.to_csv('Service_NTHU_Spring_2021_results.csv')
