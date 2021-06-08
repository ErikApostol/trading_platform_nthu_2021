import pickle
import psycopg2
import psycopg2.extras
import pandas as pd

conn = psycopg2.connect(database='root', user='root')
cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
# cur.execute("select * from strategy where competition in ('NTHU-NTU_2021_Spring_2021_April', 'Fintech_NTU_2021_Spring_NTHU_2021_Spring')")
cur.execute("select * from strategy where competition='NCCU_2021_Spring'")
sql_results = cur.fetchall()
cur.close()
conn.close()

NCCU_2021_Spring_results = pd.DataFrame(columns=['strategy_id', 'author', 'sharpe_ratio', 'competition', 'quarters_of_data', 'assets'])

for record in sql_results:
  conn = psycopg2.connect(database='root', user='root')
  cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
  cur.execute("select * from assets_in_strategy where strategy_id=%s", (record['strategy_id'],))
  asset_results = cur.fetchall()
  assets = [ asset['asset_ticker'] for asset in asset_results ]
  cur.close()
  conn.close()
  
  if record['hist_returns']:
    NCCU_2021_Spring_results.loc[len(NCCU_2021_Spring_results)] = [ record['strategy_id'],
                                            record['author'],
                                            record['sharpe_ratio'],
                                            record['competition'],
                                            len(pickle.loads(record['hist_returns'])) + 2,
                                            assets ]
  else:
    NCCU_2021_Spring_results.loc[len(NCCU_2021_Spring_results)] = [ record['strategy_id'],
                                            record['author'],
                                            record['sharpe_ratio'],
                                            record['competition'],
                                            0,
                                            assets ]
                                            
NCCU_2021_Spring_results.to_csv('NCCU_2021_Spring_results.csv')
