from flask import Flask, Blueprint, render_template, session, jsonify, request, redirect, url_for, flash, g, Markup, abort
# from flask_ipban import IpBan
from flask_login import login_required, current_user, login_user, logout_user
#from flask_environments import Environments
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import pytz
import re
import time
import json
import psycopg2
import psycopg2.extras
from tickers_sorted import *
from tickers_sorted_tw import *
from black_list import black_list
from postgresql_config import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from cvxopt import matrix, solvers
from cvxopt.blas import dot
from cvxopt.solvers import qp

import pickle

UPLOAD_FOLDER = 'static/custom_data/'
ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

main = Flask(__name__)

# https://stackoverflow.com/questions/24222220/block-an-ip-address-from-accessing-my-flask-app-on-heroku
ip_ban_list = []
@main.before_request
def block_method():
    ip = request.environ.get('REMOTE_ADDR')
    if ip in ip_ban_list:
        abort(403)

main.config['SECRET_KEY'] = os.urandom(30)
main.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from confidential_competitions import *
confidential_competitions_placeholder = ', '.join(['%s']*len(confidential_competitions_list))

# Source: https://uniwebsidad.com/libros/explore-flask/chapter-8/custom-filters
@main.template_filter('my_substitution')
def my_substitution(string):
    return re.sub(r'@[a-zA-Z0-9_\-\.]+', r'', string)

def ten_day_VaR(portfolio_value):
    # 95% z value = 1.645
    z = 1.645
    return_value = portfolio_value.pct_change().shift(-1).dropna()
    length = len(return_value)
    daily_ret = return_value.mean()
    daily_vol = return_value.std()
    return -1*(daily_ret * 10 - daily_vol * (10**0.5) * z)

# risk_free_rate = 0.00899 # U.S. 5 Year Treasury at 11:31 PM EDT, Jun 28, 2021
risk_free_rate = 0
def Linear_Reg(x,y):
    '''
    input
    x: market excess return, np.ndarray
    y: portfolio excexx return, np.ndarray
    
    output
    slope, intercept, R^2
    '''
    from sklearn import linear_model
    import numpy
    model = linear_model.LinearRegression(fit_intercept=True)
    X = x[:,numpy.newaxis]
    model.fit(X,y)
    yhat = model.predict(X)
    SSR = sum((y-yhat)**2)
    SST = sum((y-numpy.mean(y))**2)
    r_squared = 1 - (float(SSR))/SST
    return model.coef_[0], model.intercept_, r_squared
    # alpha = model.intercept_, beta = model.coef_[0]

#env.filters['my_substitution'] = my_substitution
                                    



# def connect_db():
#     sql = sqlite3.connect('strategy.db', timeout=50)
#     sql.row_factory = sqlite3.Row
#     return sql 
# 
# def get_db():
#     if not hasattr(g, 'sqlite_db'):
#         g.sqlite_db = connect_db()
#     return g.sqlite_db
# 
# @main.teardown_appcontext
# def close_db(error):
#     if hasattr(g, 'sqlite_db'):
#         g.sqlite_conn.close()


@main.route('/')
def index():
    return render_template('index.html')
    # return render_template('index_temp_0314.html')




@main.route('/create_strategy', methods=['GET', 'POST'])
def create_strategy():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    if session['USERNAME'] in black_list:
        flash('我們已經暫停您建立策略的權利，有疑問請洽finteck@my.nthu.edu.tw', 'danger')
        return redirect('/')
    print('last_creation_time: ', session['last_creation_time'])
    if time.time() - session['last_creation_time'] < 30:
        print('Strategy ', request.form['strategy_name'], ' rejected to create because of insufficient time gap.')
        flash('每兩次建立策略須間隔200秒', 'danger')
        return redirect('/')

    if request.method == 'GET':
        tw = request.values.get('tw')
        tw_digit = 1 if tw=='true' else 0 if tw=='false' else None
    
    if request.method == 'POST':
        strategy_name = request.form['strategy_name']
        create_date = datetime.strftime(datetime.now() + timedelta(hours=8), '%Y/%m/%d %H:%M:%S.%f') 
        session['last_creation_time'] = time.time()

        if strategy_name == '':
            flash('請取一個名字', 'danger')
            return render_template('create_strategy.html', asset_candidates=asset_candidates if tw=='false' else asset_candidates_tw if tw=='true' else None, tw=tw)
        competition = request.form['competition']
        tw = request.form['tw']
        tw_digit = 1 if tw=='true' else 0 if tw=='false' else None
        tickers = sorted(list(set(request.form.getlist('asset_ticker'))))
        print('The list of assets: ', tickers)

        
        # Turn off progress printing
        solvers.options['show_progress'] = False
        
        start_dates = [datetime(2015, 1, 1),
                       datetime(2015, 4, 1),
                       datetime(2015, 7, 1),
                       datetime(2015, 10, 1),
                       datetime(2016, 1, 1),
                       datetime(2016, 4, 1),
                       datetime(2016, 7, 1),
                       datetime(2016, 10, 1),
                       datetime(2017, 1, 1),
                       datetime(2017, 4, 1),
                       datetime(2017, 7, 1),
                       datetime(2017, 10, 1),
                       datetime(2018, 1, 1),
                       datetime(2018, 4, 1),
                       datetime(2018, 7, 1),
                       datetime(2018, 10, 1),
                       datetime(2019, 1, 1),
                       datetime(2019, 4, 1),
                       datetime(2019, 7, 1),
                       datetime(2019, 10, 1),
                       datetime(2020, 1, 1),
                       datetime(2020, 4, 1),
                       datetime(2020, 7, 1),
                       datetime(2020, 10, 1),
                       datetime(2021, 1, 1),
                       datetime(2021, 4, 1) ]
        
        if tw=='false': 
            all_data = pd.read_csv('data_for_trading_platform.csv')  
        elif tw=='true':
            all_data = pd.read_csv('data_for_trading_platform_tw.csv')  
        else:
            all_data = None
        all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')
        
        def stockpri(ticker, start, end):
            data = all_data[ (all_data['Ticker']==ticker) & (all_data['Date']>=start) & (all_data['Date']<=end) ]
            data.set_index('Date', inplace=True)
            data = data['Adj Close']
            return data
        
        
        portfolio_value = pd.Series([100])
        optimal_weights = None
        hist_return_series = pd.DataFrame(columns=['quarter', 'quarterly_returns'])

        index_returns_full = pd.Series()
        portfolio_returns_full = pd.Series()
        
        for i in range(len(start_dates)-3):
        
            ### Take 6 months to backtest ###
        
            start = start_dates[i]
            end   = start_dates[i+2]
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()

            if log_returns.empty:
                continue
            
            mu = np.exp(log_returns.mean()*252).values 
            # Markowitz frontier
            profit = np.linspace(np.amin(mu), np.amax(mu), 100)
            frontier = []
            w = []
            if len(tickers) >= 3:
                for p in profit:
                    # Problem data.
                    n = len(tickers)
                    S = matrix(log_returns.cov().values*252)
                    pbar = matrix(0.0, (n,1))
                    # Gx <= h
                    G = matrix(0.0, (2*n,n))
                    G[::(2*n+1)] = 1.0
                    G[n::(2*n+1)] = -1.0
                    # h = matrix(1.0, (2*n,1))
                    h = matrix(np.concatenate((0.5*np.ones((n,1)), -0.03*np.ones((n,1))), axis=0))
                    A = matrix(np.concatenate((np.ones((1,n)), mu.reshape((1,n))), axis=0))
                    b = matrix([1, p], (2, 1))
                    
                    # Compute trade-off.
                    res = qp(S, -pbar, G, h, A, b)
                
                    if res['status'] == 'optimal':
                        res_weight = res['x']
                        s = math.sqrt(dot(res_weight, S*res_weight))
                        frontier.append(np.array([p, s]))
                        w.append(res_weight)
            elif len(tickers) == 2:
                for p in profit:
                    S = log_returns.cov().values*252
                    res_weight = [1 - (p-mu[0])/(mu[1]-mu[0]), (p-mu[0])/(mu[1]-mu[0])]
                    if (res_weight[0] < 0.03) or (res_weight[0] > 0.97):
                        continue
                    s = math.sqrt(np.matmul(res_weight, np.matmul(S, np.transpose(res_weight))))
                    frontier.append(np.array([p, s]))
                    w.append(res_weight)


            frontier = np.array(frontier)
            if frontier.shape == (0,):
                continue
            x = np.array(frontier[:, 0])
            y = np.array(frontier[:, 1])
        
            frontier_sharpe_ratios = np.divide(x-1, y)
            optimal_portfolio_index = np.argmax(frontier_sharpe_ratios)
            optimal_weights = w[optimal_portfolio_index]
            
        
            ### paper trade on the next three months ###
        
            start = start_dates[i+2]
            end   = start_dates[i+3]

            if tw=='true':
                index_ticker = '0050.TW'
            elif tw=='false':
                index_ticker = 'SPY'
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers+[index_ticker] })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()

            index_returns_3_months = returns[index_ticker]
            data = data.drop(columns=[index_ticker])
            returns = returns.drop(columns=[index_ticker])
            log_returns = log_returns.drop(columns=[index_ticker])
            portfolio_returns_3_months = pd.Series(np.dot(returns, optimal_weights).flatten())

            index_returns_full = index_returns_full.append(index_returns_3_months, ignore_index=True)
            portfolio_returns_full = portfolio_returns_full.append(portfolio_returns_3_months, ignore_index=True)
        
            portfolio_cum_returns = np.dot(returns, optimal_weights).cumprod()
            portfolio_value_new_window = portfolio_value.iloc[-1].item() * pd.Series(portfolio_cum_returns)
            portfolio_value_new_window.index = pd.to_datetime(returns.index, format='%Y-%m-%d')
            portfolio_value = portfolio_value.append(portfolio_value_new_window) 

            # produce quarterly return
            hist_return_series.loc[len(hist_return_series)] = [str(start.year)+'Q'+str((start.month+2)//3), portfolio_cum_returns[-1]-1]
            
        if optimal_weights == None:
            sharpe_ratio = avg_annual_return = annual_volatility = max_drawdown = alpha = beta = r_squared = ten_day_var = 0
            optimal_weights = [0, ]*len(tickers)
            hist_returns = None
            conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
            cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
            cur.execute("""insert into strategy (strategy_name,
                                                 author,
                                                 create_date,
                                                 sharpe_ratio,
                                                 return,
                                                 volatility,
                                                 max_drawdown,
                                                 tw,
                                                 competition,
                                                 hist_returns,
                                                 alpha,
                                                 beta,
                                                 r_squared,
                                                 ten_day_var) 
                           values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning strategy_id;""",
                        (strategy_name, 
                         session['USERNAME'], 
                         create_date,
                         sharpe_ratio,
                         avg_annual_return,
                         annual_volatility,
                         max_drawdown,
                         tw_digit,
                         competition,
                         hist_returns,
                         alpha,
                         beta,
                         r_squared, 
                         ten_day_var
                        ) )
            strategy_id = cur.fetchone()[0]
            conn.commit()

            # record the list of tickers into database
            # strategy_id = cur.execute('select * from strategy where create_date=?', [create_date]).fetchone()['strategy_id']
            for i in range(len(tickers)):
                cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (%s, %s, %s)', 
                            (strategy_id, tickers[i], optimal_weights[i]))
                conn.commit()

            cur.close()
            conn.close()    

            print('Strategy_id ' + str(strategy_id) + ' optimization fails.')
            flash('無資料或無法畫出馬可維茲邊界，請換一個組合', 'danger')
            return render_template('create_strategy.html', asset_candidates=asset_candidates if tw=='false' else asset_candidates_tw if tw=='true' else None, tw=tw)
        
        avg_annual_return = np.exp(np.log(portfolio_value.pct_change() + 1).mean() * 252) - 1
        annual_volatility = portfolio_value.pct_change().std() * math.sqrt(252)
        sharpe_ratio = avg_annual_return/annual_volatility
        max_drawdown = - np.amin(np.divide(portfolio_value, np.maximum.accumulate(portfolio_value)) - 1)
        ten_day_var = ten_day_VaR(portfolio_value)
        beta, alpha, r_squared = Linear_Reg(index_returns_full.to_numpy().flatten()     - risk_free_rate, 
                                            portfolio_returns_full.to_numpy().flatten() - risk_free_rate)
        
        print('Sharpe ratio: ', sharpe_ratio, ', Return: ', avg_annual_return, ', Volatility: ', annual_volatility, ', Maximum Drawdown: ', max_drawdown)

        # hist_return_series.set_index('quarter', inplace=True)
        hist_returns = pickle.dumps(hist_return_series)

        conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
        cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
        cur.execute("""insert into strategy (strategy_name,
                                             author,
                                             create_date,
                                             sharpe_ratio,
                                             return,
                                             volatility,
                                             max_drawdown,
                                             tw,
                                             competition,
                                             hist_returns,
                                             alpha,
                                             beta,
                                             r_squared,
                                             ten_day_var) 
                       values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning strategy_id;""",
                    (strategy_name, 
                     session['USERNAME'], 
                     create_date,
                     sharpe_ratio,
                     avg_annual_return,
                     annual_volatility,
                     max_drawdown,
                     tw_digit,
                     competition,
                     hist_returns,
                     alpha,
                     beta,
                     r_squared, 
                     ten_day_var
                    ) )
        strategy_id = cur.fetchone()[0]
        conn.commit()

        # record the list of tickers into database
        # strategy_id = cur.execute('select * from strategy where create_date=?', [create_date]).fetchone()['strategy_id']
        for i in range(len(tickers)):
            cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (%s, %s, %s)', 
                        (strategy_id, tickers[i], optimal_weights[i]))
            conn.commit()

        cur.close()
        conn.close()    

        # fig, ax = plt.subplots()
        # hist_return_series.hist(column='quarterly_returns', by='quarter', ax=ax)
        hist_return_plot = hist_return_series.plot.bar(x='quarter', y='quarterly_returns').get_figure()
        plt.tight_layout()
        # plt.xticks(rotation=45)
        hist_return_plot.savefig('static/img/quarterly_returns/'+str(strategy_id)+'.png')
        plt.close()
        
        print(portfolio_value.head())
        print(portfolio_value.tail())
        # plt.xticks(rotation=45)
        plt.plot(portfolio_value.iloc[1:])
        plt.savefig('static/img/portfolio_values/'+str(strategy_id)+'.png')
        plt.close()
        print('Strategy_id ' + str(strategy_id) + ' optimization succeeds.')
        flash(Markup('回測已完成，詳情請<a href="/post_page?post_id=' + str(strategy_id) + '">點這裡查看</a>。'), 'success')
    return render_template('create_strategy.html', asset_candidates=asset_candidates if tw=='false' else asset_candidates_tw if tw=='true' else None, tw=tw)


@main.route('/create_strategy_upload', methods=['GET', 'POST'])
def create_strategy_upload():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    if session['USERNAME'] in black_list:
        flash('我們已經暫停您建立策略的權利，有疑問請洽finteck@my.nthu.edu.tw', 'danger')
        return redirect('/')
    print('last_creation_time: ', session['last_creation_time'])
    if time.time() - session['last_creation_time'] < 30:
        print('Strategy ', request.form['strategy_name'], ' rejected to create because of insufficient time gap.')
        flash('每兩次建立策略須間隔200秒', 'danger')
        return redirect('/')
    
    if request.method == 'POST':
        strategy_name = request.form['strategy_name']
        create_date = datetime.strftime(datetime.now() + timedelta(hours=8), '%Y/%m/%d %H:%M:%S.%f') 
        session['last_creation_time'] = time.time()
        
        if strategy_name == '':
            flash('請取一個名字', 'danger')
            return render_template('create_strategy_upload.html')

        f = request.files['fileToUpload']
        filename = create_date.replace('/', '').replace(' ', '').replace(':', '').replace('.', '') + '.csv'
        f.save(os.path.join(main.config['UPLOAD_FOLDER'], filename))

        all_data = pd.read_csv(main.config['UPLOAD_FOLDER'] + filename)  
        all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')

        tickers = list(all_data.columns)
        tickers.remove('Date')
        print('The list of assets: ', tickers)

        competition = request.form['competition']

        
        # Turn off progress printing
        solvers.options['show_progress'] = False
        
        
        def stockpri(ticker, start, end):
            data = all_data.loc[start:end, ['Date', ticker]]
            data.set_index('Date', inplace=True)
            data = data[ticker]
            return data
        
        
        portfolio_value = pd.Series([100])
        optimal_weights = None
        hist_return_series = pd.DataFrame(columns=['quarter', 'quarterly_returns'])

        start_dates = list(np.arange(0, len(all_data), 63)) + [len(all_data)-1]
        

        for i in range(len(start_dates)-3):
        
            ### Take 6 months to backtest ###
        
            start = start_dates[i]
            end   = start_dates[i+2]+1
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()

            if log_returns.empty:
                continue
            
            mu = np.exp(log_returns.mean()*252).values 
            # Markowitz frontier
            profit = np.linspace(np.amin(mu), np.amax(mu), 100)
            frontier = []
            w = []
            if len(tickers) >= 3:
                for p in profit:
                    # Problem data.
                    n = len(tickers)
                    S = matrix(log_returns.cov().values*252)
                    pbar = matrix(0.0, (n,1))
                    # Gx <= h
                    G = matrix(0.0, (2*n,n))
                    G[::(2*n+1)] = 1.0
                    G[n::(2*n+1)] = -1.0
                    # h = matrix(1.0, (2*n,1))
                    h = matrix(np.concatenate((0.5*np.ones((n,1)), -0.03*np.ones((n,1))), axis=0))
                    A = matrix(np.concatenate((np.ones((1,n)), mu.reshape((1,n))), axis=0))
                    b = matrix([1, p], (2, 1))
                    
                    # Compute trade-off.
                    res = qp(S, -pbar, G, h, A, b)
                
                    if res['status'] == 'optimal':
                        res_weight = res['x']
                        s = math.sqrt(dot(res_weight, S*res_weight))
                        frontier.append(np.array([p, s]))
                        w.append(res_weight)
            elif len(tickers) == 2:
                for p in profit:
                    S = log_returns.cov().values*252
                    res_weight = [1 - (p-mu[0])/(mu[1]-mu[0]), (p-mu[0])/(mu[1]-mu[0])]
                    if (res_weight[0] < 0.03) or (res_weight[0] > 0.97):
                        continue
                    s = math.sqrt(np.matmul(res_weight, np.matmul(S, np.transpose(res_weight))))
                    frontier.append(np.array([p, s]))
                    w.append(res_weight)


            frontier = np.array(frontier)
            if frontier.shape == (0,):
                continue
            x = np.array(frontier[:, 0])
            y = np.array(frontier[:, 1])
        
            frontier_sharpe_ratios = np.divide(x-1, y)
            optimal_portfolio_index = np.argmax(frontier_sharpe_ratios)
            optimal_weights = w[optimal_portfolio_index]
            
        
            ### paper trade on the next three months ###
        
            start = start_dates[i+2]
            end   = start_dates[i+3]+1
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()

        
            portfolio_cum_returns = np.dot(returns, optimal_weights).cumprod()
            portfolio_value_new_window = portfolio_value.iloc[-1].item() * pd.Series(portfolio_cum_returns)
            portfolio_value_new_window.index = pd.to_datetime(returns.index, format='%Y-%m-%d')
            portfolio_value = portfolio_value.append(portfolio_value_new_window) 

            # produce quarterly return
            hist_return_series.loc[len(hist_return_series)] = [str(start), portfolio_cum_returns[-1]-1]
            
        if optimal_weights == None:
            sharpe_ratio = avg_annual_return = annual_volatility = max_drawdown = ten_day_var = 0
            optimal_weights = [0, ]*len(tickers)
            hist_returns = None
            conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
            cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
            cur.execute("""insert into strategy (strategy_name,
                                                 author,
                                                 create_date,
                                                 sharpe_ratio,
                                                 return,
                                                 volatility,
                                                 max_drawdown,
                                                 tw,
                                                 competition,
                                                 hist_returns,
                                                 alpha,
                                                 beta,
                                                 r_squared,
                                                 ten_day_var) 
                           values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning strategy_id;""",
                        (strategy_name, 
                         session['USERNAME'], 
                         create_date,
                         sharpe_ratio,
                         avg_annual_return,
                         annual_volatility,
                         max_drawdown,
                         0, # tw_digit,
                         competition,
                         hist_returns,
                         0, # alpha,
                         0, # beta,
                         0, # r_squared, 
                         ten_day_var
                        ) )
            strategy_id = cur.fetchone()[0]
            conn.commit()

            # record the list of tickers into database
            # strategy_id = cur.execute('select * from strategy where create_date=?', [create_date]).fetchone()['strategy_id']
            for i in range(len(tickers)):
                cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (%s, %s, %s)', 
                            (strategy_id, tickers[i], optimal_weights[i]))
                conn.commit()

            cur.close()
            conn.close()    
            print('Strategy_id ' + str(strategy_id) + ' optimization fails.')
            flash('無資料或無法畫出馬可維茲邊界，請換一個組合', 'danger')
            return render_template('create_strategy_upload.html')
        
        avg_annual_return = np.exp(np.log(portfolio_value.pct_change() + 1).mean() * 252) - 1
        annual_volatility = portfolio_value.pct_change().std() * math.sqrt(252)
        sharpe_ratio = avg_annual_return/annual_volatility
        max_drawdown = - np.amin(np.divide(portfolio_value, np.maximum.accumulate(portfolio_value)) - 1)
        ten_day_var = ten_day_VaR(portfolio_value)
        
        print('Sharpe ratio: ', sharpe_ratio, ', Return: ', avg_annual_return, ', Volatility: ', annual_volatility, ', Maximum Drawdown: ', max_drawdown)

        # hist_return_series.set_index('quarter', inplace=True)
        hist_returns = pickle.dumps(hist_return_series)

        conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
        cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
        cur.execute("""insert into strategy (strategy_name,
                                             author,
                                             create_date,
                                             sharpe_ratio,
                                             return,
                                             volatility,
                                             max_drawdown,
                                             tw,
                                             competition,
                                             hist_returns,
                                             alpha,
                                             beta,
                                             r_squared,
                                             ten_day_var) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning strategy_id;""",
                    (strategy_name, 
                     session['USERNAME'], 
                     create_date,
                     sharpe_ratio,
                     avg_annual_return,
                     annual_volatility,
                     max_drawdown,
                     0, # tw_digit,
                     competition,
                     hist_returns,
                     0, # alpha,
                     0, # beta,
                     0, # r_squared, 
                     ten_day_var
                    ) )
        strategy_id = cur.fetchone()[0]
        conn.commit()

        # record the list of tickers into database
        # strategy_id = cur.execute('select * from strategy where create_date=%s', (create_date,)).fetchone()['strategy_id']
        for i in range(len(tickers)):
            cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (%s, %s, %s)', 
                        (strategy_id, tickers[i], optimal_weights[i]))
            conn.commit()

        cur.close()
        conn.close()    

        # fig, ax = plt.subplots()
        # hist_return_series.hist(column='quarterly_returns', by='quarter', ax=ax)
        hist_return_plot = hist_return_series.plot.bar(x='quarter', y='quarterly_returns').get_figure()
        plt.tight_layout()
        hist_return_plot.savefig('static/img/quarterly_returns/'+str(strategy_id)+'.png')
        plt.close()
        
        print(portfolio_value.head())
        print(portfolio_value.tail())
        plt.plot(portfolio_value.iloc[1:])
        plt.savefig('static/img/portfolio_values/'+str(strategy_id)+'.png')
        plt.close()
        print('Strategy_id ' + str(strategy_id) + ' optimization succeeds.')
        flash(Markup('回測已完成，詳情請<a href="/post_page?post_id=' + str(strategy_id) + '">點這裡查看</a>。'), 'success')
    return render_template('create_strategy_upload.html')


@main.route('/login')
def login():
    return render_template('login.html')


@main.route('/login', methods=['POST'])
def login_post():
    password = request.form.get('password')
    username = request.form.get('username')

    conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
    cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cur.execute('select * from users where username=%s', (username,))
    sql_result = cur.fetchone()
    cur.close()
    conn.close()    

    if (not sql_result) or (not check_password_hash(sql_result['password'], password)):
        flash('使用者代號不對或密碼不對，請再試一次。', 'danger')
        return redirect('/login')

    print(sql_result['username'], sql_result['user_id'])
    session['login'] = True
    session['user_id'] = sql_result['user_id']
    session['USERNAME'] = sql_result['username']
    session['last_creation_time'] = 0
    session['vip'] = sql_result['vip']
    return redirect('/')


@main.route('/logout')
#@login_required
def logout():
    #logout_user()
    session['user_id'] = -1
    session['USERNAME'] = None
    session['login'] = False
    session['last_creation_time'] = None
    session['vip'] = None
    return redirect('/login')


@main.route('/signup')
def signup():
    return render_template('signup.html')


@main.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
    cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cur.execute('select * from users where username=%s', (username,))
    sql_result = cur.fetchone()
    if sql_result:  # if a user is found, we want to redirect back to signup page so user can try again
        cur.close()
        conn.close()    
        flash('這個Email地址已經被使用', 'danger')
        return redirect('/signup')

    if not (password == confirm_password):
        cur.close()
        conn.close()    
        flash('所輸入兩次密碼不同', 'danger')
        return redirect('/signup')

    # create new user with the form data. Hash the password so plaintext version isn't saved.
    cur.execute('insert into users (username, password) values (%s, %s)', (username, generate_password_hash(password)))
    conn.commit()
    cur.close()
    conn.close()    

    print('registered')
    flash('已成功註冊', 'success')

    return redirect('/login')


@main.route('/forum')
def forum_index():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
    cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cur.execute("""select * from strategy 
                   where competition not in (""" + confidential_competitions_placeholder + """) or author=%s 
                   order by strategy_id desc;""", 
                confidential_competitions_list + (session['USERNAME'],))
    data = cur.fetchall()
    cur.close()
    conn.close()    

    content_list = []
    for d in data:
        content_list.append({
            "id": d['strategy_id'],
            "time": d['create_date'],
            "user_id": None,
            "user_email": None,
            "user_name": d['author'],
            "comment": None,
            "title": d['strategy_name'],
            "video_id": None
        })

    return_data = {
        "count": len(data),
        "content": content_list
    }

    return render_template('forum.html', forum_data=return_data)


@main.route('/post_page', methods=['GET'])
# @login_required
def post_page():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    post_id = int(request.values.get('post_id'))

    conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
    cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cur.execute('select * from strategy where strategy_id=%s', (post_id,))
    strategy_content_list = cur.fetchone()
    cur.execute('select * from assets_in_strategy where strategy_id=%s', (post_id,))
    asset_list = cur.fetchall()
    cur.execute('select * from comment where strategy_id=%s', (post_id,))
    comment_list = cur.fetchall()
    cur.close()
    conn.close()    

    print(asset_list)

    with open('latest_trading_data.txt') as all_data:
        all_data_close = json.load(all_data)

    return_data = {
        "strategy_content": strategy_content_list,
        "asset_content": asset_list,
        "comment_content": comment_list,
        "comment_count": len(comment_list),
        "asset_candidates": dict(asset_candidates + asset_candidates_tw),
        "all_trading_data": all_data_close
    }

    return render_template('post_page.html', data=return_data, strategy_id=str(post_id))

@main.route('/comment', methods=['POST'])
#@login_required
def post_comment_data():
    comment = request.form['comment']
    strategy_id = request.form['strategy_id']
    author = session['USERNAME']
    comment_date = datetime.strftime(datetime.utcnow().replace(tzinfo=pytz.timezone('Asia/Taipei')), '%Y/%m/%d %H:%M') 

    conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
    cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cur.execute('insert into comment (author, strategy_id, comment, date) values (%s, %s, %s, %s)', (author, strategy_id, comment, comment_date))
    conn.commit()

    cur.close()
    conn.close()    
    return redirect('/post_page?post_id='+str(strategy_id))


#@main.route('/profile')
#@login_required
#def profile():
#    return render_template('profile.html', name=current_user.name)


@main.route('/analysis_result')
#@login_required
def analysis_result():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')

    sortby = request.values.get('sortby')
    competition = request.values.get('competition')
    tw = request.values.get('tw')
    tw_digit = 1 if tw=='true' else 0 if tw=='false' else None

    conn = psycopg2.connect(database=POSTGRESQL_DATABASE, user=POSTGRESQL_USER)
    cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    if sortby == 'competition':
        if competition in confidential_competitions_list:
            cur.close()
            conn.close()
            flash('本競賽暫不開放查詢', 'danger')
            return redirect('analysis_result?sortby=default&tw=' + tw +'&competition=none')
        cur.execute("""select b.strategy_id,
                              a.author,
                              b.create_date,
                              b.return,
                              a.sharpe_ratio,
                              b.max_drawdown,
                              b.strategy_name,
                              b.volatility
                       from (select author, max(sharpe_ratio) as sharpe_ratio from strategy group by author) as a
                            join strategy as b on a.author=b.author and a.sharpe_ratio=b.sharpe_ratio
                       where b.competition=%s
                       order by a.sharpe_ratio desc;""", (competition,))
        #cur.execute("""select strategy_id,
        #                     author,
        #                     create_date,
        #                     return,
        #                     max(sharpe_ratio) as sharpe_ratio,
        #                     max_drawdown,
        #                     strategy_name,
        #                     volatility
        #               from strategy
        #               where competition=%s
        #               group by author
        #               order by sharpe_ratio desc;""", (competition,))
        sql_results = cur.fetchall()
        num_records_hidden = min(5, len(sql_results))
        if session['vip']==False:
            for i in range(num_records_hidden):
                for key in sql_results[i].keys():
                    if key not in ['return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
                        sql_results[i][key] = '*****'
    elif sortby == 'default':
        cur.execute("""select * from strategy 
                       where tw=%s and (competition not in (""" + confidential_competitions_placeholder + """) or author=%s) 
                       order by strategy_id desc limit 200;""", 
                    (tw_digit,) + confidential_competitions_list + (session['USERNAME'],))
        sql_results = cur.fetchall()
    elif sortby == 'myself':
        cur.execute("""select * from strategy 
                       where tw=%s and author=%s 
                       order by strategy_id desc;""", 
                    (tw_digit, session['USERNAME']))
        sql_results = cur.fetchall()
    elif sortby == 'return':
        cur.execute("""select * from strategy 
                       where tw=%s and (competition not in (""" + confidential_competitions_placeholder + """) or author=%s) 
                       order by return desc limit 1000;""", 
                    (tw_digit,) + confidential_competitions_list + (session['USERNAME'],))
        sql_results = cur.fetchall()
        # num_records_hidden = min(5, len(sql_results))
        # if session['vip']==False:
        #     for i in range(num_records_hidden):
        #         for key in sql_results[i].keys():
        #             if key not in ['return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
        #                 sql_results[i][key] = '*****'
        if session['vip']==False:
            return_threshold = 1
            for record in sql_results:
                if record['return'] > return_threshold:
                    for key in record.keys():
                        if key not in ['return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
                            record[key] = '*****'
                else:
                    break
    elif sortby == 'sharpe':
        cur.execute("""select * from strategy 
                       where tw=%s and (competition not in (""" + confidential_competitions_placeholder + """) or author=%s)  
                       order by sharpe_ratio desc limit 1000;""", 
                    (tw_digit,) + confidential_competitions_list + (session['USERNAME'],))
        sql_results = cur.fetchall()
        if session['vip']==False:
            sharpe_threshold = 5
            for record in sql_results:
                if record['sharpe_ratio'] > sharpe_threshold:
                    for key in record.keys():
                        if key not in ['return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
                            record[key] = '*****'
                else:
                    break
    elif sortby == 'vol':
        cur.execute("""select * from strategy 
                       where tw=%s and (competition not in (""" + confidential_competitions_placeholder + """) or author=%s)  and volatility!=0 
                       order by volatility asc limit 1000;""", 
                    (tw_digit,) + confidential_competitions_list + (session['USERNAME'],))
        sql_results = cur.fetchall()
        if session['vip']==False:
            vol_threshold = 0.1
            for record in sql_results:
                if record['volatility'] < vol_threshold:
                    for key in record.keys():
                        if key not in ['return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
                            record[key] = '*****'
                else:
                    break
    elif sortby == 'mdd':
        cur.execute("""select * from strategy 
                       where tw=%s and (competition not in (""" + confidential_competitions_placeholder + """) or author=%s)  and max_drawdown!=0 
                       order by max_drawdown asc limit 1000;""", 
                    (tw_digit,) + confidential_competitions_list + (session['USERNAME'],))
        sql_results = cur.fetchall()
        if session['vip']==False:
            mdd_threshold = 0.15
            for record in sql_results:
                if record['max_drawdown'] < mdd_threshold:
                    for key in record.keys():
                        if key not in ['return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
                            record[key] = '*****'
                else:
                    break
    cur.close()
    conn.close()
    return render_template('result.html', results=sql_results, tw=tw)


if __name__ == "__main__":
    main.run(host='0.0.0.0', port=80)
