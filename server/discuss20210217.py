            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()

            if log_returns.empty:
                continue
            
            # Markowitz frontier
            profit = np.linspace(0., 3., 100)
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
                    A = matrix(np.concatenate((np.ones((1,n)), np.exp(log_returns.mean()*252).values.reshape((1,n))), axis=0))
                    b = matrix([1, p], (2, 1))
                    
                    # Compute trade-off.
                    res = qp(S, -pbar, G, h, A, b)
                
                    if res['status'] == 'optimal':
                        res_weight = res['x']
                        print(type(res_weight))
                        s = math.sqrt(dot(res_weight, S*res_weight))
                        frontier.append(np.array([p, s]))
                        w.append(res_weight)
            elif len(tickers) == 2:
                for p in profit:
                    mu = np.exp(log_returns.mean()*252).values
                    S = log_returns.cov().values*252
                    res_weight = [1 - (p-mu[0])/(mu[1]-mu[0]), (p-mu[0])/(mu[1]-mu[0])]
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

