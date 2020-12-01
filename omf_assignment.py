import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from scipy.optimize import minimize


def calculate_rate_from_prices(P):
  rate = np.zeros((P.shape[0]-1, P.shape[1]))
  for i in range(rate.shape[0]):
    for j in range(rate.shape[1]):
      rate[i][j] = P[i+1][j]/P[i][j] - 1
  return rate

def markowitz_optimization_model(OM, M, R, n, m, number=1):

  if number == 1:
    I = np.ones((m, 1))
    ans = np.dot(np.linalg.inv(OM), I)
    ans /= np.dot(np.dot(I.T, np.linalg.inv(OM)), I)
    return ans.T

  if number == 2:
    e = np.ones((m, 1))
    numerator = np.array([[2*R, float(M.T @ inv(OM) @ e)], [2, float(e.T @ inv(OM) @ e)]])
    denominator = np.array([[float(M.T @ inv(OM) @ M), float(M.T @ inv(OM) @ e)],
                            [float(e.T @ inv(OM) @ M), float(e.T @ inv(OM) @ e)]])
    lambda1 = det(numerator)/det(denominator)
    numerator = np.array([[float(M.T @ inv(OM) @ M), 2*R],
                          [float(e.T @ inv(OM) @ M), 2]])
    denominator = np.array([[float(M.T @ inv(OM) @ M), float(M.T @ inv(OM) @ e)],
                            [float(e.T @ inv(OM) @ M), float(e.T @ inv(OM) @ e)]])

    lambda2 = det(numerator)/det(denominator)
    sol = 0.5 * (lambda1 * (inv(OM) @ M) + lambda2 * (inv(OM) @ e.reshape(m)))
    return sol

  if number == 3:
    Objective = lambda w: w.T @ OM @ w
    def equality(x):
      e = np.ones((m))
      return(e @ x - 1)
    def inequality(x):
      return(M @ x - 0.3)

    b = (-np.inf, np.inf)
    temp=[]
    for i in range(m):
      temp.append(b)
    limit = tuple(temp)
    y = np.ones(m)
    constraint1 = {'type':'ineq', 'fun':inequality}
    constraint2 = {'type':'eq', 'fun':equality}
    constraints = ([constraint1, constraint2])
    sol = minimize(Objective, y, method='SLSQP', bounds=limit, constraints=constraints)
    return sol


def read_data_20_assets():
  from os import listdir
  import pandas as pd
  stock_list = listdir('Data')
  data = {}

  for stock in stock_list:
    print(stock)
    _data = pd.read_csv('Data/' + stock, index_col='Date', parse_dates=True)
    _data = _data.reindex(index=_data.index[::-1])
    name = stock.split('.')[0]
    data[name] = _data.pct_change() * 100

  return data, stock_list