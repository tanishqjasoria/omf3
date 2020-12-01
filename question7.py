import omf_assignment
import pandas as pd
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt


data, stock_list = omf_assignment.read_data_20_assets()

A = []

for key in data:
  A.append(list(data[key]['Close']))

A = np.array(A)
A = pd.DataFrame(A.T)
n, m = (A.values[1:]).shape

OM = np.array(A.cov().values)
M = np.array(A.mean().values)
D = np.array(A.values) - M
D = D[1:]

e = np.ones((20,1))
mu_f = 0.0575/12
M = M.reshape(20,1)
wd = ((inv(OM))@(M-mu_f*e))/((e.T@inv(OM))@(M-mu_f*e))
market_return = wd.T@M
market_risk = wd.T@OM@wd

mu_m = (market_return.item(0)-mu_f)/(market_risk.item(0))**0.5


print("a) Slope of CML = ", mu_m)
print("b) Slope of SML = ", mu_m - mu_f)
print("c)Market Portfolio:")
print("Expected return = ", market_return.item(0))
print("Variance = ", market_risk.item(0)**0.5)
print("BETA:")
beta = 0
for i, mu_i in enumerate(M):
  print("[", str(stock_list[i]), "] = ", ((mu_i-mu_f)/(mu_m - mu_f)))