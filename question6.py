import numpy as np
from numpy.linalg import inv
import omf_assignment

A = np.array([
  [10, 15, 10],
  [15, 20, 15],
  [13, 18, 11],
  [15, 15, 14],
  [14, 16, 13]
])

rate = omf_assignment.calculate_rate_from_prices(A)

n, m = rate.shape

mu = np.mean(rate, axis=0)
D = rate - mu
om = (D.T @ D) *1/n

e = np.ones((3,1))
mu_f = 0.0575/12
mu = mu.reshape(3,1)
wd = ((inv(om))@(mu-mu_f*e))/((e.T@inv(om))@(mu-mu_f*e))
market_return = wd.T@mu
market_risk = wd.T@om@wd

mu_m = (market_return.item(0)-mu_f)/(market_risk.item(0))**0.5

print("a)Market Portfolio:")
print("Expected return = ", market_return.item(0))
print("Variance = ", market_risk.item(0)**0.5)
print("b) Slope of CML = ", mu_m)
print("c) Slope of SML = ", mu_m - mu_f)
print("BETA")
beta=0
for i, mu_i in enumerate(mu):
  print("[", i+1, "] = ", ((mu_i-mu_f)/(mu_m - mu_f)))
