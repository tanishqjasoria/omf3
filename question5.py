import omf_assignment
import pandas as pd
import numpy as np
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

ans1 = omf_assignment.markowitz_optimization_model(OM, M, 0.3, n, m, number=1)
ans2 = omf_assignment.markowitz_optimization_model(OM, M, 0.3, n, m, number=2)
ans3 = omf_assignment.markowitz_optimization_model(OM, M, 0.3, n, m, number=3)


R_1 = -3
R_2 = 3
OM = (D.T @ D) * 1/n
E = np.ones((m, 1))
w_1 = omf_assignment.markowitz_optimization_model(OM, M, R_1, n, m, number=2)
w_2 = omf_assignment.markowitz_optimization_model(OM, M, R_2, n, m, number=2)
n = 20
l = np.linspace(0.05, 1, n)
ret = [0]*n
risk = [0]*n
risk1, risk2= w_1.T@OM@w_1, w_2.T@OM@w_2
print(risk1, risk2)
# print(R_1, R_2)
# print(risk1, risk2)
for i in range(n):
  ret[i] = l[i]*R_1 + (1 - l[i])*R_2
  risk[i] =  ((risk1 * l[i]**2) + (risk2 * (1 - l[i])**2))**0.5

min_risk = risk[np.argmin(risk)]
corresponding_ret = ret[np.argmin(risk)]
# Plotting the point of minimum variance for Markowitz II model where R = 0.3
mark2_ret = 1.5
tol = 1e-4
# mark2_risk = risk[np.where(ret == 0.3)]
index = np.where(np.isclose(ret, mark2_ret, tol))[0][0]
mark2_risk = risk[index]
fig = plt.plot(risk, ret, '-')
plt.xlabel('Risk')
plt.ylabel('Return')
# plt.plot(min_risk, corresponding_ret, 'rs')
# plt.plot(mark2_risk, mark2_ret, 'gs')
plt.plot()
plt.show()
