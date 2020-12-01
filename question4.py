import omf_assignment
import numpy as np

P = [
  [10, 15, 10],
  [15, 20, 15],
  [13, 18, 11],
  [15, 15, 14],
  [14, 16, 13]
]

P = np.array(P)
S = omf_assignment.calculate_rate_from_prices(P)
n, m = S.shape
M = np.mean(S, axis=0)
DM = S - M
OM = np.dot(DM.T, DM) * 1/n


from matplotlib import pyplot as plt

# Let R_1 = -1, R_2 = 1
A = np.array([
  [10, 15, 10],
  [15, 20, 15],
  [13, 18, 11],
  [15, 15, 14],
  [14, 16, 13]
])

R_1 = -1.0
R_2 = 1.0
A = A.T[:3].T

rate = np.zeros((A.shape[0]-1, A.shape[1]))
for i in range(rate.shape[0]):
  for j in range(rate.shape[1]):
    rate[i][j] = A[i+1][j]/A[i][j] - 1

n, m = rate.shape
mu = np.mean(rate, axis=0)
D = rate - mu
# Covariance matrix of return
om = (D.T @ D) *1/n
e = np.ones((m, 1))
w_1 = omf_assignment.markowitz_optimization_model(om, mu, R_1, n, m, number=2)
w_2 = omf_assignment.markowitz_optimization_model(om, mu, R_2, n, m, number=2)

n = 100
l = np.linspace(0, 1, n)
ret = np.zeros(n)
risk = np.zeros(n) #[0]*n
risk1 = w_1.T@om@w_1
risk2 = w_2.T@om@w_2

for i in range(n):
  ret[i] = l[i]*R_1 + (1 - l[i])*R_2
  risk[i] =  ((risk1 * l[i]**2) + (risk2 * (1 - l[i])**2))**0.5

#Calculating minimum variance points
risk_model1 = risk[np.argmin(risk)]
ret_model1 = ret[np.argmin(risk)]
index = np.where(np.isclose(ret,0.3, 1e-1))[0]
risk_model2 = risk[index]


print("The efficient frontier with the minimum variance point for Markowitz model 1 and 2 is as follows: ")
fig = plt.plot(risk, ret, '-')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.plot(risk_model1, ret_model1, 'ro')
plt.annotate("Min(Var): Model 1", (risk_model1, ret_model1))
plt.plot(risk_model2[1],0.3, 'ro')
plt.annotate("Min(Var): Model 2", (risk_model2[1], 0.3))
plt.show()
