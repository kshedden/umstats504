import pandas as pd
import numpy as np
import os
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.optimize import minimize_scalar

# The date to process
dt = "2012-04-01"
base = os.path.join("results", dt)

# Load the traffic statistics (one record per minute
# within a day).
df = pd.read_csv(os.path.join(base, "traffic_stats.csv"))

# Calculate the Hurst index using means and variances.
def hurst(x):
    z = []
    # m is the block size
    for m in 15, 30, 60:
        y = np.reshape(x, (-1, m))
        v = y.mean(1).var()
        z.append([m, v])
    z = np.log(np.asarray(z))
    c = np.cov(z.T)
    b = c[0, 1] / c[0, 0]
    return b/2 + 1

# Calculate the Hurst index using absolute values.
def hurstabs(x):
    z = []
    # m is the block size
    for m in 15, 30, 60:
        y = np.reshape(x, (-1, m))
        v = np.mean(np.abs(y.mean(1)))
        z.append([m, v])
    z = np.log(np.asarray(z))
    c = np.cov(z.T)
    b = c[0, 1] / c[0, 0]
    return b + 1

# Calculate the Hurst index two ways, for all four data series.
print("Hurst parameters:")
for j, x in enumerate([df.Traffic, df.Sources, df.UDP, df.TCP]):
    print(df.columns[2+j])
    for diffo in range(3):
        x = np.asarray(x, dtype=np.float64)
        z = np.diff(x, diffo)
        z = z[0:60*(len(z)//60)]
        z -= z.mean()
        print("    %4d %7.3f %7.3f" % (diffo, hurst(z), hurstabs(z)))


# Covariance matrix for fractional Brownian motion
# of length n at Hurst parameter h.
def getR(h, n):
    k = np.arange(n)
    v = (k + 1)**(2*h) - 2*k**(2*h) + np.abs(k - 1)**(2*h)
    v /= 2
    r = toeplitz(v)
    return r, v

# Log-likelihood for data z at Hurst parameter h.
def getL(h, z):

    # https://www.hindawi.com/journals/mpe/2014/490568/
    r, v = getR(h, len(z))
    s, b = np.linalg.slogdet(r)
    if s != 1:
        1/0
    ll = -np.dot(z, solve_toeplitz(v, z)) / 2
    ll -= b/2
    return ll

# Estimation of the Hurst parameter using maximum likelihood.
for vn in ["Traffic", "Sources", "UDP", "TCP"]:
    z = df[vn]
    print(vn)

    for k in range(4):

        y = np.asarray(z.values, dtype=np.float64)
        y = np.diff(y, k)
        y -= y.mean()
        y /= y.std()

        ll = lambda h: -getL(h, y)

        f = minimize_scalar(ll, bounds=[0.3, 1], method="bounded")
        print("    %4d %7.3f" % (k, f.x))


print("\nSimulated Gaussian AR-1:\n")
for r in (0, 0.5, 0.9, 0.99):

    for k in range(4):

        y = np.random.normal(size=2000)
        for i in range(1, k):
            y[i] = r*y[i-1] + np.sqrt(1 - r**2)*y[i]

        ll = lambda h: -getL(h, y)

        f = minimize_scalar(ll, bounds=[0.3, 1], method="bounded")
        print("%4.2f %7.3f" % (r, f.x))


