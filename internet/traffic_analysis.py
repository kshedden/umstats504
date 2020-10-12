import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.special import gammaln
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import acovf

# The date to process
dt = "2012-04-01"
base = os.path.join("results", dt)

# Create a set of graphs in one pdf file.
pdf = PdfPages("internet_%s.pdf" % dt)

# Load the traffic statistics (one record per minute
# within a day).
df = pd.read_csv(os.path.join(base, "traffic_stats.csv"))

# Rename the columns
cname = {"Traffic": "Total traffic", "UDP": "UDP", "TCP": "TCP",
         "Sources": "Unique sources"}

# Calculate the Kendall's tau autocorrelation for z at lag k.
def kt(z, k):
    n = len(z)
    x = z[0:n-k]
    y = z[k:n]
    return kendalltau(x, y).correlation

# Calculate the Kendall's tau autocorrelation for z at all
# lags up to and including k.
def kta(z, k):
    a = []
    for j in range(k+1):
        a.append(kt(z, j))
    a = np.asarray(a)
    return a

# Plot each column of data (total traffic, distinct sources,
# UDP traffic, TCP traffic).
for vn in ["Traffic", "Sources", "UDP", "TCP"]:
    z = df[vn]

    # Difference up to order 3
    for k in range(4):
        y = np.diff(z, k)
        y = y.astype(np.float64) / 1000

        # Plot the data
        plt.clf()
        plt.axes([0.17, 0.1, 0.75, 0.8])
        plt.plot(y)
        plt.grid(True)
        plt.xlabel("Minutes", size=15)
        plt.ylabel(cname[vn] + (" (%d-diff)") % k, size=15)
        pdf.savefig()

        # Plot the autocovariance with and without demeaning
        for demean in False, True:
            a = acovf(y, demean=demean, fft=True)
            plt.clf()
            plt.axes([0.2, 0.1, 0.7, 0.8])
            plt.grid(True)
            plt.plot(a)
            plt.xlabel("Lag", size=15)
            plt.ylabel(("Autocovariance (%d-diff)" % k) + (" de-meaned" if demean else ""), size=15)
            pdf.savefig()

        # Plot the tau-autocorrelation
        a = kta(y, 240)
        plt.clf()
        plt.axes([0.2, 0.1, 0.7, 0.8])
        plt.grid(True)
        plt.plot(a)
        plt.xlabel("Lag", size=15)
        plt.ylabel(("Tau autocorrelation (%d-diff)" % k) + (" de-meaned" if demean else ""), size=15)
        pdf.savefig()

        # Plot the quantile function
        plt.clf()
        plt.axes([0.17, 0.1, 0.75, 0.8])
        pp = np.linspace(0, 1, len(y))
        plt.plot(pp, np.sort(y))
        plt.grid(True)
        plt.xlabel("Probability point", size=15)
        plt.ylabel(cname[vn] + " quantile (%d-diff)" % k, size=15)
        pdf.savefig()

# Get null quantiles for Kendall's tau
def kt0(ntime, maxlag=240, nrep=100):
    z = []
    for j in range(nrep):
        y = np.random.uniform(size=ntime)
        v = kta(y, maxlag)
        z.append(v)
    z = np.asarray(z)
    z.sort(0)
    return z

# Plot the tau autocorrelations as quantile functions relative to null
# k = order of differencing
for j, vn in enumerate(["Traffic", "Sources", "UDP", "TCP"]):

    b = kt0(24*60-k, 240)

    for k in range(3):

        z = df[vn]
        y = np.diff(z, k)
        a = kta(y, 240)

        plt.clf()
        plt.grid(True)
        plt.plot(b[5, :], color='grey', alpha=0.5)
        plt.plot(b[50, :], color='blue', alpha=0.5)
        plt.plot(b[95, :], color='grey', alpha=0.5)
        plt.plot(a, color='red', alpha=0.5)
        plt.ylabel("Tau autocorrelation", size=15)
        plt.xlabel("Time", size=15)
        plt.title(cname[vn] + " order %d differences" % k)
        pdf.savefig()

# Calculate the binomial coefficient "n choose m".
def bincoeff(n, m):
    return np.exp(gammaln(n + 1) - gammaln(n - m + 1) - gammaln(m + 1))


# Calculate the l-moment of order k, for the data vector x.
# The order (k) must be equal to 1, 2, 3, or 4.
def lmoment(x, k):

    n = len(x)
    x = np.sort(x)
    ii = np.arange(1, n + 1, dtype=np.float64)

    if k == 1:
        return np.mean(x)
    elif k == 2:
        c = 2*ii - n - 1
        c /= bincoeff(n, 2)
        c /= 2
        return np.dot(x, c)
    elif k == 3:
        c = bincoeff(ii-1, 2) - 2*(ii-1)*(n-ii) + bincoeff(n-ii, 2)
        c /= bincoeff(n, 3)
        c /= 3
        return np.dot(x, c)
    elif k == 4:
        c = bincoeff(ii-1, 3) - 3*bincoeff(ii-1, 2)*(n-ii) + 3*(ii-1)*bincoeff(n-ii, 2) - bincoeff(n-ii, 3)
        c /= bincoeff(n, 4)
        c /= 4
        return np.dot(x, c)

# Calculate the first four L-moments of all outcomes,
# for each hour within a day.
lmom = np.zeros((4, 24, 4))
for k, vn in enumerate(["Traffic", "Sources", "UDP", "TCP"]):
    ii = 0
    z = df[vn] / 1000
    y = np.diff(z, 1)
    for hour in range(24):
        x = y[ii:min(len(y), ii+60)]
        ii += 60
        u = np.empty(4)
        for j in range(1, 5):
            u[j - 1] = lmoment(x, j)
        lmom[k, hour, :] = u

# Plot each L moment (calculated by hour) as a function of time.
for k, vn in enumerate(["Traffic", "Sources", "UDP", "TCP"]):
    for j in range(4):
        plt.clf()
        plt.axes([0.2, 0.1, 0.75, 0.8])
        plt.grid(True)
        plt.plot(lmom[k, :, j])
        plt.xlabel("Hour", size=15)
        plt.ylabel("L-moment of order %d" % (j + 1), size=15)
        plt.title(cname[vn])
        pdf.savefig()

def hill(y):
    y = np.sort(y)[::-1]
    y = np.log(y)
    h = np.cumsum(y)
    h /= np.arange(1, len(y) + 1)
    h = h[:-1] - y[1:]
    return h

# Hill estimators
for k, vn in enumerate(["Traffic", "Sources", "UDP", "TCP"]):

    z = df[vn]
    y = np.diff(z, 1)
    y = y[y > 0]
    h = hill(y)

    plt.clf()
    plt.grid(True)
    plt.plot(h)
    plt.xlabel("Span", size=15)
    plt.ylabel("Hill coefficient", size=15)
    plt.title(vn + " 1-diff")
    pdf.savefig()

# Hill estimators for simulated data
for k in range(3):

    hp = []
    for j in range(10):
        if k == 0:
            ti = "Gaussian"
            y = np.random.normal(size=1440)
            y = np.abs(y)
        elif k == 1:
            ti = "Exponential"
            y = -np.log(np.random.uniform(size=1440))
        elif k == 2:
            ti = "Pareto"
            y = np.random.pareto(1, 1440) + 1
        h = hill(y)
        hp.append(h)

    plt.clf()
    plt.grid(True)
    plt.title(ti)
    for h in hp:
        plt.plot(h, color='grey')
    plt.xlabel("Span", size=15)
    plt.ylabel("Hill coefficient", size=15)
    pdf.savefig()

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

# Use regularized regression to study the autoregressive structure of the
# traffic and unique sources series.

from numpy.lib.stride_tricks import as_strided
import statsmodels.api as sm

labs = ["Traffic", "Sources", "UDP", "TCP"]
for j,x in enumerate([df.Traffic, df.Sources, df.UDP, df.TCP]):

    x = np.asarray(x)
    x = np.log(x)
    x -= x.mean()

    # This is a trick to lag the data without copying
    z = as_strided(x, shape=(len(x)-30, 30), strides=(8, 8))

    y = z[:, 0]
    x = z[:, 1:30]

    # Use four approaches to regression
    for jj in 0, 1, 2, 3:

        params = []

        if jj == 0:
            title = "OLS"
            result = sm.OLS(y, x).fit()
            params.append(result.params)
        elif jj == 1:
            title = "Ridge"
            for alpha in 0.001, 0.01, 0.1:
                result = sm.OLS(y, x).fit_regularized(alpha=alpha, L1_wt=0)
                params.append(result.params)
        elif jj == 2:
            title = "Lasso"
            for alpha in 0.001, 0.01, 0.1:
                result = sm.OLS(y, x).fit_regularized(alpha=alpha, L1_wt=1)
                params.append(result.params)
        elif jj == 3:
            title = "Elastic net"
            for alpha in 0.00001, 0.001, 0.1:
                result = sm.OLS(y, x).fit_regularized(alpha=alpha, L1_wt=0.1)
                params.append(result.params)

        plt.clf()
        plt.title(labs[j] + " " + title)
        for p in params:
            plt.plot(np.arange(1, len(p)+1), p)
        plt.xlabel("Lag", size=15)
        plt.ylabel("Coefficient", size=15)
        plt.grid(True)
        plt.ylim(-1, 1)
        pdf.savefig()

pdf.close()
