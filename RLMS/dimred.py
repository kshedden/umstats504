"""
Demonstrate using dimension reduction (Sliced Inverse Regression)
followed by kernel regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from statsmodels.regression.dimred import SlicedInverseReg
from data_prep import dx
from patsy import dmatrix

pdf = PdfPages("dimred.pdf")

# Log income
dx["log2J10"] = np.log2(dx["J10"])

# Log hours worked
dx["log2J8"] = np.log2(dx["J8"])

# Name of the dependent variable
yv = "log2J10"

# The dependent variable
y = dx[yv]

fml = "0 + C(status) + year + age + educ + Female + log2J8" # + C(OCCUP08)
di = dmatrix(fml, dx).design_info
x = dmatrix(di, dx, return_type="dataframe").values

# Moments for standardization
xmn = x.mean(0)
xsd = x.std(0)
ymn = y.mean()
ysd = y.std()

# Standardize the covariates
x -= xmn
x /= xsd

# The covariates are colinear, so do an initial projection to remove
# the null-space.
u, s, vt = np.linalg.svd(x, 0)
ii = np.flatnonzero(s > 1e-8)
n = x.shape[0]
proj0 = np.sqrt(n) * vt.T[:, ii] / s[ii]
x = np.dot(x, proj0)

# Standardize the dependent variable
y -= ymn
y /= ysd

m = SlicedInverseReg(y, x)
r = m.fit(slice_n=50)

# The SIR projection matrix
proj = r.params.iloc[:, 0:6]

# The SIR-reduced variables
xp = np.dot(x, proj)

# Split the data into female and male subsets
xpf = xp[dx.Female==1, :]
xpm = xp[dx.Female==0, :]
yf = y[dx.Female==1].values
ym = y[dx.Female==0].values

# Local linear regression
def kreg(zp, xp, yp, s):
    du = ((xp - zp)**2).sum(1)
    w = np.exp(-du/s**2)
    w /= w.sum()
    xp = xp * w[:, None]
    yp = yp * w
    u, s, vt = np.linalg.svd(xp, 0)
    v = vt.T
    b = np.dot(v, np.dot(u.T, yp) / s)
    return np.dot(zp, b)

# status is the urbanicity variable
for status in 1, 2, 3, 4:
    # educ is the educational attainment
    for educ in 16, 18, 21:

        # Make preditions for females and males separately
        ypa = []
        for female in 0, 1:
            dz = dx.iloc[0:50, :].copy()
            dz["status"] = status
            dz["year"] = 2010 - 2000
            dz["age"] = np.linspace(18, 80, 50)
            dz["educ"] = educ
            dz["Female"] = female
            dz["OCCUP08"] = 1
            dz["log2J8"] = np.log2(160)
            dm = dmatrix(di, dz, return_type='dataframe').values
            dm -= xmn
            dm /= xsd
            dm = np.dot(dm, proj0)

            xp = xpf if female == 1 else xpm
            yy = yf if female == 1 else ym

            # Make a prediction at each age
            yp = np.zeros(50)
            for i in range(50):
                yp[i] = kreg(np.dot(dm[i, :], proj), xp, yy, 0.6)
            yp *= ysd
            yp += ymn
            ypa.append(yp)

        plt.clf()
        plt.axes([0.12, 0.12, 0.7, 0.8])
        plt.grid(True)
        plt.plot(dz["age"], ypa[0], '-', label="Male")
        plt.plot(dz["age"], ypa[1], '-', label="Female")
        ha, lb = plt.gca().get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, "center right")
        leg.draw_frame(False)
        plt.ylim(12, 15)
        plt.xlabel("Age", size=15)
        plt.ylabel("Log income", size=15)
        plt.title("status=%d, educ=%d" % (status, educ))
        pdf.savefig()

pdf.close()
