import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from statsmodels.regression.dimred import SlicedInverseReg
from data_prep import dx
from patsy import dmatrix

pdf = PdfPages("dimred.pdf")

dx["log2J10"] = np.log2(dx["J10"])
dx["log2J8"] = np.log2(dx["J8"])

yv = "log2J10"
y = dx[yv]

fml = "0 + C(status) + year + age + educ + Female + log2J8" # + C(OCCUP08)"
di = dmatrix(fml, dx).design_info
x = dmatrix(di, dx, return_type="dataframe").values

xmn = x.mean(0)
xsd = x.std(0)

ymn = y.mean()
ysd = y.std()

x -= xmn
x /= xsd

u, s, vt = np.linalg.svd(x, 0)
ii = np.flatnonzero(s > 1e-8)
n = x.shape[0]
proj0 = np.sqrt(n) * vt.T[:, ii] / s[ii]
x = np.dot(x, proj0)

y -= ymn
y /= ysd

m = SlicedInverseReg(y, x)
r = m.fit(slice_n=100)

proj = r.params.iloc[:, 0:5]
xp = np.dot(x, proj)
xpf = xp[dx.Female==1, :]
xpm = xp[dx.Female==0, :]

yf = y[dx.Female==1]
ym = y[dx.Female==0]


def kreg(zp, xp, yp, s):
    du = ((xp - zp)**2).sum(1)
    w = np.exp(-du/s**2)
    w /= w.sum()
    return np.dot(w, yp)

for status in 1, 2, 3, 4:
    for educ in 16, 18, 21:

        ypa = []
        for female in 0, 1:
            dz = dx.iloc[0:100, :].copy()
            dz["status"] = status
            dz["year"] = 2010 - 2000
            dz["age"] = np.linspace(18, 80, 100)
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

            yp = np.zeros(100)
            for i in range(100):
                yp[i] = kreg(np.dot(dm[i, :], proj), xp, yy, 0.3)
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
        plt.xlabel("Age", size=15)
        plt.ylabel("Log income", size=15)
        plt.title("status=%d, educ=%d" % (status, educ))
        pdf.savefig()

pdf.close()
