import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.duration.survfunc import plot_survfunc
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("survival.pdf")

# Simulate grouped data with censoring
n = 200
mng = np.r_[30, 25, 50, 40]   # Mean event times per group
mn = np.kron(mng, np.ones(n)) # Mean event times per individual
mf = 40 # Mean follow-up time
evt = -mn*np.log(np.random.uniform(size=4*n)) # Event times
fut = -mf*np.log(np.random.uniform(size=4*n)) # Follow up times
y = np.where(evt < fut, evt, fut) # The time that is observed
c = (y == evt).astype(np.int)   # Censoring indicator (1 if censored)
g = np.kron(np.arange(4), np.ones(n)) # Group labels
df = pd.DataFrame({"y": y, "c": c, "g": g})

# Estimate the survival functions
sf = []
for k, dx in df.groupby("g"):
    s = sm.SurvfuncRight(dx.y, dx.c)
    sf.append(s)

# Plot the survival function estimates.
plt.clf()
plot_survfunc(sf)
ha, lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
pdf.savefig()

# Simulate data for proportional hazards regression analysis
n = 400
xmat = np.random.normal(size=(n, 4))
lp = np.dot(xmat, np.r_[1, -1, 0, 0])
evt = - np.exp(-lp) * np.log(np.random.uniform(size=n)) # Event times
fut = -5 * np.log(np.random.uniform(size=n)) # Follow up times
y = np.where(evt < fut, evt, fut) # The time that is observed
c = (y == evt).astype(np.int)   # Censoring indicator (1 if censored)
df = pd.DataFrame({"y": y, "c": c, "x0": xmat[:, 0], "x1": xmat[:, 1], "x2": xmat[:, 2], "x3": xmat[:, 3]})

# Fit a proportional hazards model
m = sm.PHReg.from_formula("y ~ x0 + x1 + x2 + x3", status="c", data=df)
r = m.fit()
print(r.summary())

# Plot the estimated cumulative hazard function (if linear, the baseline
# hazard is approximately constant).
plt.clf()
plt.grid(True)
b = r.baseline_cumulative_hazard[0] # stratum zero
plt.plot(b[0], b[1])
plt.xlabel("Time")
plt.ylabel("Estimated cumulative hazard")
pdf.savefig()

pdf.close()
