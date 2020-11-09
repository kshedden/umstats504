import numpy as np
import pandas as pd
import statsmodels.api as sm


def sim1():

    np.random.seed(342)

    nrep = 100
    n = 100 # Number of groups
    m = 5   # Sample size per group

    rx = []
    for i in range(nrep):

        if i % 5 == 0:
            print(i)

        x = np.random.normal(size=n)
        x = np.kron(x, np.ones(m))
        x += np.random.normal(size=n*m)
        u = np.random.normal(size=n)
        u = np.kron(u, np.ones(m))
        y = 0*x + u + np.random.normal(size=n*m)
        g = np.kron(np.arange(n), np.ones(m))

        df = pd.DataFrame({"y": y, "x": x, "g": g})
        m1 = sm.OLS.from_formula("y ~ x", data=df).fit()
        m2 = sm.MixedLM.from_formula("y ~ x", groups="g", data=df).fit()
        m3 = sm.GEE.from_formula("y ~ x", groups="g", data=df).fit()
        m4 = sm.GEE.from_formula("y ~ x", groups="g",
                 cov_struct=sm.cov_struct.Exchangeable(), data=df).fit()

        rx.append([m1.tvalues["x"], m2.tvalues["x"], m3.tvalues["x"],
                   m4.tvalues["x"]])

    rx = np.asarray(rx)

    print(rx.mean(0))
    print(rx.std(0))

sim1()
