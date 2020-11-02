import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_prep import get_data
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pdf = PdfPages("bp_model_2wave.pdf")

d1 = get_data(1999)
d2 = get_data(2015)

# Code year relative to 2000
d1["Year"] = -1
d2["Year"] = 15

dx = pd.concat((d1, d2), axis=0)

def plot_fit_by_age(result, fml):

    # Create a dataframe in which all variables are at the reference
    # level
    da = dx.iloc[0:100, :].copy()
    da["RIDAGEYR"] = np.linspace(18, 80, 100)
    da["RIDRETH1"] = "OH"

    plt.figure(figsize=(8, 5))
    plt.clf()
    plt.axes([0.1, 0.1, 0.56, 0.8])
    plt.grid(True)

    yp, lab = [], []
    for year in -1, 15:
        for female in 0, 1:
            for bmi in 22, 28:

                db = da.copy()
                db.Female = female
                db.BMXBMI = bmi
                db.Year = year

                pr = result.predict(exog=db)
                yp.append(pr)

                la = "Female" if female == 1 else "Male"
                la += ", BMI=%.0f" % bmi
                lab.append(la)
                la += ", year=%4d" % (2000 + year)
                plt.plot(da.RIDAGEYR.values, pr.values, '-', label=la)

    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.xlabel("Age (years)", size=15)
    plt.ylabel("BP (mm/Hg)", size=15)
    plt.title(fml, size=11)
    plt.title(fml, fontdict={"fontsize": 9})
    pdf.savefig()

    # Plot differences between years
    plt.clf()
    plt.axes([0.1, 0.1, 0.67, 0.8])
    plt.grid(True)
    for j in range(4):
        plt.plot(da.RIDAGEYR.values, (yp[j + 4] - yp[j]).values, label=lab[j])
    plt.ylabel("BP (2015 - 1999)", size=15)
    plt.xlabel("Age (years)", size=15)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
    pdf.savefig()

# Model a common nonlinear structure for age and BMI using splines,
# then allow these curves to be translated
fml1 = "BPXSY1 ~ bs(RIDAGEYR, 5) + bs(BMXBMI, 4) + Female * RIDRETH1 + C(Year)"
model1 = sm.OLS.from_formula(fml1, data=dx)
result1 = model1.fit()
plot_fit_by_age(result1, fml1)

# Model a common nonlinear structure for age and BMI using splines,
# then allow these curves to be translated
fml2 = "BPXSY1 ~ (bs(RIDAGEYR, 5) + bs(BMXBMI, 4) + Female * RIDRETH1) * C(Year)"
model2 = sm.OLS.from_formula(fml2, data=dx)
result2 = model2.fit()
plot_fit_by_age(result2, fml2)

pdf.close()
