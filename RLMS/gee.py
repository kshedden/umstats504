import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from data_prep import dx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("rlms_gee_plots.pdf")

# Mean structure, with and without sex effects
fml0 = "I(np.log2(J10)) ~ C(status) + bs(year, 5) + bs(age, 5) + bs(educ, 5) + C(OCCUP08) + I(np.log2(J8))"
fml1 = "I(np.log2(J10)) ~ C(status) + bs(year, 5) + bs(age, 5)*Female + bs(educ, 5) + C(OCCUP08) + I(np.log2(J8))"
fml2 = "I(np.log2(J10)) ~ C(status) + (bs(year, 5) + bs(age, 5) + bs(educ, 5))*Female + C(OCCUP08) + I(np.log2(J8))"
fml3 = "I(np.log2(J10)) ~ (C(status) + bs(year, 5) + bs(age, 5) + bs(educ, 5))*Female + C(OCCUP08) + I(np.log2(J8))"
fml4 = "I(np.log2(J10)) ~ (bs(year, 5) + bs(age, 5) + bs(educ, 5))*C(status)*Female + C(OCCUP08) + I(np.log2(J8))"

# OLS model with/without sex effects
ols_model0 = sm.OLS.from_formula(fml0, data=dx)
ols_result0 = ols_model0.fit()
ols_model1 = sm.OLS.from_formula(fml1, data=dx)
ols_result1 = ols_model1.fit()
ols_model2 = sm.OLS.from_formula(fml2, data=dx)
ols_result2 = ols_model2.fit()
ols_model3 = sm.OLS.from_formula(fml3, data=dx)
ols_result3 = ols_model3.fit()
ols_model4 = sm.OLS.from_formula(fml4, data=dx)
ols_result4 = ols_model4.fit()

# GEE model with/without sex effects
gee_model0 = sm.GEE.from_formula(fml0, groups="idind", cov_struct=sm.cov_struct.Exchangeable(), data=dx)
gee_result0 = gee_model0.fit()
gee_model1 = sm.GEE.from_formula(fml1, groups="idind", cov_struct=sm.cov_struct.Exchangeable(), data=dx)
gee_result1 = gee_model1.fit()
gee_model2 = sm.GEE.from_formula(fml2, groups="idind", cov_struct=sm.cov_struct.Exchangeable(), data=dx)
gee_result2 = gee_model2.fit()
gee_model3 = sm.GEE.from_formula(fml3, groups="idind", cov_struct=sm.cov_struct.Exchangeable(), data=dx)
gee_result3 = gee_model3.fit()
gee_model4 = sm.GEE.from_formula(fml4, groups="idind", cov_struct=sm.cov_struct.Exchangeable(), data=dx)
gee_result4 = gee_model4.fit()

# Compare the two models using score tests (full sex effects, limited sex effects)
print(gee_model1.compare_score_test(gee_result0))
print(gee_model2.compare_score_test(gee_result1))
print(gee_model3.compare_score_test(gee_result2))
print(gee_model4.compare_score_test(gee_result3))

# Consider the correlation by PSU.
gee_model4x = sm.GEE.from_formula(fml4, groups="psu", cov_struct=sm.cov_struct.Exchangeable(), data=dx)
gee_result4x = gee_model4x.fit()

# Plot mean curves of log income by age for women and for men,
# each with a confidence band.
def conf_band(result, status, educ, title):

    df = result.model.data.frame
    dx = df.iloc[0:100, :]

    ti = "%s, status=%d, educ=%d" % (title, status, educ)

    plt.clf()
    plt.axes([0.12, 0.12, 0.7, 0.8])
    plt.grid(True)

    for female in 0, 1:
        dx.loc[:, "Female"] = female
        dx.loc[:, "age"] = np.linspace(18, 80, 100)
        dx.loc[:, "educ"] = educ
        dx.loc[:, "J8"] = 160 # 40 x 4 hours of work
        dx.loc[:, "year"] = 2010 - 2000 # reference year
        dx.loc[:, "status"] = status
        dx.loc[:, "OCCUP08"] = 1

        # Get the estimated conditional mean values, and their
        # standard errors
        dm = patsy.dmatrix(result.model.data.design_info, dx,
                           return_type="dataframe")
        pr = np.dot(dm, result.params)
        va = np.dot(dm, np.dot(result.cov_params(), dm.T))
        se = np.sqrt(np.diag(va))

        label = "Female" if female == 1 else "Male"
        plt.plot(dx.age, pr, '-', label=label)

        plt.fill_between(dx.age, pr - 3*se, pr + 3*se, color='lightgrey')

    plt.title(ti)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
    plt.xlabel("Age (years)", size=15)
    plt.ylabel("Expected monthly income (log2 RUB)", size=15)
    plt.ylim(12, 15)
    pdf.savefig()

# Plot the estimated mean function (for log2 wages) for women and
# men over a range of ages, controlling for specific levels of
# status (region type), and educ (educational level).
for status in 1, 2, 3, 4:
    for educ in 16, 18, 21:
        conf_band(ols_result4, status=status, educ=educ, title="OLS")
        conf_band(gee_result4, status=status, educ=educ, title="GEE")

pdf.close()
