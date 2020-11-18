"""
Prepare a subset of data from the China Health and Nutrition Survey
for analysis.

The study is described here:

https://www.cpc.unc.edu/projects/china

The data are available here (registration required):

https://www.cpc.unc.edu/projects/china/data/datasets/data-downloads-registration
"""

import pandas as pd
import numpy as np

# Basic demographic data
ma = pd.read_sas("data/Master_ID_201908/mast_pub_12.sas7bdat")
va = ["Idind", "GENDER", "WEST_DOB_Y"]
ma = ma.loc[:, va]

# Education data
ed = pd.read_sas("data/Master_Educ_201804/educ_12.sas7bdat")
va = ["IDind", "A11", "WAVE", "COMMID"]
ed = ed.loc[:, va]

# Income data
inc = pd.read_sas("data/Master_Constructed_Income_201804/indinc_10.sas7bdat")
va = ["IDind", "wave", "indinc"]
inc = inc.loc[:, va]

# Diet data
c12 = pd.read_sas("data/c12diet.sas7bdat")
va = ["IDind", "wave", "d3kcal", "d3carbo", "d3fat", "d3protn", "t1", "t2"]
c12 = c12.loc[:, va]

# Merge all time-varying data
dy = pd.merge(inc, c12, left_on=("IDind", "wave"), right_on=("IDind", "wave"))
dy = pd.merge(dy, ed, left_on=("IDind", "wave"), right_on=("IDind", "WAVE"))

# Merge time-varying with non time-varying data
df = pd.merge(dy, ma, left_on="IDind", right_on="Idind")

# Some cleanup
df = df.rename(columns={"A11": "educ", "WEST_DOB_Y": "DOB"})
df["urban"] = df.t2.replace({1: 1, 2: 0}).values
df["age"] = df["wave"] - df["DOB"]
df["female"] = df.GENDER.replace({1: 0, 2: 1}).values
df = df.dropna()

if True:
    # Save a copy of the data
    df.to_csv("chns.csv.gz", index=False)
