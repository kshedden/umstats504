"""
Download the individual-level data for the Russia Longitudinal Monitoring Survey:

https://dataverse.unc.edu/dataset.xhtml?persistentId=doi:10.15139/S3/12438

There are data files and codebook files, you will want one of each.
"""

import numpy as np
import pandas as pd

# Variables to keep
va = [
        "idind", # individual ID
        "J1", # current work status (1 = working)
        "age", # current age
        "year", # current year
        "J69_9C", # year of birth
        "educ", # education
        "status", # area type (city, etc.)
        "J8", # hours worked last 30 days
        "J10", # after tax wages
        "H5", # gender 1 = male, 2 = female
        "psu", # survey PSU and geographic area
        "OCCUP08", # occupation code
     ]

df = pd.read_csv("RLMS-HSE_IND_1994_2018_STATA.tab.gz", sep="\t", usecols=va)

# Drop people who are not working
df = df.loc[df.J1==1, :]

# Recode gender
df["Female"] = (df.H5 == 2).astype(np.int)

# Drop the original versions of variables that we no longer need.
df = df.drop(columns=["J1", "H5"])

# Drop rows with missing values
dx = df.dropna()

# Center year at 2000, making it more centered
dx.year -= 2000

# Remove special codes
for v in "J10", "J8", "educ", "age":
    ii = dx.loc[:, v] < 99999997
    dx = dx.loc[ii, :]
