# ## Covid case/mortality analysis for 173 countries
#
# We use data from the European Center for Disease Prevention and Control, that can be found
# at this site:
#
# https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
#
# Download the csv data file and save it in the working directory with the file name "ecdpc.csv", then
# compress it.
#
# A direct link to the data is here:
#
# https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

import numpy as np
import statsmodels.api as sm
import pandas as pd
from datetime import datetime

# Load the data.

df = pd.read_csv("ecdpc.csv.gz")

# Simplify some variable names.

df["country"] = df.countryterritoryCode
df = df.loc[pd.notnull(df.country), :]

# Clean up the dates.

def f(x):
    u = x.split("/")
    return "%s-%s-%s" % tuple(u[::-1])
df["date_rep"] = df.dateRep.apply(f)
df["date"] = pd.to_datetime(df.date_rep)

# Remove meaningless rows with future dates.

today = datetime.today().strftime('%Y-%m-%d')
today = pd.to_datetime(today)
df = df.loc[df.date <= today]

# Only keep countries having at least one month of data

n = df.groupby("country").size()
n = n[n > 30]
df = df.loc[df.country.isin(n.index), :]

# Sort first by country, then within country by date.

df = df.sort_values(["country", "date"])

# Days is the number of calendar days in 2020 for each record.

df["days"] = (df.date - pd.to_datetime("2020-01-01")).dt.days

# Create variables containing the number of new cases
# within a week, for each of the four weeks preceding
# the current day.

# Sum x from d2 days back in time to d1 days back in time, inclusive of
# both endpoints.  d2 must be greater than d1.

def wsum(x, d1, d2):
    w = np.ones(d2 + 1)
    if d1 > 0:
        w[-d1:] = 0
    y = np.zeros_like(x)
    y[d2:] = np.convolve(x.values, w[::-1], mode='valid')
    return y

for j in range(4):
    xx = df.groupby("country").cases.transform(lambda x: wsum(x, 7*j, 7*j+6))
    df["cumpos%d" % j] = df.groupby("country").cases.transform(lambda x: wsum(x, 7*j, 7*j+6))
    df["logcumpos%d" % j] = np.log(df["cumpos%d" % j] + 1)

# Calculate the date of the first death in each country,
# then remove data prior to 10 days after this date.
# rdays is the number of days in each country since
# its first reported Covid death.

def firstdeath(x):
    if (x.deaths == 0).all():
        return pd.to_datetime("2100-1-1")
    ii = np.flatnonzero(x.deaths > 0)[0]
    return x.date.iloc[ii]

xx = df.groupby("country").apply(firstdeath)
xx.name = "firstdeath"
df = pd.merge(df, xx, left_on="country", right_index=True)

df["rdays"] = (df.date - df.firstdeath).dt.days
df = df.loc[df.rdays >= 10, :]

df.to_csv("testing_mortality_world.csv", index=None)
