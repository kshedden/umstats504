# Obtain the ICD 10 mortality data files from this site:
#
# https://www.who.int/healthinfo/statistics/mortality_rawdata/en/
#
# There are two mortality files to download, called Morticd10_part1 and
# Morticd10_part2.
#
# You also need to download the "Population and live births" file.
#
# Place these files in the current working directory, compress
# them with gzip, and run this script.

import pandas as pd
import gzip
import csv
import numpy as np

def prep_deaths(fn):

    df = pd.read_csv(fn)

    # Select only ICD10
    df = df.loc[df.List == 104, :]
    df = df.loc[df.Cause == "AAA", :]

    id_vars = df.columns[0:9]
    dx = df.melt(id_vars=id_vars, var_name="Age", value_name="Deaths")
    dx = dx.drop(["Admin1", "SubDiv", "Frmat", "IM_Frmat", "Cause", "List"], axis=1)
    dx = dx.loc[[x.startswith("Deaths") for x in dx.Age], :]
    dx["Age"] = [x.replace("Deaths", "") for x in dx.Age]
    dx["Age"] = pd.to_numeric(dx["Age"])

    return dx

def prep_pop():

    df = pd.read_csv("pop.gz")
    id_vars = df.columns[0:6]
    dx = df.melt(id_vars=id_vars, var_name="Age", value_name="Pop")
    dx = dx.drop(["Admin1", "SubDiv", "Frmat"], axis=1)
    dx = dx.loc[[x.startswith("Pop") for x in dx.Age], :]
    dx["Age"] = [x.replace("Pop", "") for x in dx.Age]
    dx["Age"] = pd.to_numeric(dx["Age"])
    dx = dx.drop_duplicates(["Country", "Year", "Sex", "Age"])

    return dx

dt1 = prep_deaths("Morticd10_part1.gz")
dt2 = prep_deaths("Morticd10_part2.gz")
dt = pd.concat((dt1, dt2), axis=0)

pop = prep_pop()

mv = ["Country", "Year", "Sex", "Age"]
df = pd.merge(dt, pop, left_on=mv, right_on=mv, how="left")
df = df.dropna()
df = df.loc[df.Pop > 0, :]

df.to_csv("who_allcause.csv.gz", index=None)
