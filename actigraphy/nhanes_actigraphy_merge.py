"""
Merge actigraphy data with other NHANES data.

The actigraphy data are from the 2003-2004 wave of NHANES.

Below are some of the other NHANES files from the same wave that
we can merge with the actigraphy data, using the SEQN
subject identifier.

https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/DEMO_C.XPT
https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/BPX_C.XPT
"""

import pandas as pd

# Actigraphy data
act = pd.read_csv("ssa_eig.csv.gz")

# Demographic data
demo = pd.read_sas("DEMO_C.XPT")
demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]

# Blood pressure data
bp = pd.read_sas("BPX_C.XPT")
bp = bp[["SEQN", "BPXSY1", "BPXSY2"]]

df = pd.merge(demo, act, left_on="SEQN", right_on="SEQN")
df = pd.merge(df, bp, left_on="SEQN", right_on="SEQN")

df.to_csv("nhanes_merged.csv.gz", index=None)
