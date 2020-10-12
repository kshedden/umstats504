"""
Calculate the entropy of network traffic over all ports, by minute.

All files in the 'results' directory named TTT.dports.csv.gz are processed,
where TTT is a time index. These files should have 65536 columns, corresponding
to the possible port numbers.

Each row of an input file is a one-minute time block.

Zeros are dropped prior to computing the entropy.

The output file is always named entropy_minute.csv, and consists of one entropy
value per minute of traffic data.
"""

import numpy as np
import os
import pandas as pd

# Process data for this date
dt = "2012-04-25"
base = os.path.join("results", dt)

dport_files = os.listdir(base)
dport_files = [x for x in dport_files if x.endswith(".dports.csv.gz")]

# Make sure the files are in temporal order
d = [int(x.split(".")[0]) for x in dport_files]
ii = np.argsort(d)
dport_files = [dport_files[i] for i in ii]

# Caclulate the entropy, dropping 0's.
def entropy(x):
    p = x.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    return -np.sum(p * np.log(p))

# There should be one file per hour.
all_ent = []
for f in dport_files:

    pa = os.path.join("results/" + dt, f)
    mat = np.genfromtxt(pa, delimiter=",")

    # Loop over minutes within the hour
    for k in range(60):
        all_ent.append(entropy(mat[k, :]))

fid = open(os.path.join(base, "entropy_minute.csv"), "w")
for x in all_ent:
    fid.write("%f\n" % x)
fid.close()
