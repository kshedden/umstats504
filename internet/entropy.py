"""
Calculate the entropy of network traffic over all ports, by minute,
for one day.

The port usage files have 60 rows and 65536 columns, with the rows
corresponding to minutes and the columns corresponding to port
numbers.  Each file corresponds to one hour of data.

There are 24 port-usage files per day, all these files are placed into
one tar archive.

The time values have Unix epoch format.  To get the human-readable
time, you can use:

import datetime
datetime.datetime.fromtimestamp(1335394800)

Zeros are dropped prior to computing the entropy.

The output file is always named entropy_minute.csv, and consists of
one entropy value per minute of traffic data.
"""

# Process data for this date
dt = "2012-04-25"

import numpy as np
import os
import pandas as pd
import tarfile
import gzip
import io

# The tar archive file name
tfn = os.path.join("results", dt, "ports.tar")

# The tar archive
tf = tarfile.open(tfn)

# Read all the port usage files for one day
df = {}
for m in tf.getnames():
    b = tf.extractfile(m).read()
    c = gzip.decompress(b)
    c = c.decode()
    x = np.genfromtxt(io.StringIO(c), delimiter=",")
    n = os.path.split(m)[-1].split(".")[0]
    df[n] = x
    print(n)

# Get the time values in order
times = list(df.keys())
times.sort()

# Caclulate the entropy, dropping 0's.
def entropy(x):
    p = x.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    return -np.sum(p * np.log(p))

# There should be one file per hour.
all_ent = []
for ti in times:

    # Loop over minutes within the hour
    mat = df[ti]
    for k in range(60):
        all_ent.append(entropy(mat[k, :]))

