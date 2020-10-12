"""
Append all the single-hour traffic statistics summary files to get a
traffic statistics file for one day.
"""

import numpy as np
import os
import pandas as pd
from datetime import date

# Process these days
days = (1, 5, 25)


def process_day_stats(day):

    bdate = date(2012, 4, day)

    base = "results/%s" % bdate.isoformat()
    files = os.listdir(base)
    files = [x for x in files if x.endswith(".csv")]

    def f(x):
        y = x.split(".")[0]
        return str.isdigit(y) and len(y) == 10

    files = [x for x in files if f(x)]

    # Make sure the files are in temporal order
    d = [int(x.split(".")[0]) for x in files]
    ii = np.argsort(d)
    files = [files[i] for i in ii]

    adf = []
    for f in files:
        df = pd.read_csv(os.path.join("results", bdate.isoformat(), f))
        df = df.iloc[:, 1:]
        adf.append(df)
    dx = pd.concat(adf, axis=0)
    dx.columns = ["Traffic", "Sources", "UDP", "TCP"]

    dx["Minute"] = np.arange(dx.shape[0])
    dx["Hour"] = np.floor(dx.Minute / 60)
    dx.Minute = dx.Minute % 60
    dx.Minute = dx.Minute.astype(np.int)
    dx.Hour = dx.Hour.astype(np.int)

    dx = dx[["Hour", "Minute", "Traffic", "Sources", "UDP", "TCP"]]

    dx.to_csv(os.path.join(base, "traffic_stats.csv"), index=None)

for day in days:
    process_day_stats(day)
    bdate = date(2012, 4, day)
    base = "results/%s" % bdate.isoformat()
    d = bdate.isoformat()
    target = os.path.join(base, "ports.tar")
    os.system("tar -cvf %s results/%s/*.dports.csv.gz" % (target, d ))
