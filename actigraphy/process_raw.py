"""
This script processes the NHANES actigraphy data into a
csv file.  Note that the resulting file will be too large
to read into memory on most computers.

Documentation:
https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/PAXRAW_C.htm

Get the data:
wget https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/PAXRAW_C.ZIP
"""

import pandas as pd
import gzip
import numpy as np

rdr = pd.read_sas("paxraw_c.xpt", format="xport", chunksize=10000000)

out = gzip.open("paxraw.csv.gz", "wt")
first = True

jj = 0
while True:

    try:
        df = next(rdr)
    except StopIteration:
        break

    df = df.astype(np.int)

    out.write(df.to_csv(header=first, index=False))
    first = False

    jj += df.shape[0]
    print(rdr.nobs - jj)
    print(df.shape)

out.close()
