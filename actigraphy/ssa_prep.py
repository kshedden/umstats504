"""
Compute the eigenvalues from a Singular Spectrum Analysis (SSA) of the NHANES
actigraphy data, and save them to a file.

Reference:

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181762
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.stats import kendalltau
from scipy.linalg import toeplitz

rdr = pd.read_csv("paxraw.csv.gz", error_bad_lines=False, chunksize=24*60*7)

# If true, use Kandall's-tau, else use standard autocorrelation.
# Calculating the tau correlations for all data takes several
# hours, standard autocorrelations are much faster.
tau = False

# Number of autocorrelation lags to compute
nlags = 60*24

results = []

def process(df):

    if tau:
        x = df.PAXINTEN
        a = np.zeros(nlags+1)
        a[0] = 1
        for l in range(1, nlags+1):
            a[l] = kendalltau(x[0:-l], x[l:]).correlation
    else:
        a = acf(df.PAXINTEN, nlags=nlags, fft=True)

    if np.isnan(a).any():
        eiv = np.nan*np.ones(len(a))
    else:
        c = toeplitz(a)
        eiv, eig = np.linalg.eigh(c)
        ii = np.argsort(-eiv)
        eiv = eiv[ii]
        eig = eig[:, ii]

    eiv = np.concatenate(([df.SEQN.iloc[0]], eiv))
    results.append(eiv)


chunks = []
nsubject = 0
while True:

    try:
        df1 = rdr.get_chunk()
    except StopIteration:
        break

    for idx, v in df1.groupby("SEQN"):
        chunks.append((idx, v))

    ids = [x[0] for x in chunks]

    if len(set(ids)) > 1:
        # z contains complete data for one subject and is
        # ready to process
        z = [x[1] for x in chunks if x[0] == ids[0]]
        chunks = [x for x in chunks if x[0] != ids[0]]
    else:
        continue

    z = pd.concat(z, axis=0)
    process(z)

    nsubject += 1
    if nsubject % 10 == 0:
        print(nsubject)

results = pd.DataFrame.from_records(results)
results.columns = ["SEQN"] + ["eig%03d" % k for k in range(nlags+1)]
results["SEQN"] = results["SEQN"].astype(np.int)
if tau:
    results.to_csv("ssa_eig_tau.csv.gz", index=None)
else:
    results.to_csv("ssa_eig.csv.gz", index=None)
