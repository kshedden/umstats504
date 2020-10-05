import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.stats import kendalltau
from scipy.linalg import toeplitz, hankel
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

rdr = pd.read_csv("paxraw.csv.gz", error_bad_lines=False, chunksize=24*60*7)

# Number of autocorrelation lags to compute
nlags = 60*24

pdf = PdfPages("ssa_components.pdf")

results = []

def process(df, tau):

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

    y = df.PAXINTEN.values
    y = y - y.mean()

    # Get the component corresponding to each
    # singular value.
    xm = hankel(y)[:, 0:nlags+1]
    s = np.sqrt(eiv)
    u = np.dot(xm, eig) / s

    if not tau:
        plt.clf()
        plt.grid(True)
        plt.plot(y)
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.title(df.SEQN.iloc[0])
        pdf.savefig()

    # Plot the components
    plt.clf()
    plt.grid(True)
    for j in 0, 1, 2:
        plt.plot(u[:, j], label=str(j+1))
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
    if tau:
        plt.title(str(df.SEQN.iloc[0]) + " (tau)")
    else:
        plt.title(df.SEQN.iloc[0])
    plt.xlabel("Time")
    plt.ylabel("Loading")
    pdf.savefig()

    # Plot the coefficients
    plt.clf()
    plt.grid(True)
    for j in 0, 1, 2:
        plt.plot(eig[:, j], label=str(j+1))
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
    if tau:
        plt.title(str(df.SEQN.iloc[0]) + " (tau)")
    else:
        plt.title(df.SEQN.iloc[0])
    plt.xlabel("Time")
    plt.ylabel("Coefficient")
    pdf.savefig()

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

    for tau in False, True:
        process(z, tau)

    nsubject += 1
    if nsubject % 10 == 0:
        print(nsubject)

    if nsubject > 20:
        break

pdf.close()