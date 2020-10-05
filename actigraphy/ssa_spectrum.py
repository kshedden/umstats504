import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.linalg import toeplitz
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pdf = PdfPages("ssa_spectrum.pdf")

for tau in False, True:

    if tau:
        eiv = pd.read_csv("ssa_eig_tau.csv.gz")
    else:
        eiv = pd.read_csv("ssa_eig.csv.gz")

    eiv = eiv.set_index("SEQN")

    # Normalize
    eivn = eiv.div(eiv.sum(1), axis=0)

    plt.clf()
    plt.grid(True)
    lk = np.log10(1 + np.arange(eivn.shape[1]))
    for i in range(20):
        plt.plot(lk, np.log10(eivn.iloc[i, :]), color='grey', alpha=0.5)
    y = eivn.mean(0)
    plt.plot(lk, np.log10(y), color='purple')
    plt.xlabel(r"$\log\, k$", size=16)
    plt.ylabel(r"$\log\, \lambda_k$", size=16)
    if tau:
        plt.title("tau autocorrelation")
    else:
        plt.title("Standard autocorrelation")
    pdf.savefig()

pdf.close()
