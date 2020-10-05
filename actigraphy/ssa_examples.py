import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import acf
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

ntime = 24*60*7
nseries = 10
nlags = 24*60

gen_examples = []

gen_examples.append(("White noise", lambda : np.random.normal(size=(ntime, nseries))))

def f():
    x = np.random.normal(size=(ntime, nseries))
    x += np.outer(np.linspace(-2, 2, ntime), np.random.normal(size=nseries))
    return x
gen_examples.append(("Random lines", f))

def f():
    r = 0.99
    x = np.random.normal(size=(ntime, nseries))
    for k in range(1, ntime):
        x[k, :] = r*x[k-1, :] + np.sqrt(1-r**2)*x[k, :]
    return x
gen_examples.append(("AR 0.99", f))

def f():
    r = 0.99
    x = np.random.normal(size=(ntime, nseries))
    f = np.linspace(-100, 100, 24*60)
    f = 1 / (1 + np.abs(f))
    f /= np.sqrt(np.sum(f*f))
    z = np.apply_along_axis(lambda x: np.convolve(x, f, mode='same'), axis=0, arr=x)
    return z
gen_examples.append(("Convolution", f))

def f():
    t = np.sin(np.linspace(0, 2*np.pi*7, ntime))
    t += np.sin(np.linspace(0, 2*np.pi*14, ntime))
    x = t[:, None] + np.random.normal(size=(ntime, nseries))
    return x
gen_examples.append(("Sinusoid", f))

pdf = PdfPages("ssa_examples.pdf")

lk = np.log10(1 + np.arange(nlags + 1))

for k in range(len(gen_examples)):

    da = gen_examples[k][1]()

    for tau in False, True:

        plt.clf()
        plt.grid(True)
        plt.title(gen_examples[k][0])

        for j in range(nseries):

            x = da[:, j]

            if tau:
                a = np.zeros(nlags+1)
                a[0] = 1
                for l in range(1, nlags+1):
                    a[l] = kendalltau(x[0:-l], x[l:]).correlation
            else:
                a = acf(x, nlags=nlags, fft=True)

            c = toeplitz(a)
            eiv, eig = np.linalg.eigh(c)
            ii = np.argsort(-eiv)
            eiv = eiv[ii]
            eig = eig[:, ii]

            # Normalize
            eivn = eiv / eiv.sum()

            plt.plot(lk, np.log10(eivn))

        plt.ylim(-8, 0)
        plt.xlabel(r"$\log\, k$", size=16)
        plt.ylabel(r"$\log\, \lambda_k$", size=16)
        pdf.savefig()

pdf.close()
