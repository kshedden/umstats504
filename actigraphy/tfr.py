"""
Time/frequency representation of actigraphy data.
"""

import pandas as pd
import numpy as np
from scipy.signal import stft
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

rdr = pd.read_csv("paxraw.csv.gz", error_bad_lines=False, chunksize=24*60*7)

pdf = PdfPages("tfr.pdf")

def process(y):
    w = stft(y, nperseg=60*24)

    plt.clf()
    plt.grid(True)
    plt.plot(y)
    plt.xlabel("Time")
    plt.ylabel("Intensity")
    pdf.savefig()

    plt.clf()
    f, t, zxx = stft(y)
    plt.pcolormesh(t, f, np.abs(zxx), vmin=0)
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    pdf.savefig()

# Some toy examples to illustrate how the method works

y = np.sin(np.linspace(0, 2*np.pi*7, 24*60*7))
process(y)

y = np.sin(np.linspace(0, 2*np.pi*70, 24*60*7))
process(y)

y = np.sin(np.linspace(0, 2*np.pi*700, 24*60*7))
process(y)

y = np.sin(np.linspace(0, 2*np.pi*7, 24*60*7))
y += np.sin(np.linspace(0, 2*np.pi*700, 24*60*7))
process(y)

x = np.linspace(0, 2*np.pi*7, 24*60*7)
y = np.sin(x + 200*np.cos(x))
process(y)

# Make TFR plots for a few of the actual time series.

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
    process(z.PAXINTEN)

    nsubject += 1
    if nsubject > 10:
        break

pdf.close()