#!/usr/bin/env python3
# coding: utf-8

# How to use:
# Within ipython:
# %run cr2res_plot_nodding.py extracted_spectrum.fits1 color scaling  [this line can be repeated multiple times to plot several files]
# (optional) %run cr2res_plot_nodding.py extracted_spectrum.fits2 color scaling
# plt.show()

# where "extracted_spectrum.fits" is the DRS output file with extracted spectrum
# "color" is a string specifying the color of the line plot (can be "k", "C0", "red" etc ...)
# "scaling" is a scaling factor to multiply the spectrum (use "1." if you don't want to scale the spectrum )

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

f, col, scale = sys.argv[1], sys.argv[2], np.float(sys.argv[3])
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = [12, 6]
o = fits.open(f)

w,s= [],[]
chip = ['CHIP1.INT1', 'CHIP2.INT1', 'CHIP3.INT1']
d = [o[chip[0]].data,o[chip[1]].data,o[chip[2]].data]
for i in range(3):
	cpw = np.sort([i for i in d[i].dtype.names if "WL" in i])
	cpi = np.sort([i for i in d[i].dtype.names if "SPEC" in i])
	for j in range(len(cpw)):
		plt.plot(d[i][cpw[j]][8:-8], d[i][cpi[j]][8:-8]*scale, linewidth=1, color=col)
		w,s = np.append(w,d[i][cpw[j]]),np.append(s,d[i][cpi[j]])
plt.ylim(-50, np.percentile(1.4*np.nan_to_num(s),98))
plt.xlabel("Wavelength (nm)")
plt.ylabel('Extracted spectrum')
