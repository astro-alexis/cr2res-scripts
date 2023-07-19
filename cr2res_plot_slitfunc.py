#!/usr/bin/env python3
# coding: utf-8

# How to use:
# python3 cr2res_plot_slitfunc.py slit_func.fits oversampling
# where 
# - slit_func.fits is a slit_func file as output by DRS recipe
# - oversampling is the value of the --oversample or --extract_oversample keyword
# used in the reduction

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

f, osamp = sys.argv[1], sys.argv[2]

plt.rcParams['font.size'] = 12
o = fits.open(f)
fig,ax = plt.subplots(1,3,figsize=(11,4))
chip = ['CHIP1.INT1', 'CHIP2.INT1', 'CHIP3.INT1']
for i in range(3):
	data = o[chip[i]].data
	cp = np.sort([order for order in data.dtype.names if "SLIT_FUNC" in order])
	pix = np.arange(len(data[cp[0]]))/np.float(osamp)
	for j in range(len(cp)):
		ax[i].plot(pix,data[cp[j]], linewidth=1, label=cp[j])
	ax[i].legend()
	ax[i].set_xlabel("Pixels")

ax[0].set_ylabel('Slit function')
plt.show()
