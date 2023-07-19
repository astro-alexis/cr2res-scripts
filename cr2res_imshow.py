#!/usr/bin/env python3
# coding: utf-8

# How tu use: 
# python3 cr2res_imshow file.fits [bpm.fits]

# What it does
# Display (using imshow) the CRIRES+ frame file.fits. If a second argument is given, it is assumed to be a
# a bad pixel mask and it is applied to the first frame 

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

f = sys.argv[1]
if len(sys.argv) > 2:
    bpm = fits.open(sys.argv[2])
    use_bpm = True
else:
    use_bpm = False

plt.rcParams['figure.figsize'] = [12, 6]
o = fits.open(f)
chip = ['CHIP1.INT1', 'CHIP2.INT1', 'CHIP3.INT1']
d = [o[chip[0]].data,o[chip[1]].data,o[chip[2]].data]
vmax = np.max([np.percentile(1.3*np.nan_to_num(d[0]),98), np.percentile(1.3*np.nan_to_num(d[1]),98), np.percentile(1.3*np.nan_to_num(d[2]),98)])

fig,ax = plt.subplots(1,3)
for i in range(3):
	if use_bpm == True: d[i][np.where(bpm[i+1].data>0)] = 0
	ax[i].imshow(np.nan_to_num(d[i]), origin="lower", vmin=0, vmax=vmax)
plt.show()
