# This scripts takes a reduced transit timeseries with CRIRES
# reduced the usual way by Alexis and outputs the data in a 
# pickle file, that can be loaded with the atmosferix pipeline
# developed for SPIRou and MAROON-x data

# For now, it is set to work for a precise case: data in K2148 setting
# and more particularly the GJ 3470 GTO data

# How you run:
# python3 cr2res-to-atmosferix.py
#

# What you need:
# - to be in the directory of the raw obs files CR*fits
# - to have the usual 'filelist-raw-to-reduced.ascii' in the same dir linking raw to reduced files
#   that file is output by nodding_mk-sofs.py
# - to have the obs_nodding reduced files in `pairNN` directories with NN being a serie of two digit numbers
# - to have molecfit outputs in `pairNN/molecfitA` and `pairNN/molecfitB` with products in `MODEL` `CORRECT` subdirs
# - pray to the pagan gods that it works

import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import pickle
from scipy import interpolate
from scipy.optimize import least_squares,minimize
from astropy.time import Time
from barycorrpy import utc_tdb, get_BC_vel
import warnings
import sys
warnings.filterwarnings('ignore')

def mincrosscol_wave(offset,w1,s1,wref,sref, eref):
    ii = ~np.isnan(eref)
    w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
    ii = np.where(eref != 0)
    w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
    w1n = offset + w1 
    f1 = interpolate.interp1d(w1n, s1, bounds_error=False, fill_value="extrapolate")
    s1i = f1(wref)
    return (np.sum( (s1i[20:-20]-sref[20:-20])**2./eref[20:-20]**2 ))

def mincrosscol_scale(scale,s1,sref,eref):
    ii = ~np.isnan(eref)
    s1,sref,eref = s1[ii],sref[ii],eref[ii]
    ii = np.where(eref != 0)
    s1,sref,eref = s1[ii],sref[ii],eref[ii]
    s1 = s1 * scale
    return (np.sum( (s1[100:-100]-sref[100:-100])**2./eref[100:-100]**2 ))

def gen_corr_spec(w,wref,s,params):
    scale,offset = params[0],params[1]
    w1 = offset +  w
    w = w1
    s = s * scale
    f = interpolate.interp1d(w,s, bounds_error=False, fill_value="extrapolate")
    s1 = f(wref)
    return s1

# Read the filelist linking raw to reduced files
filelist =  np.loadtxt('filelist-raw-to-reduced.ascii', skiprows=1, dtype="object")
pair,rawf,nodpos = filelist[:,0],filelist[:,1],filelist[:,2]

# Initialize the data dictionary
data = {'nodpos' : np.empty(2048)}

# Sorting raw file list by timestamp
ii = np.argsort(rawf)
pair,rawf,nodpos = pair[ii], rawf[ii], nodpos[ii]

data['nodpos'] = nodpos
data['rawfilename'] = rawf
data['nodpair'] = pair

# Creating a list of reduced files redf corresponding to the raw files
redf, modelf, corrf = [], [], []
for i in range(len(rawf)):
    redf.append(pair[i]+'/cr2res_obs_nodding_extracted'+nodpos[i]+'.normalized.fits')
    modelf.append(pair[i]+'/molecfit'+nodpos[i]+'/MODEL/BEST_FIT_MODEL.fits')
    corrf.append(pair[i]+'/molecfit'+nodpos[i]+'/CORRECT/SCIENCE_TELLURIC_CORR_SCIENCE.fits')

# Listing the orders which are present in both A and B spectra
a1,b1 = np.squeeze(np.argwhere(nodpos=="A")[0]),np.squeeze(np.argwhere(nodpos=="B")[0])
corrA,corrB = fits.open(corrf[a1]),fits.open(corrf[b1])
oA = [corrA[i+1].header['EXTNAME'] for i in range(len(corrA)-1)]
oB = [corrB[i+1].header['EXTNAME'] for i in range(len(corrB)-1)]
ordernames = np.intersect1d(oA,oB)

pp = np.arange(2008)+20

mod = fits.open(modelf[0])[1].data

nord = len(ordernames) 
nobs = len(rawf)
npix = len(pp)

w = np.zeros((nord,nobs,npix))
I = np.zeros((nord,nobs,npix))
E = np.zeros((nord,nobs,npix))
tell = np.zeros((nord,nobs,npix))
snr = np.zeros((nord,nobs))
airmass=np.zeros((nobs))
berv = np.zeros((nobs))
orders = np.arange((nord))+1
V0Vp = np.zeros((nobs))
Tobs = np.zeros((nobs))
phase = np.zeros((nobs))
window = np.zeros((nobs))

for i in range(len(rawf)):
    # Open raw file
    raw = fits.open(rawf[i])
    # Open reduced file
    red = fits.open(redf[i])
    hh = np.array(raw[0].header.cards)
    fwhm1 = np.array([value for value in np.array(red['CHIP1.INT1'].header.cards) if "FWHM" in value[0]])
    fwhm2 = np.array([value for value in np.array(red['CHIP2.INT1'].header.cards) if "FWHM" in value[0]])
    fwhm3 = np.array([value for value in np.array(red['CHIP3.INT1'].header.cards) if "FWHM" in value[0]])
    if i == 0:
        hhs = hh.shape
        data['rawheaders'] = np.empty([nobs,hhs[0],hhs[1]], dtype="object")
        data['slitfunctionFWHM-det1'] = np.empty([nobs,fwhm1.shape[0],fwhm1.shape[1]], dtype="object")
        data['slitfunctionFWHM-det2'] = np.empty([nobs,fwhm2.shape[0],fwhm2.shape[1]], dtype="object")
        data['slitfunctionFWHM-det3'] = np.empty([nobs,fwhm3.shape[0],fwhm3.shape[1]], dtype="object")
    data['rawheaders'][i,:,:] = hh
    data['slitfunctionFWHM-det1'][i,:,:], data['slitfunctionFWHM-det2'][i,:,:],data['slitfunctionFWHM-det3'][i,:,:] = fwhm1, fwhm2, fwhm3
    # Open molecfit MODEL resuts
    mod = fits.open(modelf[i])[1].data
    cor = fits.open(corrf[i])
    airmass[i] = np.mean([raw[0].header['HIERARCH ESO TEL AIRM START'],raw[0].header['HIERARCH ESO TEL AIRM END']])
    print(rawf[i])
    # loop over orders
    index = 0
    for o in range(len(cor)-1):
        if len(np.intersect1d(ordernames,cor[o+1].header['EXTNAME'])) ==0 :
            continue
        ii = np.squeeze(np.argwhere(mod['chip'] == o+1)) #not-telluric corrected case
        w[index,i,:] = mod['mlambda'][ii][pp] * 1000. # not-tell
        if index > 0 :
#        if nodpos[i] == 'B':
            # Cross-correlate mean up on mean down
            flux = mod['flux'][ii][pp] # not-telluric corrected spectrum
            res = least_squares(mincrosscol_scale, 1., args=(flux, I[index,0,:], flux*0.+1.), bounds=(0.,5.), verbose=0, max_nfev=2500)
            scale_AB = np.copy(res.x)
            res = least_squares(mincrosscol_wave, 0., args=(w[index,i,:], flux*scale_AB, 
                w[index,0,:], I[index,0,:],I[index,0,:]*0+1), bounds=(-1,1), verbose=0, max_nfev=2500)
            wave_offset = np.copy(res.x)
            # Produce interpolated up spectra
            u1i = gen_corr_spec(w[index,i,:], w[index,0,:], flux ,[1.,wave_offset])
            telli = gen_corr_spec(w[index,i,:], w[index,0,:], mod['mflux'][ii][pp] ,[1.,wave_offset])
            I[index,i,:] = u1i
            tell[index,i,:] = telli
        else: 
            I[index,i,:] = mod['flux'][ii][pp]
            tell[index,i,:] = mod['mflux'][ii][pp]
        w[index,i,:] = w[index,0,:]
        snr[index,i] = np.nanmedian(cor[index+1].data['SPEC'][pp]/cor[index+1].data['ERR'][pp])
        E[index,i,:] = cor[index+1].data['ERR'][pp] 
        index+=1

    # Compute the UTC_TDB obs date
    # Retrieve MJD from header and convert to JD
    # Add half of exptime to shift from exp start to mid exp
    half_exptime = raw[0].header['HIERARCH ESO DET SEQ1 DIT'] * raw[0].header['HIERARCH ESO DET NDIT'] * raw[0].header['HIERARCH ESO SEQ NEXPO'] / (2. * 86400.)
    # Convert half_exptime from seconds to day
    jd = Time(2400000.5 + half_exptime + raw[0].header['MJD-OBS'], format='jd', scale='utc')
     # Convert JD_UTC to UTC_TDB
    utc_tbd = utc_tdb.JDUTC_to_BJDTDB(jd, starname=raw[0].header['OBJECT'],lat=raw[0].header['HIERARCH ESO TEL GEOLAT'],
                                longi=raw[0].header['HIERARCH ESO TEL GEOLON'], alt=raw[0].header['HIERARCH ESO TEL GEOELEV'])
    Tobs[i] = np.squeeze(utc_tbd[0])

    # Compute BERV
    bcvel = get_BC_vel(JDUTC=jd, starname=raw[0].header['OBJECT'], lat=raw[0].header['HIERARCH ESO TEL GEOLAT'], 
                                longi=raw[0].header['HIERARCH ESO TEL GEOLON'], alt=raw[0].header['HIERARCH ESO TEL GEOELEV'], zmeas=0.0)
    berv[i] = np.squeeze(bcvel[0])
    # end of loop on files ##########################
blaze=I

wii = np.argsort(w[:,0,0])

w = w[wii,:,:]
I = I[wii,:,:]
snr = snr[wii,:]

data['orders'] = orders
data['wavelength'] = w
data['intensity'] = I
data['tellurics'] = tell
data['utc_tbd'] = Tobs
data['berv'] = berv
data['airmass'] = airmass
data['snr'] = snr
data['error'] = E
#savedata = (orders,w,I,blaze,tell,Tobs,phase,window,berv,V0Vp,airmass,snr)
with open(sys.argv[1], 'wb') as specfile:
    pickle.dump(data,specfile)
