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

def mincrosscol(params, w1,s1,wref,sref,eref):
    scale, offset = params[0], params[1]
    ii = ~np.isnan(eref)
    w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
    ii = np.where(eref != 0)
    w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
    w1n = offset + w1 
    f1 = interpolate.interp1d(w1n, s1, bounds_error=False, fill_value="extrapolate")
    s1i = f1(wref) * scale
    return  (s1i[20:-20]-sref[20:-20])**2./eref[20:-20]**2

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


redf = [] # Initializing the array of reduced filenames

# Loop on raw files, find the reduced files
for i in range(len(rawf)):
    redf.append(pair[i]+'/molecfit'+nodpos[i]+'/SCIENCE.fits')

# Open the first A and B:
i0A, i0B = np.where(nodpos=='A')[0][0], np.where(nodpos=='B')[0][0]
red0A, red0B = fits.open(redf[i0A]), fits.open(redf[i0B])
# Listing the orders which are present in both A and B spectra
oA = [red0A[i+1].header['EXTNAME'] for i in range(len(red0A)-1)]
oB = [red0B[i+1].header['EXTNAME'] for i in range(len(red0B)-1)]
ordernames = np.intersect1d(oA,oB)

pp = np.arange(2008)+20
nord = len(ordernames) 
nobs = len(rawf)
npix = len(pp)

data['wave'] = np.zeros((nord,nobs,npix))
data['spec'] = np.zeros((nord,nobs,npix))
data['err'] = np.zeros((nord,nobs,npix))
data['wave_xcorr'] = np.zeros((nord,nobs,npix))
data['spec_xcorr'] = np.zeros((nord,nobs,npix))

data['snr'] = np.zeros((nord,nobs))
data['airmass'] = np.zeros((nobs))
data['utc_tbd'] = np.zeros((nobs))
data['berv'] = np.zeros((nobs))
data['orders'] = np.arange((nord))+1

masterA = fits.open('master/molecfitA/SCIENCE.fits')
# MAIN LOOP
for i in range(len(rawf)):
    raw = fits.open(rawf[i]) # open raw file
    red = fits.open(redf[i]) # open reduce file

    hh = np.array(raw[0].header.cards) # store raw header
    # Store FWHM of PSF from reduced file, for the 3 det
    ##fwhm1 = np.array([value for value in np.array(red['CHIP1.INT1'].header.cards) if "FWHM" in value[0]])
    ##fwhm2 = np.array([value for value in np.array(red['CHIP2.INT1'].header.cards) if "FWHM" in value[0]])
    ##fwhm3 = np.array([value for value in np.array(red['CHIP3.INT1'].header.cards) if "FWHM" in value[0]])
    if i == 0:
        hhs = hh.shape
        data['rawheaders'] = np.empty([nobs,hhs[0],hhs[1]], dtype="object")
       ## data['slitfunctionFWHM-det1'] = np.empty([nobs,fwhm1.shape[0],fwhm1.shape[1]], dtype="object")
       ## data['slitfunctionFWHM-det2'] = np.empty([nobs,fwhm2.shape[0],fwhm2.shape[1]], dtype="object")
      ## data['slitfunctionFWHM-det3'] = np.empty([nobs,fwhm3.shape[0],fwhm3.shape[1]], dtype="object")
    data['rawheaders'][i,:,:] = hh # store raw header in dictionary
    ##data['slitfunctionFWHM-det1'][i,:,:], data['slitfunctionFWHM-det2'][i,:,:],data['slitfunctionFWHM-det3'][i,:,:] = fwhm1, fwhm2, fwhm3
    data['airmass'][i] = np.mean([raw[0].header['HIERARCH ESO TEL AIRM START'],raw[0].header['HIERARCH ESO TEL AIRM END']])
    print(rawf[i])
    # loop over orders
    index = 0
    for o in range(len(red)-1):
        if len(np.intersect1d(ordernames,red[o+1].header['EXTNAME'])) ==0 :
            continue
        wave = red[o+1].data['WAVE'][pp]
        flux = red[o+1].data['SPEC'][pp]
        error = red[o+1].data['ERR'][pp]
        waveM = masterA[o+1].data['WAVE'][pp]
        refflux = masterA[o+1].data['SPEC'][pp]
        # Cross-correlate mean up on mean down
        res = least_squares(mincrosscol, [1,0.], args=(wave, flux, waveM, refflux, error),bounds=([0.1,-0.1],[10,0.1]), x_scale=[0.1,0.0002],gtol=1e-12,ftol=1e-12, xtol=1e-12, max_nfev=10000)
        fluxinterp = gen_corr_spec(wave, waveM, flux ,[1.,res.x[1]])
        data['wave'][index,i,:], data['wave_xcorr'][index,i,:] = wave, waveM 
        data['spec'][index,i,:], data['spec_xcorr'][index,i,:] = flux, fluxinterp
        data['snr'][index,i] = np.nanmedian(flux/error)
        data['err'][index,i,:] = error
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
    data['utc_tbd'] = np.squeeze(utc_tbd[0])

    # Compute BERV
    bcvel = get_BC_vel(JDUTC=jd, starname=raw[0].header['OBJECT'], lat=raw[0].header['HIERARCH ESO TEL GEOLAT'], 
                                longi=raw[0].header['HIERARCH ESO TEL GEOLON'], alt=raw[0].header['HIERARCH ESO TEL GEOELEV'], zmeas=0.0)
    data['berv'][i] = np.squeeze(bcvel[0])
    # end of loop on files ##########################

wii = np.argsort(data['wave'][:,0,0])

data['wave'] = data['wave'][wii,:,:]
data['wave_xcorr'] = data['wave_xcorr'][wii,:,:]
data['spec'] = data['spec'][wii,:,:]
data['spec_xcorr'] = data['spec_xcorr'][wii,:,:]
data['err'] = data['err'][wii,:,:]
data['snr'] = data['snr'][wii,:]


with open(sys.argv[1], 'wb') as specfile:
    pickle.dump(data,specfile)
