# This scripts takes a reduced transit timeseries with CRIRES
# reduced the usual way by Alexis and outputs the data in a 
# pickle file,

# A documentation is available at : https://box.in2p3.fr/index.php/s/pCJKtJbzzD4NtaM

# How you run:
# python3 cr2res-to-atmos output.pickle

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

data = {'script_version' : '2.2'}

# Read the filelist linking raw to reduced files
filelist =  np.loadtxt('filelist-raw-to-reduced.ascii', skiprows=1, dtype="object")
pair,rawf,nodpos = filelist[:,0],filelist[:,1],filelist[:,2]

# Sorting raw file list by timestamp
ii = np.argsort(rawf)
pair,rawf,nodpos = pair[ii], rawf[ii], nodpos[ii]

data['nodpos'] = nodpos
data['rawfilename'] = rawf
data['nodpair'] = pair

redf = [] # Initializing the array of reduced filenames

# Loop on raw files, find the reduced files
for i in range(len(rawf)):
    redf.append(pair[i]+'/cr2res_obs_nodding_extracted'+nodpos[i]+'.normalized.fits')
    

# Open the first A and B:
i0A, i0B = np.where(nodpos=='A')[0][0], np.where(nodpos=='B')[0][0]
red0A, red0B = fits.open(redf[i0A]), fits.open(redf[i0B])

# Listing the orders which are present in both A and B spectra
roA1 = ["1-"+ord[0:2] for ord in red0A['CHIP1.INT1'].data.names if 'SPEC' in ord]
roA2 = ["2-"+ord[0:2] for ord in red0A['CHIP2.INT1'].data.names if 'SPEC' in ord]
roA3 = ["3-"+ord[0:2] for ord in red0A['CHIP3.INT1'].data.names if 'SPEC' in ord]
oA = np.append(roA1,roA2)
oA = np.append(oA,roA3)

roB1 = ["1-"+ord[0:2] for ord in red0B['CHIP1.INT1'].data.names if 'SPEC' in ord]
roB2 = ["2-"+ord[0:2] for ord in red0B['CHIP2.INT1'].data.names if 'SPEC' in ord]
roB3 = ["3-"+ord[0:2] for ord in red0B['CHIP3.INT1'].data.names if 'SPEC' in ord]
oB = np.append(roB1,roB2)
oB = np.append(oB,roB3)
ordernames = np.intersect1d(oA,oB)

pp = np.arange(2008)+20
nord = len(ordernames) 
nobs = len(rawf)
npix = len(pp)

# Initializing most keys of the data dictionary
data['wave'] = np.zeros((nord,nobs,npix))
data['wave_model'] = np.zeros((nord,nobs,npix))
data['spec'] = np.zeros((nord,nobs,npix))
data['err'] = np.zeros((nord,nobs,npix))
data['snr'] = np.zeros((nord,nobs))
data['airmass'] = np.zeros((nobs))
data['bjd_tdb'] = np.zeros((nobs))
data['berv'] = np.zeros((nobs))
data['orders'] = np.empty((nord), dtype="object")

# MAIN LOOP
for i in range(len(rawf)):
    raw = fits.open(rawf[i]) # open raw file
    red = fits.open(redf[i]) # open reduce file
    # Open molecfit model for MASTER A/B spectra
    if nodpos[i] == 'A': master = fits.open('master/molecfitA/MODEL/BEST_FIT_MODEL.fits')[1].data
    else: master = fits.open('master/molecfitB/MODEL/BEST_FIT_MODEL.fits')[1].data
    hh = np.array(raw[0].header.cards) # store raw header
    # Store FWHM of PSF from reduced file, for the 3 det
    fwhm1 = np.array([value for value in np.array(red['CHIP1.INT1'].header.cards) if "FWHM" in value[0]])
    fwhm2 = np.array([value for value in np.array(red['CHIP2.INT1'].header.cards) if "FWHM" in value[0]])
    fwhm3 = np.array([value for value in np.array(red['CHIP3.INT1'].header.cards) if "FWHM" in value[0]])

    if i == 0:
        hhs = hh.shape
        data['rawheaders'] = np.empty([nobs,hhs[0],hhs[1]], dtype="object")
        data['slitfunctionFWHM-det1'] = np.empty([nobs,fwhm1.shape[0],fwhm1.shape[1]], dtype="object")
        data['slitfunctionFWHM-det2'] = np.empty([nobs,fwhm2.shape[0],fwhm2.shape[1]], dtype="object")
        data['slitfunctionFWHM-det3'] = np.empty([nobs,fwhm3.shape[0],fwhm3.shape[1]], dtype="object")
    data['rawheaders'][i,:,:] = hh # store raw header in dictionary
    data['slitfunctionFWHM-det1'][i,:,:], data['slitfunctionFWHM-det2'][i,:,:],data['slitfunctionFWHM-det3'][i,:,:] = fwhm1, fwhm2, fwhm3
    # Compute airmass
    data['airmass'][i] = np.mean([raw[0].header['HIERARCH ESO TEL AIRM START'],raw[0].header['HIERARCH ESO TEL AIRM END']])
    print(rawf[i])
    # loop over orders
    index = 0
    for o in ordernames:
        # If the order exists in the file, load data. Else: go to next
        try:
            dd,oo = np.int32(o[0]), np.int32(o[2:])
            wave = red[dd].data[str(oo).zfill(2)+'_01_WL'][pp]
            flux = red[dd].data[str(oo).zfill(2)+'_01_SPEC'][pp]
            error = red[dd].data[str(oo).zfill(2)+'_01_ERR'][pp]
        except: 
            continue
        # If the wavelength is nan, order is fucked, go to next
        if len(np.argwhere(np.isnan(wave))) > 0:
            index+=1
            continue
        # Find the right order in master
        for iim in range(len(np.unique(master['chip']))):
            iimm = np.squeeze(np.argwhere(master['chip'] == iim+1))
            wwm = master['mlambda'][iimm] * 1000
            ffm = master['flux'][iimm]
            if np.logical_and(np.nanmedian(wave) < np.max(wwm),np.nanmedian(wave) > np.min(wwm)):
                waveM = wwm[pp]
                refflux = ffm[pp]
        data['orders'][index] = o
        data['wave'][index,i,:] = wave
        data['wave_model'][index,i,:] = waveM
        data['spec'][index,i,:] = flux
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
    data['bjd_tdb'][i] = np.squeeze(utc_tbd[0])

    # Compute BERV
    bcvel = get_BC_vel(JDUTC=jd, starname=raw[0].header['OBJECT'], lat=raw[0].header['HIERARCH ESO TEL GEOLAT'], 
                                longi=raw[0].header['HIERARCH ESO TEL GEOLON'], alt=raw[0].header['HIERARCH ESO TEL GEOELEV'], zmeas=0.0)
    data['berv'][i] = np.squeeze(bcvel[0])
    # end of loop on files ##########################


wii = np.argsort(data['wave'][:,0,0])

data['wave'] = data['wave'][wii,:,:]
data['wave_model'] = data['wave_model'][wii,:,:]
data['spec'] = data['spec'][wii,:,:]
data['err'] = data['err'][wii,:,:]
data['snr'] = data['snr'][wii,:]
data['orders'] = data['orders'][wii]

with open(sys.argv[1], 'wb') as specfile:
    pickle.dump(data,specfile)
