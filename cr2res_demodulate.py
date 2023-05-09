#!/usr/bin/env python3
import os,sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares,minimize
from scipy import interpolate

np.seterr(divide='ignore', invalid='ignore')

plt.rcParams['text.usetex'] = False

def mincrosscol(params,w1,s1,wref,sref, eref):
        scale,offset = params[0],params[1]
        ii = ~np.isnan(eref)
        w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
        ii = ~np.isnan(sref)
        w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
        ii = ~np.isnan(s1)
        w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
        ii = np.where(eref != 0)
        w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
        w1n = w1 + offset
        s1 = s1 * scale
        f1 = interpolate.interp1d(w1n, s1, bounds_error=False, fill_value="extrapolate")
        s1i = f1(wref)
        return (np.sum( (s1i-sref)**2./eref**2 ))

def gen_corr_spec(w,wref,s,scale,offset):
        w += offset
        s = s * scale
        f = interpolate.interp1d(w,s, bounds_error=False, fill_value="extrapolate")
        s1 = f(wref)
        return s1

def FitCon(
    wave,
    flux,
    deg=3,
    niter=10,
    sig=None,
    swin=7,
    k1=1,
    k2=3,
    mask=None,
):
    if mask is None:
        mask = np.full(wave.shape, 0)    
    else:
        sig = np.ones_like(wave)    

    wave = wave 
    fmean = np.nanmean(flux)
    flux = flux - fmean
    idx = np.squeeze(np.argwhere(mask!=2)) 
    if sig is not None:        
        sig[idx] = 1. / sig[idx]
    smooth = gaussian_filter1d(flux[idx], swin)
    coeff = np.polyfit(wave[idx], smooth, deg, w=sig[idx])
    con = np.polyval(coeff, wave)
    rms = np.nansum((con[idx] - flux[idx]) ** 2/len(idx))**0.5    # iterate niter times
    for _ in range(niter):
        idx = np.argwhere(
            (((flux - con) > (- k1 * rms)) & (flux - con < (k2 * rms)) & (mask != 2))
        )
        idx = np.squeeze(idx)
        coeff = np.polyfit(wave[idx], flux[idx], deg, w=sig[idx])
        con = np.polyval(coeff, wave)
    rms = np.nansum((con[idx] - flux[idx]) ** 2 /len(idx))**0.5     # Re-add mean flux value
    con+=fmean
    return coeff, con,fmean, idx



# Colorbrewer colours for plot. Red-blue
c0 = "#9e0142"
c1 = '#d53e4f'
c2 = '#f46d43'
c3 = '#66c2a5'
c4 = '#3288bd'
c5 = '#5e4fa2'
np.seterr(invalid='ignore')

# Open files
A1d = fits.open('cr2res_obs_pol_extractedA_1d.fits')
A1u = fits.open('cr2res_obs_pol_extractedA_1u.fits')
A2d = fits.open('cr2res_obs_pol_extractedA_2d.fits')
A2u = fits.open('cr2res_obs_pol_extractedA_2u.fits')
A3d = fits.open('cr2res_obs_pol_extractedA_3d.fits')
A3u = fits.open('cr2res_obs_pol_extractedA_3u.fits')
A4d = fits.open('cr2res_obs_pol_extractedA_4d.fits')
A4u = fits.open('cr2res_obs_pol_extractedA_4u.fits')
B1d = fits.open('cr2res_obs_pol_extractedB_1d.fits')
B1u = fits.open('cr2res_obs_pol_extractedB_1u.fits')
B2d = fits.open('cr2res_obs_pol_extractedB_2d.fits')
B2u = fits.open('cr2res_obs_pol_extractedB_2u.fits')
B3d = fits.open('cr2res_obs_pol_extractedB_3d.fits')
B3u = fits.open('cr2res_obs_pol_extractedB_3u.fits')
B4d = fits.open('cr2res_obs_pol_extractedB_4d.fits')
B4u = fits.open('cr2res_obs_pol_extractedB_4u.fits')


pp = np.arange(20,2028)
chips = ['CHIP1.INT1','CHIP2.INT1','CHIP3.INT1']
# Loop on detectors
for ic in range(3):
    ch = chips[ic]
    # find orders in up and down frames
    o1 = [nam[0:-2] for nam in A1d[ch].data.names if "WL" in nam]
    o2 = [nam[0:-2] for nam in B1u[ch].data.names if "WL" in nam]
    orders = np.intersect1d(o1,o2)
    # loop on orders
    for io in range(len(orders)):
        print('detector: ' + str(ic) + ' order: ' + str(io))
        data = {'A1u' : np.empty(2048)}
        fig,ax = plt.subplots(3, figsize=(14,10)) 
        order = orders[io]

        # Load all the spectra: u for up beam, d for down, 1234 are subexposure number, AB are nodding position
        data['A1u'],data['A2u'],data['A3u'],data['A4u'] = A1u[ch].data[order+'SPEC'],A2u[ch].data[order+'SPEC'],A3u[ch].data[order+'SPEC'],A4u[ch].data[order+'SPEC']
        data['B1u'],data['B2u'],data['B3u'],data['B4u'] = B1u[ch].data[order+'SPEC'],B2u[ch].data[order+'SPEC'],B3u[ch].data[order+'SPEC'],B4u[ch].data[order+'SPEC']
        data['A1d'],data['A2d'],data['A3d'],data['A4d'] = A1d[ch].data[order+'SPEC'],A2d[ch].data[order+'SPEC'],A3d[ch].data[order+'SPEC'],A4d[ch].data[order+'SPEC']
        data['B1d'],data['B2d'],data['B3d'],data['B4d'] = B1d[ch].data[order+'SPEC'],B2d[ch].data[order+'SPEC'],B3d[ch].data[order+'SPEC'],B4d[ch].data[order+'SPEC']
        # Load all the error spectra: u for up beam, d for down, 1234 are subexposure number, AB are nodding position
        data['eA1u'],data['eA2u'],data['eA3u'],data['eA4u'] = A1u[ch].data[order+'ERR'],A2u[ch].data[order+'ERR'],A3u[ch].data[order+'ERR'],A4u[ch].data[order+'ERR']
        data['eB1u'],data['eB2u'],data['eB3u'],data['eB4u'] = B1u[ch].data[order+'ERR'],B2u[ch].data[order+'ERR'],B3u[ch].data[order+'ERR'],B4u[ch].data[order+'ERR']
        data['eA1d'],data['eA2d'],data['eA3d'],data['eA4d'] = A1d[ch].data[order+'ERR'],A2d[ch].data[order+'ERR'],A3d[ch].data[order+'ERR'],A4d[ch].data[order+'ERR']
        data['eB1d'],data['eB2d'],data['eB3d'],data['eB4d'] = B1d[ch].data[order+'ERR'],B2d[ch].data[order+'ERR'],B3d[ch].data[order+'ERR'],B4d[ch].data[order+'ERR']

        # Define reference wavelength for A and B
        data['WL_A'] = A1u[ch].data[order+'WL']
        data['wl_dA']  = A1d[ch].data[order+'WL'] 
        data['WL_B'] = B1u[ch].data[order+'WL']
        data['wl_dB']  = B1d[ch].data[order+"WL"]

        keys = [i for i in data.keys() if np.logical_and('d' in i, 'wl' not in i)]
        # Interpolate down on up for spectra and err spectra
        for beam in keys:
            if 'A' in beam: wlref,wld = "WL_A", "wl_dA"
            else: wlref,wld = "WL_B","wl_dB"
            data[beam] = interp1d(data[wld], data[beam], kind='linear', bounds_error=False)(data[wlref])  

        ka,kb = [i for i in data.keys() if np.logical_and(len(i) == 3, 'A' in i) ], [i for i in data.keys() if np.logical_and(len(i) == 3, 'B' in i) ]
        meanIA,meanIB = sum(data[item] for item in ka)/8.,sum(data[item] for item in kb)/8.
        meanEIA,meanEIB = sum(data['e'+item]**2. for item in ka)**0.5 ,sum(data['e'+item]**2. for item in kb)**0.5
        mask = np.ones((len(meanEIA)))
        mask[np.where(np.isnan(meanIA))] = 2
        mask[np.where(np.isnan(meanIB))] = 2
        mask[np.where(np.isnan(meanEIA))] = 2
        mask[np.where(np.isnan(meanEIB))] = 2
        mask[np.where(meanEIA <= 0)] = 2
        mask[np.where(meanEIB <= 0)] = 2
        mask[0:100] = 2
        mask[-100:-1] = 2
        mask=mask.astype(int)
        coefA,cIA,fmA,idxA = FitCon(data['WL_A'],meanIA,deg=3,niter=50,sig=meanEIA,swin=7,k1=1,k2=3,mask=mask)
        coefB,cIB,fMB,idxB = FitCon(data['WL_B'],meanIB,deg=3,niter=50,sig=meanEIB,swin=7,k1=1,k2=3,mask=mask)

        for k in ka:
            res = least_squares(mincrosscol, [1.,0.], args=(data['WL_A'][20:-20],data[k][20:-20],data['WL_A'][20:-20],meanIA[20:-20],data['e'+k][20:-20]),verbose=0, max_nfev=2500)
            data[k], data['e'+k] = data[k]*res.x[0]/1., data['e'+k]*res.x[0]/1.
        for k in kb:
            res = least_squares(mincrosscol, [1.,0.], args=(data['WL_B'][20:-20],data[k][20:-20],data['WL_B'][20:-20],meanIB[20:-20],data['e'+k][20:-20]),verbose=0, max_nfev=2500)
            data[k], data['e'+k] = data[k]*res.x[0]/1., data['e'+k]*res.x[0]/1.

        # Demodulate with ratio method for A and B
        # P is Pol spectrum, I is intensity spectrum, N is null spectrum, E is error spectrum

        RA = data['A1u']/data['A1d'] * data['A2d']/data['A2u'] * data['A3d']/data['A3u'] * data['A4u']/data['A4d']  
        R4A = RA**0.25
        data['STOKES_A'] = (R4A -1.) / (R4A +1.) # polarization spectrum
        RB = data['B1u']/data['B1d'] * data['B2d']/data['B2u'] * data['B3d']/data['B3u'] * data['B4u']/data['B4d']
        R4B = RB**0.25
        data['STOKES_B'] = (R4B -1.) / (R4B +1.)

        RNA = data['A1u']/data['A1d'] * data['A2u']/data['A2d'] * data['A3d']/data['A3u'] * data['A4d']/data['A4u'] 
        data['NULL_A'] = (RNA**0.25 -1.) / (RNA**0.25 +1.) # null spectrum
        RNB = data['B1u']/data['B1d'] * data['B2u']/data['B2d'] * data['B3d']/data['B3u'] * data['B4d']/data['B4u'] 
        data['NULL_B'] = (RNB**0.25 -1.) / (RNB**0.25 +1.) # null spectrum

        keysA, keysB = [i for i in data.keys() if np.logical_and(len(i)==3, 'A' in i)],[i for i in data.keys() if np.logical_and(len(i)==3, 'B' in i)]
        data['INTENS_A'], data['INTENS_B'] = sum(data[item] for item in keysA)/8., sum(data[item] for item in keysB)/8.
        keysA, keysB = [i for i in data.keys() if np.logical_and('e' in i, 'A' in i)],[i for i in data.keys() if np.logical_and('e' in i, 'B' in i)]
        data['ERR_A'], data['ERR_B'] = np.sqrt(sum(data[item]**2. for item in keysA)), np.sqrt(sum(data[item]**2. for item in keysB))

        sumEA = ((data["eA1u"]/data["A1u"])**2. + (data["eA2u"]/data["A2u"])**2. + (data["eA3u"]/data["A3u"])**2. + (data["eA4u"]/data["A4u"])**2. 
              +  (data["eA1d"]/data["A1d"])**2. + (data["eA2d"]/data["A2d"])**2. + (data["eA3d"]/data["A3d"])**2. + (data["eA4d"]/data["A4d"])**2.)**0.5
        sumEB = ((data["eB1u"]/data["B1u"])**2. + (data["eB2u"]/data["B2u"])**2. + (data["eB3u"]/data["B3u"])**2. + (data["eB4u"]/data["B4u"])**2. 
              +  (data["eB1d"]/data["B1d"])**2. + (data["eB2d"]/data["B2d"])**2. + (data["eB3d"]/data["B3d"])**2. + (data["eB4d"]/data["B4d"])**2.)**0.5

        data['ERR_STOKES_A'] = sumEA * 0.5 * RA / (RA+1.)**2. # Error spectrum for polarisation
        data['ERR_STOKES_B'] = sumEB * 0.5 * RB / (RB+1.)**2. # Error spectrum for polarisation
        data['ERR_NULL_A']   = sumEA * 0.5 * RNA / (RNA+1.)**2. # Error spectrum on the null
        data['ERR_NULL_B']   = sumEB * 0.5 * RNB / (RNB+1.)**2. # Error spectrum on the null

        PAB = np.abs(np.append(data['STOKES_A'],data['STOKES_B']))
        PABlim = np.nanpercentile(PAB, 99.5)*1.5
        IAB = np.append(data['INTENS_B'],data['INTENS_B'])
        IABlim = np.nanpercentile(IAB, 99)*1.2
        key = [i for i in data.keys() if len(i) == 3]
        for k in key:
            if k[0] == 'A': col = c1
            else: col = c4 
            ax[0].plot(data['WL_'+k[0:1]][pp], data[k][pp], color=col,alpha=0.2)
        ax[0].plot(data['WL_A'][pp], data['INTENS_A'][pp], color=c0)
        ax[0].plot(data['WL_B'][pp], data['INTENS_B'][pp], color=c5)
        ax[0].hlines(1, np.min(data['WL_A'][pp]), np.max(data['WL_A'][pp]), linestyle='dashed', linewidth=1, color="C1", zorder=10)
        ax[1].plot(data['WL_A'][pp], data['STOKES_A'][pp], color=c1, alpha=0.8)
        ax[1].fill_between(data['WL_A'][pp], data['STOKES_A'][pp]-0.5*data['ERR_STOKES_A'][pp],data['STOKES_A'][pp]+0.5*data['ERR_STOKES_A'][pp], color=c1, alpha=0.2)
        ax[1].plot(data['WL_B'][pp], data['STOKES_B'][pp], color=c4, alpha=0.8)
        ax[1].fill_between(data['WL_B'][pp], data['STOKES_B'][pp]-0.5*data['ERR_STOKES_B'][pp],data['STOKES_B'][pp]+0.5*data['ERR_STOKES_B'][pp], color=c4, alpha=0.2)
        ax[1].hlines(0, np.min(data['WL_A'][pp]), np.max(data['WL_A'][pp]), linestyle='dashed', linewidth=1, color="#dedede", zorder=10)
        ax[2].plot(data['WL_A'][pp], data['NULL_A'][pp], color=c2, alpha=0.5)
        ax[2].plot(data['WL_B'][pp], data['NULL_B'][pp], color=c5, alpha=0.5)

        ax[1].set_ylim(-PABlim,PABlim),ax[2].set_ylim(-PABlim,PABlim),ax[0].set_ylim(-0.1,IABlim)
        ax[2].set_xlabel('Wavelength [nm]'), ax[2].set_ylabel('Null spectrum')
        ax[1].set_ylabel('Stokes ' +'$' + A1u[0].header["HIERARCH ESO INS POL TYPE"]+'$/$I$'),ax[0].set_ylabel('Stokes $I$')
        plt.savefig('plot-demodulation_order'+str(io)+'-det' + str(ic) + '.png')
        plt.close()
