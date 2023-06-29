import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy import load
import glob
from scipy import interpolate
from scipy.optimize import least_squares,minimize
from scipy.ndimage import gaussian_filter1d
plt.rcParams['figure.figsize'] = [16, 6]
import warnings 
warnings.simplefilter('ignore', np.RankWarning)
np.seterr(all='ignore')

# Colorbrewer colours for plot. Red-blue
c0 = "#9e0142"
c1 = '#d53e4f'
c2 = '#f46d43'
c3 = '#66c2a5'
c4 = '#3288bd'
c5 = '#5e4fa2'

# Function used to cross-correlate up on down channels.
# Variables are scaling of the spectrum, and first-order
# polynomial correction to the wavelength solution
def mincrosscol_wave(offset,w1,s1,wref,sref, eref):
    ii = ~np.isnan(eref)
    w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
    ii = np.where(eref != 0)
    w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
    w1n = offset + w1 
    f1 = interpolate.interp1d(w1n, s1, bounds_error=False, fill_value="extrapolate")
    s1i = f1(wref)
    return (np.sum( (s1i[100:-100]-sref[100:-100])**2./eref[100:-100]**2 ))

def mincrosscol_scale(scale,s1,sref,eref):
    ii = ~np.isnan(eref)
    s1,sref,eref = s1[ii],sref[ii],eref[ii]
    ii = np.where(eref != 0)
    s1,sref,eref = s1[ii],sref[ii],eref[ii]
    s1 = s1 * scale
    return (np.sum( (s1[100:-100]-sref[100:-100])**2./eref[100:-100]**2 ))
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
    wave = wave 
    fmean = np.nanmean(flux)
    flux = flux - fmean
    idx = np.squeeze(np.argwhere(mask!=2)) 
    if sig is not None:
        sig[idx] =  1./ sig[idx]
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

def gen_corr_spec(w,wref,s,params):
    scale,offset = params[0],params[1]
    w1 = offset +  w
    w = w1
    s = s * scale
    f = interpolate.interp1d(w,s, bounds_error=False, fill_value="extrapolate")
    s1 = f(wref)
    return s1
version_number = "1.0"

# BEGINNING OF MAIN CODE"
# Load science files. There should be 4 files
f = glob.glob('*science.ech')
if len(f) != 4: stop()

f1 = fits.open(f[0])[1].data
f2 = fits.open(f[1])[1].data
f3 = fits.open(f[2])[1].data
f4 = fits.open(f[3])[1].data

print('Target: \t\t' + fits.open(f[0])[0].header['OBJECT'])
print('Date of 1st subexp: \t' + fits.open(f[0])[0].header['DATE-OBS'])

primary_hdu = fits.open(f[0])[0] # keep primary HDU with header
hdul_output = fits.HDUList([primary_hdu])

chip = 0
snr_max,snr_median = np.array([]),np.array([])
# Load the wavelength solution.
# For now, the wavelength solution is determined from HARPS spectroscopy data for N orders
# In HARPSpol there are N*2 spectra (N orders * 2 channels).
# So I propagate the same wavelength solution to up and down spectra. Then cross-
# correlate. [TODO]: fix the wavelength calibration step
wf = glob.glob("harps*thar.npz")[0]
thard = load(glob.glob("harps*thar.npz")[0])
mode=wf[wf.find('_')+1:wf.find('.thar')]
wave = thard['wave']
#flat = load(glob.glob("harps*flat_norm.npz")[0])
#blaze=flat['blaze']

# Loop on orders
for i in range(int(f1['SPEC'].shape[1]/2)):
    print('Order: ' + str(i).zfill(2))
    fig,ax = plt.subplots(5, figsize=(12,9))
    # Load up and down spectra
    d1,d2,d3,d4 = f1['SPEC'][0][2*i],f2['SPEC'][0][2*i],f3['SPEC'][0][2*i],f4['SPEC'][0][2*i]
    u1,u2,u3,u4 = f1['SPEC'][0][2*i+1],f2['SPEC'][0][2*i+1],f3['SPEC'][0][2*i+1],f4['SPEC'][0][2*i+1]
    # Load errors
    ed1,ed2,ed3,ed4 = f1['SIG'][0][2*i],f2['SIG'][0][2*i],f3['SIG'][0][2*i],f4['SIG'][0][2*i]
    eu1,eu2,eu3,eu4 = f1['SIG'][0][2*i+1],f2['SIG'][0][2*i+1],f3['SIG'][0][2*i+1],f4['SIG'][0][2*i+1]
    # Load the same wavelength solution for up and down
    wd,wu = wave[i,:],wave[i,:]
    data = {'WAVE' : np.empty(len(d1))}
    data['WAVE'] = wd
    # Compute mean up and down spectra
    meanu,meand = (u1+u2+u3+u4)/4., (d1+d2+d3+d4)/4.
    # Compute mean up and down errors
    meaned = (ed1**2 + ed2**2 + ed3**2 + ed4**2)**0.5
    # Cross-correlate mean up on mean down
    res = least_squares(mincrosscol_scale, 1., args=(meanu[20:-20], meand[20:-20],meaned[20:-20]), bounds=(0.,5.), verbose=0, max_nfev=2500)
    # only improve the wavelength solution
    scale_updown = np.copy(res.x)
    res = least_squares(mincrosscol_wave, 0., args=(data['WAVE'][20:-20], meanu[20:-20], data['WAVE'][20:-20], meand[20:-20],meaned[20:-20]), bounds=(-2,2), verbose=0, max_nfev=2500)
    wave_offset = np.copy(res.x)
    # Produce interpolated up spectra
    u1i = gen_corr_spec(wu, wd, u1 ,[1.,wave_offset])
    u2i = gen_corr_spec(wu, wd, u2 ,[1.,wave_offset])
    u3i = gen_corr_spec(wu, wd, u3 ,[1.,wave_offset])
    u4i = gen_corr_spec(wu, wd, u4 ,[1.,wave_offset])
	# Produce interpolated  up error spectra
    eu1i = gen_corr_spec(wu, wd, eu1 ,[1.,wave_offset])
    eu2i = gen_corr_spec(wu, wd, eu2 ,[1.,wave_offset])
    eu3i = gen_corr_spec(wu, wd, eu3 ,[1.,wave_offset])
    eu4i = gen_corr_spec(wu, wd, eu4 ,[1.,wave_offset])
    meaneui = (eu1i**2 + eu2i**2 + eu3i**2 + eu4i**2)**0.5

    # Scaling individual up spectro to mean up, individual down to mean down
    res_u1i = least_squares(mincrosscol_scale, 1., args=(u1i[20:-20],meanu[20:-20],eu1i[20:-20]),bounds=(0,10.),verbose=0, max_nfev=2500)
    res_u2i = least_squares(mincrosscol_scale, 1., args=(u2i[20:-20],meanu[20:-20],eu2i[20:-20]),bounds=(0,10.),verbose=0, max_nfev=2500)
    res_u3i = least_squares(mincrosscol_scale, 1., args=(u3i[20:-20],meanu[20:-20],eu3i[20:-20]),bounds=(0,10.),verbose=0, max_nfev=2500)
    res_u4i = least_squares(mincrosscol_scale, 1., args=(u4i[20:-20],meanu[20:-20],eu4i[20:-20]),bounds=(0,10.),verbose=0, max_nfev=2500)
    res_d1  = least_squares(mincrosscol_scale, 1., args=(d1[20:-20], meand[20:-20],ed1[20:-20]), bounds=(0,10.), verbose=0, max_nfev=2500)
    res_d2  = least_squares(mincrosscol_scale, 1., args=(d2[20:-20], meand[20:-20],ed2[20:-20]),bounds=(0,10.),verbose=0, max_nfev=2500)
    res_d3  = least_squares(mincrosscol_scale, 1., args=(d3[20:-20], meand[20:-20],ed3[20:-20]),bounds=(0,10.),verbose=0, max_nfev=2500)
    res_d4  = least_squares(mincrosscol_scale, 1., args=(d4[20:-20], meand[20:-20],ed4[20:-20]),bounds=(0,10.),verbose=0, max_nfev=2500)

    # Rescale all individual spectra to the mean down spectrum. Also rescale error spectra
    u1i,u2i,u3i,u4i = u1i*res_u1i.x[0]*scale_updown, u2i*res_u2i.x[0]*scale_updown, u3i*res_u3i.x[0]*scale_updown, u4i*res_u4i.x[0]*scale_updown
    eu1i,eu2i,eu3i,eu4i = eu1i*res_u1i.x[0]*scale_updown, eu2i*res_u2i.x[0]*scale_updown, eu3i*res_u3i.x[0]*scale_updown, eu4i*res_u4i.x[0]*scale_updown
    d1,d2,d3,d4 = d1*res_d1.x[0], d2*res_d2.x[0], d3*res_d3.x[0], d4*res_d4.x[0]
    ed1,ed2,ed3,ed4 = ed1*res_d1.x[0], ed2*res_d2.x[0], ed3*res_d3.x[0], ed4*res_d4.x[0]
    meanu = meanu * scale_updown

    # Creating the mask for the continuum fitting
    mask = np.ones((len(meand))) 
	# Excluding points where
    mask[np.where(np.isnan(meand))] = 2 # spectrum is nan
    mask[np.where(np.isnan(meaned))] = 2 # error is nan
    mask[np.where(meaned <= 0)] = 2 # error is zero or negative
    mask[0:20] = 2 # first 20 pixels
    mask[-20:-1] = 2 # last 20 pixels
    mask=mask.astype(int) # convert mask to integer
    # Fit continuum to the mean down spectra
    coefd,cId,fmd,idxd = FitCon(data['WAVE'],meand,deg=6,niter=25,sig=meaned,swin=10,k1=0.1,k2=0.3,mask=mask)

    data['CONTINUUM'] = cId * 8.

    # Compute ratio for demodulation
    R = u1i/d1 * d2/u2i * d3/u3i * u4i/d4
    # Compute Stokes parameter (V/I, Q/I, U/I)
    data['STOKES'] = (R**0.25 - 1.) / (R**0.25 + 1)

    # Compute total intensity spectrum (Stokes I) 
    data['INTENS'] = d1+d2+d3+d4 + u1i+u2i+u3i+u4i
    data['INTENS_NORM'] = data['INTENS'] / data['CONTINUUM']

    data['STOKES/Ic'] = data['STOKES'] * data['INTENS_NORM']
    # Compute error on Stokes parameter
    sumE = ((eu1i/u1i)**2 + (eu2i/u2i)**2 + (eu3i/u3i)**2 + (eu4i/u4i)**2
      + (ed1/d1)**2   + (ed2/d2)**2   + (ed3/d3)**2   + (ed4/d4)**2 )**0.5
    data['ERR_STOKES'] = sumE * 0.5 * R / (R+1.)**2. # Error spectrum for polarisation

    data['STOKES/Ic'] = data['STOKES'] * data['INTENS_NORM']
    data['ERR_STOKES/Ic'] = data['ERR_STOKES'] * data['INTENS_NORM']
    # Compute error spectrum for intensity (quadratic sum of error vectors) 
    data['ERR_INTENS'] = np.sqrt(ed1**2 + ed2**2 + ed3**2 + ed4**2 + eu1i**2 + eu2i**2 + eu3i**2 + eu4i**2)
    data['ERR_INTENS_NORM'] = np.sqrt((ed1/data['CONTINUUM'])**2 + (ed2/data['CONTINUUM'])**2 + (ed3/data['CONTINUUM'])**2 + (ed4/data['CONTINUUM'])**2 + (eu1i/data['CONTINUUM'])**2 + (eu2i/data['CONTINUUM'])**2 + (eu3i/data['CONTINUUM'])**2 + (eu4i/data['CONTINUUM'])**2)

    # Compute NULL
    RN = u1i/d1 * u2i/d2 *d3/u3i *d4/u4i 
    data['NULL'] = (RN**0.25 -1.) / (RN**0.25 +1.)
    data['ERR_NULL'] = sumE * 0.5 * RN / (RN+1.)**2.
    data['NULL/Ic'] = data['NULL'] * data['INTENS_NORM']
    data['ERR_NULL/Ic'] = data['ERR_NULL'] * data['INTENS_NORM']

    data['ORDER'] = data['WAVE'] * 0 + chip
    snr_max = np.append(snr_max,np.nanpercentile(data['INTENS']/data['ERR_INTENS'], 98))
    snr_median = np.append(snr_median,np.nanpercentile(data['INTENS']/data['ERR_INTENS'], 50))
    print('SNR (median, peak): {:.1f}\t{:.1f}'.format(np.nanpercentile(data['INTENS']/data['ERR_INTENS'], 50),np.nanpercentile(data['INTENS']/data['ERR_INTENS'], 98)))
    # Plotting demodulation plot for each order
    ax[0].plot(data['WAVE'][15:-15],data['INTENS_NORM'][15:-15], color="k", linewidth=0.7)
    ax[0].plot(data['WAVE'][15:-15],np.ones(len(data['STOKES'][15:-15])), color="r", linestyle="dashed", linewidth=0.5)
    ax[0].set_ylabel('Normalized intensity')

    ax[1].plot(data['WAVE'][15:-15],u1i[15:-15], color=c1, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],u2i[15:-15], color=c1, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],u3i[15:-15], color=c1, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],u4i[15:-15], color=c1, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],d1[15:-15], color=c4, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],d2[15:-15], color=c4, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],d3[15:-15], color=c4, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],d4[15:-15], color=c4, linewidth=0.7, alpha=0.5)
    ax[1].plot(data['WAVE'][15:-15],meand[15:-15], color=c5)
    ax[1].plot(data['WAVE'][15:-15],meanu[15:-15], color=c0)
    ax[1].plot(data['WAVE'][15:-15],data['CONTINUUM'][15:-15]/8., linewidth=1, linestyle='dotted', color='k')
    ax[1].set_ylabel('Stokes $I$')

    ax[2].plot(data['WAVE'][15:-15],data['STOKES'][15:-15], color='k', linewidth=0.7)
    ax[2].plot(data['WAVE'][15:-15],np.zeros(len(data['STOKES'][15:-15])), color="r", linestyle="dashed", linewidth=0.5)
    ax[2].set_ylabel('Stokes $V/I$')

    ax[3].plot(data['WAVE'][15:-15],data['NULL'][15:-15], color='k', linewidth=0.7, alpha=0.7)
    ax[3].plot(data['WAVE'][15:-15],np.zeros(len(data['STOKES'][15:-15])), color="r", linestyle="dashed", linewidth=0.5)
    ax[3].set_ylabel('Null spectrum')
    ax[4].plot(data['WAVE'][15:-15], data['INTENS'][15:-15]/data['ERR_INTENS'][15:-15], linewidth=0.7, color='k', alpha=0.5)
    ax[4].set_xlabel('Wavelength [nm]')
    ax[4].set_ylabel('SNR')

    plt.savefig("plot-demodulation_order-"+str(i).zfill(2)+".png")
    plt.close()

    keys_wanted = [ "WAVE","INTENS","INTENS_NORM","ERR_INTENS","ERR_INTENS_NORM","CONTINUUM","STOKES",
                    "ERR_STOKES","STOKES/Ic","ERR_STOKES/Ic","NULL","ERR_NULL","NULL/Ic","ERR_NULL/Ic","ORDER"
    ] 
    if chip==0:
        data_all = {key: data[key] for key in keys_wanted}
    else:
        data_buffer = {key: data[key] for key in keys_wanted}
        ds = [data_all, data_buffer]
        d = {}
        for k in data_all.keys():
            d[k] = np.concatenate(list(d[k] for d in ds))
        data_all = d
    chip += 1

print("median SNR, entire spectrum: {:.1f}".format(np.median(snr_median)))

col1  = fits.Column(name='WAVE', format='D', array=data_all['WAVE'][15:-15])
col2  = fits.Column(name='INTENS', format='D', array=data_all['INTENS'][15:-15])
col3  = fits.Column(name='INTENS_NORM', format='D', array=data_all['INTENS_NORM'][15:-15])
col4  = fits.Column(name='ERR_INTENS', format='D', array=data_all['ERR_INTENS'][15:-15])
col5  = fits.Column(name='ERR_INTENS_NORM', format='D', array=data_all['ERR_INTENS_NORM'][15:-15])
col6  = fits.Column(name='CONTINUUM', format='D', array=data_all['CONTINUUM'][15:-15])
col7  = fits.Column(name='STOKES', format='D', array=data_all['STOKES'][15:-15])
col8  = fits.Column(name='ERR_STOKES', format='D', array=data_all['ERR_STOKES'][15:-15])
col9  = fits.Column(name='STOKES/Ic', format='D', array=data_all['STOKES/Ic'][15:-15])
col10  = fits.Column(name='ERR_STOKES/Ic', format='D', array=data_all['ERR_STOKES/Ic'][15:-15])
col11  = fits.Column(name='NULL', format='D', array=data_all['NULL'][15:-15])
col12  = fits.Column(name='ERR_NULL', format='D', array=data_all['ERR_NULL'][15:-15])
col13  = fits.Column(name='NULL/Ic', format='D', array=data_all['NULL/Ic'][15:-15])
col14  = fits.Column(name='ERR_NULL/Ic', format='D', array=data_all['ERR_NULL/Ic'][15:-15])
col15  = fits.Column(name='ORDER', format='D', array=data_all['ORDER'][15:-15])
table_hdu = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,
                                            col11,col12,col13,col14,col15])
hdul_output.append(table_hdu)
hdul_output[0].header['DEMOD_VERS_ID'] =  (version_number, 'version number of demod script' )
hdul_output.writeto('HARPSpol-'+mode+'_demodulated.fits', overwrite=True)
