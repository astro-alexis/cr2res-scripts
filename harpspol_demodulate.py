import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy import load
import glob
from scipy import interpolate
from scipy.optimize import least_squares,minimize
plt.rcParams['figure.figsize'] = [16, 6]

np.seterr(all='ignore')

# Function used to cross-correlate up on down channels.
# Variables are scaling of the spectrum, and first-order
# polynomial correction to the wavelength solution
def mincrosscol(params,w1,s1,wref,sref, eref):
        scale,offset,a1 = params[0],params[1]/1000.,1.+params[2]/10000.
        ii = ~np.isnan(eref)
        w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
        ii = np.where(eref != 0)
        w1,s1,wref,sref,eref = w1[ii],s1[ii],wref[ii],sref[ii],eref[ii]
        w1n = offset + a1 * w1 
        s1 = s1 * scale
        f1 = interpolate.interp1d(w1n, s1, bounds_error=False, fill_value="extrapolate")
        s1i = f1(wref)
        return (np.sum( (s1i[100:-100]-sref[100:-100])**2./eref[100:-100]**2 ))

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
        scale,offset,a1 = params[0],params[1]/1000.,1.+params[2]/10000.
        w1 = offset + a1 * w
        w = w1
        s = s * scale
        f = interpolate.interp1d(w,s, bounds_error=False, fill_value="extrapolate")
        s1 = f(wref)
        return s1
version = "1.0"

# BEGINNING OF MAIN CODE"
# Load science files. There should be 4 files
f = glob.glob('*science.ech')
if len(f) != 4: stop()

f1 = fits.open(f[0])[1].data
f2 = fits.open(f[1])[1].data
f3 = fits.open(f[2])[1].data
f4 = fits.open(f[3])[1].data

# Load the wavelength solution.
# For now, the wavelength solution is determined from HARPS spectroscopy data for N orders
# In HARPSpol there are N*2 spectra (N orders * 2 channels).
# So I propagate the same wavelength solution to up and down spectra. Then cross-
# correlate. [TODO]: fix the wavelength calibration step
thard = load(glob.glob("harps*thar.npz")[0])
wave = thard['wave']

# Loop on orders
for i in range(int(f1['SPEC'].shape[1]/2)):
	print('Order: ' + str(i).zfill(2))
	fig,ax = plt.subplots(3)
	# Load up and down spectra
	d1,d2,d3,d4 = f1['SPEC'][0][2*i],f2['SPEC'][0][2*i],f3['SPEC'][0][2*i],f4['SPEC'][0][2*i]
	u1,u2,u3,u4 = f1['SPEC'][0][2*i+1],f2['SPEC'][0][2*i+1],f3['SPEC'][0][2*i+1],f4['SPEC'][0][2*i+1]
	# Load errors
	ed1,ed2,ed3,ed4 = f1['SIG'][0][2*i],f2['SIG'][0][2*i],f3['SIG'][0][2*i],f4['SIG'][0][2*i]
	eu1,eu2,eu3,eu4 = f1['SIG'][0][2*i+1],f2['SIG'][0][2*i+1],f3['SIG'][0][2*i+1],f4['SIG'][0][2*i+1]
	# Load the same wavelength solution for up and down
	wd,wu = wave[i,:],wave[i,:]
	data = {'INTENS' : np.empty(len(d1))}
	data['WAVE'] = wd
	# Cross-correlate up on down
	res = least_squares(mincrosscol, [1.,0.,0], args=(wu[20:-20].astype(dtype=np.float64),u1[20:-20].astype(dtype=np.float64),wd[20:-20].astype(dtype=np.float64),
			d1[20:-20],f1['SIG'][0][2*i][20:-20].astype(dtype=np.float64)),verbose=0, max_nfev=2500)

	# save the scaling factor into scale
	scale = res.x[0]

	# Produce interpolated and scaled up spectra
	u1i = gen_corr_spec(wu, wd, u1 ,res.x)
	u2i = gen_corr_spec(wu, wd, u2 ,res.x)
	u3i = gen_corr_spec(wu, wd, u3 ,res.x)
	u4i = gen_corr_spec(wu, wd, u4 ,res.x)
	# Produce interpolated and scaled up error spectra
	eu1i = gen_corr_spec(wu, wd, eu1 ,res.x)
	eu2i = gen_corr_spec(wu, wd, eu2 ,res.x)
	eu3i = gen_corr_spec(wu, wd, eu3 ,res.x)
	eu4i = gen_corr_spec(wu, wd, eu4 ,res.x)

	# Compute ratio for demodulation
	R = u1i/d1 * d2/u2i * d3/u3i * u4i/d4

	# Compute Stokes parameter (V/I, Q/I, U/I)
	data['STOKES'] = (R**0.25 - 1.) / (R**0.25 + 1)

	# Compute total intensity spectrum (Stokes I) 
	I1 = d1+d2+d3+d4 
	I2 = u1i + u2i + u3i + u4i
	data['INTENS'] = I1+I2

	# Compute error on Stokes parameter
	sumE = ((eu1i/u1i)**2 + (eu2i/u2i)**2 + (eu3i/u3i)**2 + (eu4i/u4i)**2
              + (ed1/d1)**2   + (ed2/d2)**2   + (ed3/d3)**2   + (ed4/d4)**2 )**0.5

	data['ERR_STOKES'] = sumE * 0.5 * R / (R+1.)**2. # Error spectrum for polarisation
	# Compute error spectrum for intensity (quadratic sum of error vectors) 
	meanE = np.sqrt(ed1**2 + ed2**2 + ed3**2 + ed4**2 + eu1i**2 + eu2i**2 + eu3i**2 + eu4i**2)
	data['ERR_INTENS'] = meanE

	# Plotting demodulation plot for each order
	ax[0].plot(data['WAVE'][15:-15],I1[15:-15], 'C0')
	ax[0].plot(data['WAVE'][15:-15],I2[15:-15], 'C1')

	ax[1].plot(data['WAVE'][15:-15],data['STOKES'][15:-15], 'k')

	ax[2].plot(data['WAVE'][15:-15], data['INTENS'][15:-15]/data['ERR_INTENS'][15:-15], linewidth=2, color='gray')
	ax[2].set_xlabel('Wavelength [nm]')
	ax[2].set_ylabel('SNR')
	ax[1].set_ylabel('Stokes V/I')
	ax[0].set_ylabel('Stokes I')
	plt.savefig("plot-demodulation_order-"+str(i).zfill(2)+".png")
	plt.close()
