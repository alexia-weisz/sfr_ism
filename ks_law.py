import numpy as np
import astropy.io.fits as pyfits
import glob
import sys, os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pdb import set_trace

# Global Parameters
# -----------------
D_M31 = 0.7837 * 10**3  # in kpc
M_H = 1.6733e-24  #g mass of hydrogen atom
M_SUN = 1.99e33  #g
CM_TO_PC = 3.086e18  #cm / pc
XCO = 2e20 #atom / cm^2  Strong & Mattox 1996, Dame et al 2001
ALPHA_CO = 4.35  #Msun / pc^2 (K km / s)^-1; equivalent to above XCO

if os.environ['PATH'][1:6] == 'astro':
    TOP_DIR = '/astro/store/phat/arlewis/'
else:
    TOP_DIR = '/Users/alexialewis/research/PHAT/'
    os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
    

DATA_DIR = TOP_DIR + 'maps/analysis/'
ISM_DIR = TOP_DIR + 'ism/'
PLOT_DIR = ISM_DIR + 'plots/'

def get_args():
    """ Grab the command line arguments if necessary. """
    import argparse
    parser = argparse.ArgumentParser(description='Gather SFR and HI, H2 surface densities for KS law analysis.')
    parser.add_argument('--save', action='store_true', help='Save plots.')
    parser.add_argument('--home', action='store_true',
                        help='Set if on laptop, to change top directory.')
    return parser.parse_args()


def get_data(datafile):
    """Gather data from fits files.

    Parameters
    ----------
    datafile: str
        Full location of data.

    Returns
    -------
    ndarray: data
    astropy.io.fits.Header: header

    """

    data, hdr = pyfits.getdata(datafile, header=True)
    return data, hdr


def linear_fit(x, a, b):
    return a + b * x


def get_avg_sfr(sfr_array, timebins, tstart=6.6, tstop=8.0):
    sel = (timebins[:,0] >= tstart) & (timebins[:,1] <= tstop)
    sfr_timesel = sfr_array[sel,:]
    mass100 = sfr_timesel * (10**timebins[sel,1, np.newaxis] -
                             10**timebins[sel,0, np.newaxis])

    total_mass = np.zeros((mass100.shape[1]))
    for i in range(mass100.shape[1]):
        reg_mass = np.sum(mass100[:,i][np.isfinite(mass100[:,i])])
        total_mass[i] = reg_mass
    sfr100 = total_mass / (10**tstop - 10**tstart)
    return sfr100


def plot_data(sigma_sfr, sigma_sfr_100, sigma_hi, sigma_co, save=False):
    f, ax = plt.subplots(1, 3, figsize=(10, 4))

    x1 = np.log10(sigma_hi)
    x2 = np.log10(sigma_co)
    x3 = np.log10(sigma_hi + sigma_co)
    y = np.log10(sigma_sfr_100)

    sgas = 10**np.arange(-12, 3, 0.05)
    sfe100, sfe10, sfe1 = 1.0, 0.1, 0.01

    for i, x in enumerate([x1, x2, x3]):
        sel = (np.isfinite(x)) & (np.isfinite(y))
        sel2 = sel & (y > -6)
        
        p = np.polyfit(x[sel2], y[sel2], 1)
        #p = curve_fit(linear_fit, x[sel], y[sel])

        print 'A = ', p[1]
        print 'N = ', p[0]


        ax[i].scatter(x[sel], y[sel], s=3, color='k')

        for sfe in [sfe100, sfe10, sfe1]:
            sigsfr = sfe * sgas / 1e8 * 1e6
            ax[i].plot(np.log10(sgas), np.log10(sigsfr), 'r--', lw=2)

        ax[i].plot(np.log10(sgas), p[1] + p[0] * np.log10(sgas),'b', lw=3)
        #ax[0].plot(np.log10(sgas), -6.5 + 3.2 * np.log10(sgas), 'c', lw=3)
        #ax[1].plot(np.log10(sgas), -2.0 + 3.0 * np.log10(sgas), 'c', lw=3)
        #ax[2].plot(np.log10(sgas), -6.6 + 3.25 * np.log10(sgas), 'c', lw=3)


        ax[i].set_xlim(-2.5, 2)
        ax[i].set_ylim(-6, -1)
        ax[i].xaxis.set_ticks([-2, -1, 0, 1, 2])
        ax[1].yaxis.set_ticks([])
        ax[2].yaxis.set_ticks([])
        
    ax[0].set_ylabel(r'${\rm log}_{10}\, \Sigma_{\rm SFR}\, \left[{\rm M}_{\odot} \, {\rm yr}^{-1} \, {\rm kpc}^{-2}\right]$', fontsize=18)
    ax[0].set_xlabel(r'${\rm log}_{10}\, \Sigma_{\rm HI} \, \left[{\rm M}_{\odot} \, {\rm pc}^{-2}\right]$', fontsize=18)
    ax[1].set_xlabel(r'${\rm log}_{10}\, \Sigma_{\rm H_2} \, \left[{\rm M}_{\odot} \, {\rm pc}^{-2}\right]$', fontsize=18)
    ax[2].set_xlabel(r'${\rm log}_{10}\, \Sigma_{\rm HI+H_2} \, \left[{\rm M}_{\odot} \, {\rm pc}^{-2}\right]$', fontsize=18)

        
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.95,
                        wspace=0.02)
    
    if save:
        plt.savefig(PLOT_DIR + 'sf_law_hi+h2.png', bbox_inches='tight', dpi=300)
    else:
        plt.show()

def convert_to_density(data, dtype):
    if dtype == 'hi':
        sigma = 10**data * M_H * CM_TO_PC**2 / M_SUN
    elif dtype == 'co':
        sigma = data * XCO * 1.36 * M_H * CM_TO_PC**2 / M_SUN#ALPHA_CO
    return sigma


def main(**kwargs):
    """ Spatially- and temporally-resolved Kennicutt-Schmidt star formation
    relation in M31. """
    hi_file = DATA_DIR + 'hi_braun.fits'
    co_file = DATA_DIR + 'co_nieten.fits'
    #co_file = DATA_DIR + 'co_carma.fits'
    sfr_files = sorted(glob.glob(DATA_DIR + 'sfr_evo*.fits'))[:14]

    # Get the gas: HI and CO
    hi_data, hi_hdr = get_data(hi_file)
    co_data, co_hdr = get_data(co_file)

    dthetax = np.radians(np.abs(hi_hdr['cdelt1']))
    dthetay = np.radians(np.abs(hi_hdr['cdelt2']))
    dx, dy = np.tan(dthetax) * D_M31, np.tan(dthetay) * D_M31
    pix_area = dx * dy

    sigma_hi = convert_to_density(hi_data, 'hi')
    sigma_co = convert_to_density(co_data, 'co')
    
    n_times = len(sfr_files)
    n_regions = len(sigma_hi.flatten())

    sfr_array = np.zeros((n_times, n_regions))
    time_bins = np.zeros((n_times, 2))
    for i in range(n_times):
        sfr_data, sfr_hdr = pyfits.getdata(sfr_files[i], header=True)
        ts, te = sfr_files[i].split('/')[-1].rstrip('.fits').split('_')[-1].split('-')
        if te == '6.7':
            sfr_data = sfr_data * (10**6.7 - 10**6.6) / 10**6.7
        sfr_array[i,:] = sfr_data.flatten()
        time_bins[i,:] = [float(ts), float(te)]

    sfr100 = get_avg_sfr(sfr_array, time_bins)
    sfr10 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=7.0)
    sfr10_100 = get_avg_sfr(sfr_array, time_bins, tstart=7.0, tstop=8.0)
    sfr316 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.5)
    sfr400 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.6)
    sigma_sfr = sfr_array / pix_area

    sfr_time = sfr10_100
    sigma_sfr_time = sfr_time / pix_area

    sel = (np.isfinite(sigma_hi.flatten())) & (np.isfinite(sigma_co.flatten())) & (np.isfinite(sigma_sfr_time))


    return sigma_sfr[:,sel], sigma_sfr_time[sel], sigma_hi.flatten()[sel], sigma_co.flatten()[sel]
    
#    plot_data(sigma_sfr[:,sel], sigma_sfr_time[sel],
#              sigma_hi.flatten()[sel], sigma_co.flatten()[sel], 
#              save=kwargs['save'])

if __name__ == '__main__':
    args = get_args()
    sigma_sfr, sigma_sfr_time, sigma_hi, sigma_co = main(**vars(args))
