import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS, Distance, Angle
import glob
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import deproject
from pdb import set_trace


def get_args():
    """ Grab the command line arguments if necessary. """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gas', default='hi+co',
                        choices=['hi+co', 'hi', 'co'],
                        help='gas surface density. default=hi+co')
    parser.add_argument('--save', action='store_true', help='Save plots.')

    return parser.parse_args()


D_M31 = 0.7837 * 10**3  # in kpc
M_H = 1.6733e-24  #g mass of hydrogen atom
M_SUN = 1.99e33  #g
CM_TO_PC = 3.086e18  #cm / pc
XCO = 2e20 #atom / cm^2  Strong & Mattox 1996, Dame et al 2001
ALPHA_CO = 4.35  #Msun / pc^2 (K km / s)^-1; equivalent to above XCO
INCL = np.radians(77.)

if os.environ['PATH'][1:6] == 'astro':
    _TOP_DIR = '/astro/store/phat/arlewis/'
else:
    _TOP_DIR = '/Users/alexialewis/research/PHAT/'
    os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'


_DATA_DIR = _TOP_DIR + 'maps/analysis/'
_ISM_DIR = _TOP_DIR + 'ism/project/'
_PLOT_DIR = _ISM_DIR + 'plots/'


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
    return sfr100, [10**tstart, 10**tstop]


def convert_to_density(data, dtype):
    ## multiply by cos(incl) here to account for inclination
    ## need to multiply instead of divide because this is affecting the
    ## data itself, not just the area
    ## data * incl
    ## sfr / (area / incl) = sfr / area * incl
    if dtype == 'hi':
        sigma = 10**data * np.cos(INCL) * M_H * CM_TO_PC**2 / M_SUN
    elif dtype == 'co':
        sigma = data * np.cos(INCL) * ALPHA_CO#XCO * 1.36 *2 * M_H * CM_TO_PC**2 / M_SUN#ALPHA_CO
    return sigma


def calc_int_dep_time(hi, co, area):
    sfr = 0.28
    total_co = np.sum(co[np.isfinite(co)] * (area * 1000**2))
    return total_co / sfr / 1e6 #Myr


#def main()
args = get_args()


res_dir = 'res_90pc/'

weights_file = _DATA_DIR + res_dir + 'weights_orig.fits'
hi_file = _DATA_DIR + res_dir + 'hi_braun.fits'
co_file = _DATA_DIR + res_dir + 'co_nieten.fits'
sfr_files = sorted(glob.glob(_DATA_DIR + res_dir + 'sfr_evo*.fits'))[:-1]#[:14]

# get the gas: HI and CO
hi_data, hi_hdr = get_data(hi_file)
co_data, co_hdr = get_data(co_file)
weights, w_hdr = get_data(weights_file)

sel1 = weights == 0
#weights[weights > 0.95] = 1

# determine pixel area
dthetax = np.radians(np.abs(hi_hdr['cdelt1']))
dthetay = np.radians(np.abs(hi_hdr['cdelt2']))
dx, dy = np.tan(dthetax) * D_M31, np.tan(dthetay) * D_M31 #kpc
pix_area = dx * dy / np.cos(INCL)

# convert gas to surface density
sigma_hi = convert_to_density(hi_data, 'hi') * weights
sigma_co = convert_to_density(co_data, 'co') * weights

total_dep_co = calc_int_dep_time(sigma_hi, sigma_co, pix_area)
print total_dep_co
sys.exit()
n_times = len(sfr_files)
n_regions = len(sigma_hi.flatten())

# set up SFR array
sfr_array = np.zeros((n_times, n_regions))
time_bins = np.zeros((n_times, 2))
for i in range(n_times):
    sfr_data, sfr_hdr = pyfits.getdata(sfr_files[i], header=True)
    ts, te = sfr_files[i].split('/')[-1].rstrip('.fits').split('_')[-1].split('-')

    sfr_array[i,:] = sfr_data.flatten()
    time_bins[i,:] = [float(ts), float(te)]

# compute sfrs in different time bins
sfr100, t100 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.0)
sfr10, t10 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=7.0)
sfr10_100, t10_100 =get_avg_sfr(sfr_array, time_bins, tstart=7.0,
                                tstop=8.0)
sfr316, t316 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.5)
sfr400, t400 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.6)
sfr300_400, t300_400 = get_avg_sfr(sfr_array, time_bins, tstart=8.5,
                                   tstop=8.6)
sfr100_400, t100_400 = get_avg_sfr(sfr_array, time_bins, tstart=8.0,
                                   tstop=8.6)

sfarray = [sfr10, sfr100, sfr10_100, sfr316, sfr400, sfr300_400,
           sfr100_400]
tarray = [t10, t100, t10_100, t316, t400, t300_400, t100_400]

sfrshape = sigma_hi.shape

## depeletion timescale = sigma_gas / sigma_sfr

# select desired sfr time
#for ind in range(len(sfarray)):
for ind in [1]:
    sigma_sfr = (sfarray[ind] / pix_area).reshape(sfrshape)
    total_sigma_co = np.copy(sigma_co)
    total_sigma_co[np.isnan(total_sigma_co)] = 0.0
    total_gas = sigma_hi + total_sigma_co

    if args.gas == 'hi+co':
        gas = total_gas
        cb_label = r'{\rm H\,{\sc i}}\,+\,$\rm H_2$'
        xx = 0.99
        plotfile = _PLOT_DIR + 'depletion_time_totalgas.png'
    elif args.gas == 'hi':
        gas = sigma_hi
        cb_label = r'{\rm H\,{\sc i}}'
        xx = 0.96
        plotfile = _PLOT_DIR + 'depletion_time_hi.png'
    elif args.gas == 'co':
        gas = sigma_co
        cb_label = r'$\rm H_2$'
        xx = 0.96
        plotfile = _PLOT_DIR + 'depletion_time_co.png'
    else:
        print 'Gas options are hi+co, hi, or co.'
        sys.exit()

    sigma_sfr[sel1] = np.nan
    gas[sel1] = np.nan

    ## multipy gas x 1e6 to convert from pc^-2 to kpc^-2 like sigma_sfr
    tau_dep = gas * 1e6 / sigma_sfr
    cnorm = mcolors.LogNorm(vmin=1e8, vmax=1e11)

    cmap = plt.cm.RdYlBu_r
    cmap.set_bad('0.65')

    fig = plt.figure(figsize=(4,7))
    plt.clf()

    plt.imshow(tau_dep, origin='lower', cmap=cmap,
               interpolation='none', norm=cnorm, aspect='equal')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.text(xx, 0.02, cb_label, transform=plt.gca().transAxes,
             horizontalalignment='right', size=16)

    cb = plt.colorbar(pad=0.02)
    cb.set_label('Depletion Time [yr]', fontsize=16)

    plt.subplots_adjust(left=-0.5)

    if args.save:
        plt.savefig(plotfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


#if _name__ == '__main__':
#    main()
