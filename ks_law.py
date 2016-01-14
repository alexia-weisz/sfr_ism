import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS, Distance, Angle
import glob
import sys, os
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.optimize import curve_fit
import deproject
from pdb import set_trace

# Global Parameters
# -----------------
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

def get_args():
    """ Grab the command line arguments if necessary. """
    import argparse
    parser = argparse.ArgumentParser(description='Gather SFR and HI, H2 surface densities for KS law analysis.')
    parser.add_argument('--plot', action='store_true', help='plot.')
    parser.add_argument('--save', action='store_true', help='Save plots.')
    return parser.parse_args()


def get_reg_coords(regfile):
    f = open(regfile, 'r')
    lines = f.readlines()
    f.close()
    regs = lines[3:]
    nregs = len(regs)

    regpaths = []
    reg_coords_x = []
    reg_coords_y = []
    for r in regs:
        reg_str = r.lstrip('polygon(').rstrip(')\n').split(',')
        reg_flt = np.array([float(x) for x in reg_str])
        regpath = Path(zip(reg_flt[0::2], reg_flt[1::2]))
        regpaths.append(regpath)

    return regpaths


def get_pixel_coords(data, hdr):
    indexing = 'ij'
    w = pywcs.WCS(hdr, naxis=2)
    x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y, indexing=indexing)
    xx, yy = X.flatten(), Y.flatten()
    pixels = np.array(zip(yy,xx))
    pixel_matrix = pixels.reshape(data.shape[0], data.shape[1], 2)

    return pixels


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


def get_coords(data, hdr, m31_ra=10.6833, m31_dec=41.2692,
               pa=38.5, incl=77.):
    w = pywcs.WCS(hdr, naxis=2)
    orig_shape = (data.shape[0], data.shape[1])
    x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    xx, yy = X.flatten(), Y.flatten()
    pixels = np.array(zip(yy,xx))
    pixel_matrix = pixels.reshape(data.shape[0], data.shape[1], 2)
    world = w.wcs_pix2world(pixels, 1)
    world_matrix = world.reshape(data.shape[0], data.shape[1], 2)

    points = SkyCoord(world, frame='icrs', unit=w.wcs.cunit)
    coords = zip(points.ra, points.dec)

    coord = SkyCoord(coords, unit=u.deg)
    rads, theta = deproject.correct_rgc(coord,
                                        glx_ctr=SkyCoord(m31_ra, m31_dec,
                                                         unit=u.deg),
                                        glx_PA=Angle(pa, unit=u.deg),
                                        glx_incl=Angle(incl, unit=u.deg),
                                        glx_dist=Distance(783, unit=u.kpc))

    return rads.reshape(orig_shape), theta.reshape(orig_shape)



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
        sigma = data * np.cos(INCL) * ALPHA_CO#XCO * 1.36 * M_H * CM_TO_PC**2 / M_SUN#ALPHA_CO
    return sigma


def get_color_scheme(n):
    cm = plt.get_cmap('RdYlBu_r')
    color = [cm(1.*i/(n-1)) for i in range(n)]
    return color


def plot_contour(ax, xx, yy, bins=50, lw=2):
    H, xedges, yedges = np.histogram2d(xx, yy, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    levels = (1, 5, 10, 20, 50)
    colors = get_color_scheme(len(levels))
    cset = ax.contour(H.T, levels, origin='lower', extent=extent,
                      linewidths=lw, colors=colors)


def plot_data(sigma_sfr, sigma_sfr_100, sigma_hi, sigma_co, time=None,
              save=False):
    f, ax = plt.subplots(1, 3, figsize=(10, 4))

    x1 = np.log10(sigma_hi).flatten()
    x2 = np.log10(sigma_co).flatten()
    x3 = np.log10(sigma_hi + sigma_co).flatten()
    y = np.log10(sigma_sfr_100).flatten()

    sgas = 10**np.arange(-12, 3, 0.05)
    sfe100, sfe10, sfe1 = 1.0, 0.1, 0.01

    for i, x in enumerate([x1, x2, x3]):
        sel = (np.isfinite(x)) & (np.isfinite(y))
        sel2 = sel & (y > -6)

        p = np.polyfit(x[sel2], y[sel2], 1)
        #p = curve_fit(linear_fit, x[sel], y[sel])

        #print 'A = ', p[1]
        #print 'N = ', p[0]

        nn = str(np.around(p[0], 2))

        #ax[i].scatter(x[sel], y[sel], s=1, color='k')
        ax[i].scatter(x, y, s=1, color='k')
        plot_contour(ax[i], x[sel], y[sel])

        for sfe in [sfe100, sfe10, sfe1]:
            sigsfr = sfe * sgas / 1e8 * 1e6
            ax[i].plot(np.log10(sgas), np.log10(sigsfr), 'r--', lw=2)

        #ax[i].plot(np.log10(sgas), p[1] + p[0] * np.log10(sgas),'b', lw=3)
        #ax[0].plot(np.log10(sgas), -6.5 + 3.2 * np.log10(sgas), 'c', lw=3)
        #ax[1].plot(np.log10(sgas), -2.0 + 3.0 * np.log10(sgas), 'c', lw=3)
        #ax[2].plot(np.log10(sgas), -6.6 + 3.25 * np.log10(sgas), 'c', lw=3)


        ax[i].set_xlim(-2.5, 2)
        ax[i].set_ylim(-6, -1)
        ax[i].xaxis.set_ticks([-2, -1, 0, 1, 2])
        ax[i].tick_params(axis='both', labelsize=14)

        ax[i].text(0.06, 0.9, 'N = ' + nn, fontsize=14,
                       transform=ax[i].transAxes)


    ax[1].yaxis.set_ticks([])
    ax[2].yaxis.set_ticks([])

    ax[0].set_ylabel(r'${\rm log}_{10}\, \Sigma_{\rm SFR}\, \left[{\rm M}_{\odot} \, {\rm yr}^{-1} \, {\rm kpc}^{-2}\right]$', fontsize=18)
    ax[0].set_xlabel(r'${\rm log}_{10}\, \Sigma_{\rm HI} \, \left[{\rm M}_{\odot} \, {\rm pc}^{-2}\right]$', fontsize=18)
    ax[1].set_xlabel(r'${\rm log}_{10}\, \Sigma_{\rm H_2} \, \left[{\rm M}_{\odot} \, {\rm pc}^{-2}\right]$', fontsize=18)
    ax[2].set_xlabel(r'${\rm log}_{10}\, \Sigma_{\rm HI+H_2} \, \left[{\rm M}_{\odot} \, {\rm pc}^{-2}\right]$', fontsize=18)

    if time is not None:
        if time[0] == 10**6.6/1e6:
            time[0] = 0
        time_str = str(int(time[0])) + ' -- ' + str(int(time[1])) + ' Myr'
        ax[0].text(.06, 0.8, time_str, fontsize=16,
                   transform=ax[0].transAxes)

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.95,
                        wspace=0.02)

    if save:
        if time is not None:
            time_str2 = str(int(time[0])) + '-' + str(int(time[1]))
            plotfile = _PLOT_DIR + 'sf_law_hi+h2_' + time_str2 + '_sfregions.png'
        else:
            plotfile = P_LOT_DIR + 'sf_law_hi+h2.png'
        plt.savefig(plotfile, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def main(**kwargs):
    """ Spatially- and temporally-resolved Kennicutt-Schmidt star formation
    relation in M31. """

    res_dir = 'res_90pc/'

    hi_file = _DATA_DIR + res_dir + 'hi_braun.fits'
    co_file = _DATA_DIR + res_dir + 'co_nieten.fits'
    #co_file = DATA_DIR + 'co_carma.fits'
    weights_file = _DATA_DIR + res_dir + 'weights_orig.fits'
    sfr_files = sorted(glob.glob(_DATA_DIR + res_dir+ 'sfr_evo*-*.fits'))#[:14]

    regfile = _TOP_DIR + 'ism/project/sf_regions_image.reg'
    regpaths = get_reg_coords(regfile)

    # get the gas: HI and CO
    hi_data, hi_hdr = get_data(hi_file)
    co_data, co_hdr = get_data(co_file)
    weights, w_hdr = get_data(weights_file)

    dshape = co_data.shape[0], co_data.shape[1]
    pixels = get_pixel_coords(co_data, co_hdr)
    ring_reg = regpaths[0].contains_points(pixels).reshape(dshape)
    inner_reg = regpaths[1].contains_points(pixels).reshape(dshape)
    outer_reg = regpaths[2].contains_points(pixels).reshape(dshape)

    # determine pixel area
    dthetax = np.radians(np.abs(hi_hdr['cdelt1']))
    dthetay = np.radians(np.abs(hi_hdr['cdelt2']))
    dx, dy = np.tan(dthetax) * D_M31, np.tan(dthetay) * D_M31
    pix_area = dx * dy / np.cos(INCL)

    # get galactocentric distances
    #  only need to use one set of data because they're all on the same grid
    rads, theta = get_coords(hi_data, hi_hdr)

    # convert gas to surface density
    sigma_hi = convert_to_density(hi_data, 'hi') * weights
    sigma_co = convert_to_density(co_data, 'co') * weights

    n_times = len(sfr_files)
    n_regions = len(sigma_hi.flatten())

    # set up SFR array
    sfr_array = np.zeros((n_times, n_regions))
    time_bins = np.zeros((n_times, 2))
    for i in range(n_times):
        sfr_data, sfr_hdr = pyfits.getdata(sfr_files[i], header=True)
        ts, te = sfr_files[i].split('/')[-1].rstrip('.fits').split('_')[-1].split('-')
        #if te == '6.7':
        #    sfr_data = sfr_data * (10**6.7 - 10**6.6) / 10**6.7
        sfr_array[i,:] = sfr_data.flatten()
        time_bins[i,:] = [float(ts), float(te)]

    # compute sfrs in different time bins
    sfr100, t100 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.0)
    sfr10, t10 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=7.0)
    sfr10_100, t10_100 = get_avg_sfr(sfr_array, time_bins, tstart=7.0,
                                     tstop=8.0)
    sfr316, t316 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.5)
    sfr400, t400 = get_avg_sfr(sfr_array, time_bins, tstart=6.6, tstop=8.6)
    sfr300_400, t300_400 = get_avg_sfr(sfr_array, time_bins, tstart=8.5,
                                       tstop=8.6)
    sfr100_400, t100_400 = get_avg_sfr(sfr_array, time_bins, tstart=8.0,
                                       tstop=8.6)
    sfr30_40, t30_40 = get_avg_sfr(sfr_array, time_bins, tstart=7.5,
                                   tstop=7.6)
    sfr20_30, t20_30 = get_avg_sfr(sfr_array, time_bins, tstart=7.3,
                                   tstop=7.5)

    sfarray = [sfr10, sfr100, sfr10_100, sfr316, sfr400, sfr300_400,
               sfr100_400, sfr30_40, sfr20_30]
    tarray = [t10, t100, t10_100, t316, t400, t300_400, t100_400, t30_40,
              t20_30]

    # select desired sfr time
    for ind in range(len(sfarray)):
    #for ind in [1]:
        sigma_sfr = sfr_array[ind] / pix_area
        sfr_time, t_time = sfarray[ind], np.array(tarray[ind])/1e6
#sfr10, np.array(t100)/1e6
        sigma_sfr_time = sfr_time / pix_area

        # choose only regions where values are finite
        sel = (np.isfinite(sigma_hi.flatten())) & (np.isfinite(sigma_co.flatten())) & (np.isfinite(sigma_sfr_time)) & ((inner_reg.flatten()) | (ring_reg.flatten()) | (outer_reg.flatten()))

        #sel = (np.isfinite(sigma_hi.flatten())) & (np.isfinite(sigma_sfr_time)) & ((outer_reg.flatten()))


        total_sigma_hi = np.copy(sigma_hi)
        total_sigma_hi[np.isnan(total_sigma_hi)] = 0.0
        total_sigma_co = np.copy(sigma_co)
        total_sigma_co[np.isnan(total_sigma_co)] = 0.0
        total_gas = total_sigma_hi + total_sigma_co


        if args.plot:
            #plot_data(sigma_sfr[:,sel], sigma_sfr_time[sel],
            #          sigma_hi.flatten()[sel], sigma_co.flatten()[sel],
            #          time=t_time, save=kwargs['save'])
            plot_data(sigma_sfr, sigma_sfr_time,
                      total_sigma_hi, total_sigma_co,
                      time=t_time, save=kwargs['save'])


    return sigma_sfr[:,sel], sigma_sfr_time[sel], sigma_hi.flatten()[sel], sigma_co.flatten()[sel]

if __name__ == '__main__':
    args = get_args()
    sigma_sfr, sigma_sfr_time, sigma_hi, sigma_co = main(**vars(args))
    #main(**vars(args))

