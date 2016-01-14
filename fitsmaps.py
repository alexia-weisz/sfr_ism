import numpy as np
import astropy.io.fits as pyfits

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
s
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


def convert_to_density(data, dtype, INCL=0):
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


def calc_int_dep_time(hi, co, area, sfr=0.28):
    total_co = np.sum(co[np.isfinite(co)] * (area * 1000**2))
    return total_co / sfr / 1e6 #Myr
