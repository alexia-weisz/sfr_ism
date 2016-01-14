import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy.ndimage.filters as scf
import sys
import matplotlib as mpl
from matplotlib import rc, font_manager
from matplotlib.path import Path
from pdb import set_trace
import astropy.io.fits as pyfits
from astropy import wcs
import os
import settings
import correlation

D_M31 = 0.7837 * 10**3  # in kpc
M_H = 1.6733e-24  #g mass of hydrogen atom
M_SUN = 1.99e33  #g
CM_TO_PC = 3.086e18  #cm / pc
XCO = 2e20 #atom / cm^2  Strong & Mattox 1996, Dame et al 2001
ALPHA_CO = 4.35  #Msun / pc^2 (K km / s)^-1; equivalent to above XCO
INCL = np.radians(77.)


def get_args():
    """ Grab the command line arguments if necessary. """
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dosmooth', action='store_true', help='Smooth the arrays.')
    parser.add_argument('--save', action='store_true', help='Save plots.')
    parser.add_argument('--format', default='png', help='plotfile format. Default: png')
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
    w = wcs.WCS(hdr, naxis=2)
    x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y, indexing=indexing)
    xx, yy = X.flatten(), Y.flatten()
    pixels = np.array(zip(yy,xx))
    pixel_matrix = pixels.reshape(data.shape[0], data.shape[1], 2)

    return pixels


def smooth_array(array, window):
    """
    Smooths an array using a desired filter. This needs to be refined.
    """
    ## Need to mask the array so that nan values are not included in the
    ## smoothing computation
    mask = np.isnan(array)
    test_array = ma.array(array, mask=mask)
    test_array[mask] = np.mean(test_array)

    ## Choose a filter
    #sm = scf.gaussian_filter(array, window)
    sm = scf.uniform_filter(test_array, size=window)
    return sm


def convert_to_density(data, dtype):
    ## multiply by cos(incl) here to account for inclination
    ## need to multiply instead of divide because this is affecting the
    ## data itself, not just the area
    ## data * incl
    ## sfr / (area / incl) = sfr / area * incl
    if dtype == 'hi':
        sigma = 10**data * np.cos(INCL) * M_H * CM_TO_PC**2 / M_SUN
    elif dtype == 'co':
        sigma = data * np.cos(INCL) * ALPHA_CO#XCO * 1.36 * M_H * CM_TO_PC**2 / M_SUN
    return sigma


def inita(ntimes, narrays):
    all_arrays = [np.zeros(ntimes)] * narrays
    return all_arrays


## Some ideas on how to get started doing gas/dust/broadband
## comparisons.

## &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
## DEFAULTS AND DEFINITIONS
## &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

## Read in multiwavelength data. It's already all on the same
## astrometry and more or less at the same resolution. More care
## upstream is needed for a final version, but this works for
## prototyping.
args = get_args()

if os.environ['PATH'][1:6] == 'astro':
    _TOP_DIR = '/astro/store/phat/arlewis/'
else:
    _TOP_DIR = '/Users/alexialewis/research/PHAT/'
    os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

_DATA_DIR = _TOP_DIR + 'maps/analysis/'
_PLOT_DIR = _TOP_DIR + 'ism/project/plots/'

res_dir = 'res_90pc/'

weights = pyfits.getdata(_DATA_DIR + res_dir + 'weights_orig.fits')
co_data, hdr = pyfits.getdata(_DATA_DIR + res_dir + 'co_nieten.fits', header=True)
hi_data = pyfits.getdata(_DATA_DIR + res_dir + 'hi_braun.fits')
allmdust = pyfits.getdata(_DATA_DIR + res_dir + 'mdust_draine.fits')
fuv = pyfits.getdata(_DATA_DIR + res_dir + 'galex_fuv0.fits')
nuv = pyfits.getdata(_DATA_DIR + res_dir + 'galex_nuv0.fits')
ha = pyfits.getdata(_DATA_DIR + res_dir + 'ha.fits')
i24 = pyfits.getdata(_DATA_DIR + res_dir + 'mips_24.fits')

sel1 = weights == 0
co_data[sel1] = np.nan
hi_data[sel1] = np.nan
allmdust[sel1] = np.nan
fuv[sel1] = np.nan
nuv[sel1] = np.nan
ha[sel1] = np.nan
i24[sel1] = np.nan

#hdr = pyfits.getheader(_DATA_DIR + 'co_nieten.fits')

regfile = _TOP_DIR + 'ism/project/sf_regions_image.reg'
regpaths = get_reg_coords(regfile)

dshape = co_data.shape[0], co_data.shape[1]

pixels = get_pixel_coords(co_data, hdr)

# convert CO and HI to surface density
allco = convert_to_density(co_data, 'co')
allhi = convert_to_density(hi_data, 'hi')

ring_reg = regpaths[0].contains_points(pixels).reshape(dshape)
inner_reg = regpaths[1].contains_points(pixels).reshape(dshape)
outer_reg = regpaths[2].contains_points(pixels).reshape(dshape)

## Convert data to flux
# fuv, nuv, ha, i24


## Get the star formation history cube.
sfh, sfh_hdr = pyfits.getdata(_DATA_DIR + res_dir + 'sfr_evo_cube.fits', 0,
                              header=True)

## Make the time axis and note the size of the cube
taxis = np.linspace(6.6, 8.6, 21)
sz = sfh.shape

## Make a total gas map by assuming a conversion factor
#xco = 2.0e20
#tg = hi + co*xco*2.0
alltg = allhi + allco

## Combine SFR tracers
ha_i24 = ha + i24
fuv_i24 = fuv + i24

#####################
## SMOOTH (?)
#####################

## Optionally boxcar smooth the data before analysis. Lots of options
## on HOW to do this. This is another axis for exploration.

if args.dosmooth:
    all_arrays = [co, hi, mdust, fuv, nuv, ha, i24, tg, ha_i24, fuv_i24]
    kern = 1

    for i in range(len(all_arrays)):
        all_arrays[i] = smooth_array(all_arrays[i], kern)


###########################
## LOOP OVER TIMES
###########################
## Note number of times
nt = len(taxis)

## Initialize output arrays for various rank correlations
rcorr_int_co, rcorr_int_hi = np.zeros(nt), np.zeros(nt)
rcorr_int_tg, rcorr_int_mdust = np.zeros(nt), np.zeros(nt)
rcorr_int_ha, rcorr_int_fuv = np.zeros(nt), np.zeros(nt)
rcorr_int_nuv, rcorr_int_i24  = np.zeros(nt), np.zeros(nt)
rcorr_int_ha_i24, rcorr_int_fuv_i24  = np.zeros(nt), np.zeros(nt)

rcorr_delt_co, rcorr_delt_hi = np.zeros(nt), np.zeros(nt)
rcorr_delt_tg, rcorr_delt_mdust  = np.zeros(nt), np.zeros(nt)
rcorr_delt_ha, rcorr_delt_fuv = np.zeros(nt), np.zeros(nt)
rcorr_delt_nuv, rcorr_delt_i24 = np.zeros(nt), np.zeros(nt)
rcorr_delt_ha_i24, rcorr_delt_fuv_i24 = np.zeros(nt), np.zeros(nt)

#rcorr_int_jdebv = np.zeros(ntimes)


#sel = inner_reg #ring_reg inner_reg outer_reg None
sel = inner_reg | ring_reg | outer_reg#None#ring_reg | inner_reg | outer_reg
co = allco[sel]
hi = allhi[sel]
mdust = allmdust[sel]
tg = alltg[sel]

if sel == None:
    co = co[0,:,:]
    hi = hi[0,:,:]
    mdust = mdust[0,:,:]
    tg = tg[0,:,:]

## Figure out the pixels where we want to do the corelations
ind = (np.isfinite(co)) & (np.isfinite(hi))
ind2 = ind

for i in range(0, nt):
    # Calculate the SFR integrated from the time we are currently
    #  thinking about to the present.
    if i == 0:
        sfr = sfh[0,:,:]
    else:
        sfr = np.sum(sfh[0:i,:,:], axis=0) / (i + 1.0)
    sfr = sfr[sel]
    if sel == None:
        sfr = sfr[0,:,:]

    # Calculate the SFR in a two-plane window at the current time.
    if i == 0:
        sfr_delt = np.sum(sfh[0:1,:,:], axis=0) / 2.0
    else:
        sfr_delt = np.sum(sfh[i-1:i,:,:], axis=0) / 2.0
    sfr_delt = sfr_delt[sel]
    if sel == None:
        sfr_delt = sfr_delt[0,:,:]

    # Calculate rank correlations
    rcorr_int_co[i] = (spearmanr(co[ind2], sfr[ind2]))[0]
    rcorr_int_hi[i] = (spearmanr(hi[ind], sfr[ind2]))[0]
    rcorr_int_tg[i] = (spearmanr(tg[ind], sfr[ind2]))[0]
    rcorr_int_mdust[i] = (spearmanr(mdust[ind2], sfr[ind2]))[0]

    #rcorr_int_ha[i] = (spearmanr(ha[ind2], sfr[ind2]))[0]
    #rcorr_int_fuv[i] = (spearmanr(fuv[ind2], sfr[ind2]))[0]
    #rcorr_int_nuv[i] = (spearmanr(nuv[ind2], sfr[ind2]))[0]
    #rcorr_int_i24[i] = (spearmanr(i24[ind2], sfr[ind2]))[0]
    #rcorr_int_ha_i24[i] = (spearmanr(ha_i24[ind2], sfr[ind2]))[0]
    #rcorr_int_fuv_i24[i] = (spearmanr(fuv_i24[ind2], sfr[ind2]))[0]

    rcorr_delt_co[i] = (spearmanr(co[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_hi[i] = (spearmanr(hi[ind], sfr_delt[ind2]))[0]
    rcorr_delt_tg[i] = (spearmanr(tg[ind], sfr_delt[ind2]))[0]
    rcorr_delt_mdust[i] = (spearmanr(mdust[ind2], sfr_delt[ind2]))[0]

    #rcorr_delt_ha[i] = (spearmanr(ha[ind2], sfr_delt[ind2]))[0]
    #rcorr_delt_fuv[i] = (spearmanr(fuv[ind2], sfr_delt[ind2]))[0]
    #rcorr_delt_nuv[i] = (spearmanr(nuv[ind2], sfr_delt[ind2]))[0]
    #rcorr_delt_i24[i] = (spearmanr(i24[ind2], sfr_delt[ind2]))[0]
    #rcorr_delt_ha_i24[i] = (spearmanr(ha_i24[ind2], sfr_delt[ind2]))[0]
    #rcorr_delt_fuv_i24[i] = (spearmanr(fuv_i24[ind2], sfr_delt[ind2]))[0]

# CORRELATIONS AMONG THE COMPARISON DATA
#rcorr_i24_co = (spearmanr(co[ind2], i24[ind2]))[0]
#rcorr_ha_co = (spearmanr(co[ind2], ha[ind2]))[0]
#rcorr_i24_hi = (spearmanr(hi[ind], i24[ind2]))[0]
#rcorr_ha_hi = (spearmanr(hi[ind], ha[ind2]))[0]


settings.plot_env()
p1 = True
p2 = True
p3 = False
p4 = False
dpi = 300


xlab1 = r'log( t$_{\bf{max}} \;$ [Myr] )'
xlab2 = 'log( t [Myr] )'
ylab1 = r'Rank Correlation with $\langle \,$ SFR $\,$(t $<$ t$_{\bf{max}}$) $\, \rangle$'
ylab2 = 'Rank Correlation with SFR(t)'


lcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy']
fcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy']
ecol = ['purple', 'PaleVioletRed', 'darkorange', 'navy']
sym  = ['o', '^', '*', 's']
ms   = [8, 8, 11, 7]
lab  = ['CO', 'HI', 'CO+HI', r'$\textit{Herschel}$ M$_{dust}$']

if p1:
    yarr = [rcorr_int_co, rcorr_int_hi, rcorr_int_tg, rcorr_int_mdust]
    pnfile = _PLOT_DIR + 'rcorr_int_sfgas.' + args.format

    correlation.plot_correlation(taxis, yarr, sym, lcol, fcol, ecol, ms,
                                 lab, xlab1, ylab1, pnfile, save=args.save)

if p2:
    yarr = [rcorr_delt_co, rcorr_delt_hi, rcorr_delt_tg, rcorr_delt_mdust]
    pnfile = _PLOT_DIR + 'rcorr_delt_sfgas.' + args.format
    correlation.plot_correlation(taxis, yarr, sym, lcol, fcol, ecol, ms,
                                 lab, xlab2, ylab2, pnfile, save=args.save)




lcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy', 'royalblue']
fcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy', 'royalblue']
ecol = ['purple', 'PaleVioletRed', 'darkorange', 'navy', 'royalblue']
sym  = ['o', '^', '*', 's', 'D' ]
ms   = [8, 8, 11, 7, 6]
lab  = [r'H$\bf{\alpha}$', 'FUV', 'IR24', r'H$\bf{\alpha}$ + IR24',
        'FUV + IR24']

if p3:
    yarr = [rcorr_int_ha, rcorr_int_fuv, rcorr_int_i24, rcorr_int_ha_i24,
            rcorr_int_fuv_i24]
    pnfile = _PLOT_DIR + 'rcorr_int_sfsf.' + args.format

    correlation.plot_correlation(taxis, yarr, sym, lcol, fcol, ecol, ms,
                                 lab, xlab1, ylab1, pnfile, save=args.save)


if p4:
    yarr = [rcorr_delt_ha, rcorr_delt_fuv, rcorr_delt_i24,
            rcorr_delt_ha_i24, rcorr_delt_fuv_i24]
    pnfile = _PLOT_DIR + 'rcorr_delt_sfsf.' + args.format
    correlation.plot_correlation(taxis, yarr, sym, lcol, fcol, ecol, ms,
                                 lab, xlab2, ylab2, pnfile, save=args.save)
