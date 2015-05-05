import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy.ndimage.filters as scf
import sys
import matplotlib as mpl
from matplotlib import rc, font_manager
from pdb import set_trace
import astropy.io.fits as pyfits
import os
import settings


def get_args():
    """ Grab the command line arguments if necessary. """
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dosmooth', action='store_true', help='Smooth the arrays.')
    parser.add_argument('--save', action='store_true', help='Save plots.')
    parser.add_argument('--format', default='png', help='plotfile format. Default: png')
    parser.add_argument('--home', action='store_true',
                        help='Set if on laptop, to change top directory.')
    return parser.parse_args()


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
    TOP_DIR = '/astro/store/phat/arlewis/'
else:
    TOP_DIR = '/Users/alexialewis/research/PHAT/'
    os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

DATA_DIR = TOP_DIR + 'maps/analysis/'
PLOT_DIR = TOP_DIR + 'ism/project/plots/'

co = pyfits.getdata(DATA_DIR + 'co_nieten.fits')
hi = pyfits.getdata(DATA_DIR + 'hi_braun.fits')
mdust = pyfits.getdata(DATA_DIR + 'mdust_draine.fits')
fuv = pyfits.getdata(DATA_DIR + 'galex_fuv0.fits')
nuv = pyfits.getdata(DATA_DIR + 'galex_nuv0.fits')
ha = pyfits.getdata(DATA_DIR + 'ha.fits')
i24 = pyfits.getdata(DATA_DIR + 'mips_24.fits')


## Convert data to flux
# fuv, nuv, ha, i24


## Get the star formation history cube.
sfh, sfh_hdr = pyfits.getdata(DATA_DIR + 'sfr_evo_cube.fits', 0, header=True)

## Make the time axis and note the size of the cube
taxis = np.linspace(6.6, 8.6, 21)
sz = sfh.shape

## Figure out the pixels where we want to do the corelations
ind = (np.isfinite(co)) & (np.isfinite(hi))
ind2 = ind

## Make a total gas map by assuming a conversion factor
xco = 2.0e20
tg = hi + co*xco*2.0

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
ntimes = len(taxis)

## Initialize output arrays for various rank correlations
rcorr_int_co = np.zeros(ntimes)
rcorr_int_hi = np.zeros(ntimes)
rcorr_int_tg = np.zeros(ntimes)
rcorr_int_mdust = np.zeros(ntimes)
rcorr_int_ha = np.zeros(ntimes)
rcorr_int_fuv = np.zeros(ntimes)
rcorr_int_nuv = np.zeros(ntimes)
rcorr_int_i24 = np.zeros(ntimes)
rcorr_int_jdebv = np.zeros(ntimes)
rcorr_int_ha_i24 = np.zeros(ntimes)
rcorr_int_fuv_i24 = np.zeros(ntimes)

rcorr_delt_co = np.zeros(ntimes)
rcorr_delt_hi = np.zeros(ntimes)
rcorr_delt_tg = np.zeros(ntimes)
rcorr_delt_mdust = np.zeros(ntimes)
rcorr_delt_ha = np.zeros(ntimes)
rcorr_delt_fuv = np.zeros(ntimes)
rcorr_delt_nuv = np.zeros(ntimes)
rcorr_delt_i24 = np.zeros(ntimes)
rcorr_delt_ha_i24 = np.zeros(ntimes)
rcorr_delt_fuv_i24 = np.zeros(ntimes)

for i in range(0, ntimes):
    # Calculate the SFR integrated from the time we are currently
    #  thinking about to the present.
    if i == 0:
        sfr = sfh[0,:,:]
    else:
        sfr = np.sum(sfh[0:i,:,:], axis=0)/(i+1.0)

    # Calculate the SFR in a two-plane window at the current time.
    if i == 0:
        sfr_delt = np.sum(sfh[0:1,:,:], axis=0)/2.0
    else:
        sfr_delt = np.sum(sfh[i-1:i,:,:], axis=0)/2.0

    # Calculate rank correlations
    rcorr_int_co[i] = (spearmanr(co[ind2], sfr[ind2]))[0]
    rcorr_int_hi[i] = (spearmanr(hi[ind], sfr[ind2]))[0]
    rcorr_int_tg[i] = (spearmanr(tg[ind], sfr[ind2]))[0]
    rcorr_int_mdust[i] = (spearmanr(mdust[ind2], sfr[ind2]))[0]
    
    rcorr_int_ha[i] = (spearmanr(ha[ind2], sfr[ind2]))[0]
    rcorr_int_fuv[i] = (spearmanr(fuv[ind2], sfr[ind2]))[0]
    rcorr_int_nuv[i] = (spearmanr(nuv[ind2], sfr[ind2]))[0]
    rcorr_int_i24[i] = (spearmanr(i24[ind2], sfr[ind2]))[0]
    rcorr_int_ha_i24[i] = (spearmanr(ha_i24[ind2], sfr[ind2]))[0]
    rcorr_int_fuv_i24[i] = (spearmanr(fuv_i24[ind2], sfr[ind2]))[0]
    
    rcorr_delt_co[i] = (spearmanr(co[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_hi[i] = (spearmanr(hi[ind], sfr_delt[ind2]))[0]
    rcorr_delt_tg[i] = (spearmanr(tg[ind], sfr_delt[ind2]))[0]
    rcorr_delt_mdust[i] = (spearmanr(mdust[ind2], sfr_delt[ind2]))[0]
        
    rcorr_delt_ha[i] = (spearmanr(ha[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_fuv[i] = (spearmanr(fuv[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_nuv[i] = (spearmanr(nuv[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_i24[i] = (spearmanr(i24[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_ha_i24[i] = (spearmanr(ha_i24[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_fuv_i24[i] = (spearmanr(fuv_i24[ind2], sfr_delt[ind2]))[0]

# CORRELATIONS AMONG THE COMPARISON DATA
rcorr_i24_co = (spearmanr(co[ind2], i24[ind2]))[0]
rcorr_ha_co = (spearmanr(co[ind2], ha[ind2]))[0]
rcorr_i24_hi = (spearmanr(hi[ind], i24[ind2]))[0]
rcorr_ha_hi = (spearmanr(hi[ind], ha[ind2]))[0]


settings.plot_env()
p1 = True
p2 = True
p3 = True
p4 = True
dpi = 300

lcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy']
fcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy']
ecol = ['purple', 'PaleVioletRed', 'darkorange', 'navy']
sym  = ['o', '^', '*', 's']
ms   = [8, 8, 11, 7]
lab  = ['CO', 'HI', 'CO+HI', r'$\textit{Herschel}$ M$_{dust}$']



xlab1 = r't$_{\bf{max}} \;\;$ [Myr]'
xlab2 = 't  [Myr]'
ylab1 = r'Rank Correlation with $\langle \,$ SFR $\,$(t $<$ t$_{\bf{max}}$) $\, \rangle$'
ylab2 = 'Rank Correlation with SFR(t)'

if p1:
    yarr = [rcorr_int_co, rcorr_int_hi, rcorr_int_tg, rcorr_int_mdust]

    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)

    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], 
                 markerfacecolor=fcol[j],
                 markeredgecolor=ecol[j], markersize=ms[j], label=lab[j],
                 linewidth=2, linestyle='-')

    plt.legend(numpoints=1, markerscale=1.5, frameon=False, loc=0)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel(xlab1, fontsize=18)
    plt.ylabel(ylab1, fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    
    if args.save:
        pnfile = PLOT_DIR + 'rcorr_int_sfgas.' + args.format
        plt.savefig(pnfile, dpi=dpi, bbox_inches='tight')


if p2:
    yarr = [rcorr_delt_co, rcorr_delt_hi, rcorr_delt_tg, rcorr_delt_mdust]
    
    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)    
    
    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], mfc=fcol[j],
                 mec=ecol[j], markersize=ms[j], label=lab[j], linewidth=2,
                 linestyle='-')

    plt.legend(numpoints=1, markerscale=1.5, frameon=False)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel(xlab2, fontsize=18)
    plt.ylabel(ylab2, fontsize=18)
    plt.tick_params(axis='both', labelsize=14)

    if args.save:
        pnfile = PLOT_DIR + 'rcorr_delt_sfgas.' + args.format
        plt.savefig(pnfile, dpi=300, bbox_inches='tight')




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

    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)

    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], 
                 markerfacecolor=fcol[j],
                 markeredgecolor=ecol[j], markersize=ms[j], label=lab[j],
                 linewidth=2, linestyle='-')
        
    plt.legend(numpoints=1, markerscale=1.5, frameon=False)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel(xlab1, fontsize=18)
    plt.ylabel(ylab1, fontsize=18)
    plt.tick_params(axis='both', labelsize=14)

    if args.save:
        pnfile = PLOT_DIR + 'rcorr_int_sfsf.' + args.format
        plt.savefig(pnfile, dpi=300, bbox_inches='tight')


if p4:
    yarr = [rcorr_delt_ha, rcorr_delt_fuv, rcorr_delt_i24, 
            rcorr_delt_ha_i24, rcorr_delt_fuv_i24]

    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)    

    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], mfc=fcol[j],
                 mec=ecol[j], markersize=ms[j], label=lab[j], linewidth=2,
                 linestyle='-')

    plt.legend(numpoints=1, markerscale=1.5, frameon=False)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel(xlab2, fontsize=18)
    plt.ylabel(ylab2, fontsize=18)
    plt.tick_params(axis='both', labelsize=14)

    if args.save:
        pnfile = PLOT_DIR + 'rcorr_delt_sfsf.' + args.format
        plt.savefig(pnfile, dpi=300, bbox_inches='tight')


if not args.save:
    plt.show()
