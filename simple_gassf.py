import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy.ndimage.filters as scf
import sys
import matplotlib as mpl
from matplotlib import rc, font_manager
from pdb import set_trace
import pyfits


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

brick = str(sys.argv[1]).zfill(2)

#topdir = '/astro/store/student-scratch1/arlewis/PHAT/ism/'
topdir = '/Users/alexialewis/research/PHAT/ism/'
bdir = 'b'+brick+'/'
indir = topdir+'working_data/'+bdir
plotdir = topdir+'plots/'

co = pyfits.getdata(indir+'co_on_sfh.fits')
hi = pyfits.getdata(indir+'hi_on_sfh.fits')
#ebv = pyfits.getdata(indir+'ebv_on_sfh.fits')
mdust = pyfits.getdata(indir+'mdust_on_sfh.fits')
fuv = pyfits.getdata(indir+'fuv_on_sfh.fits')
nuv = pyfits.getdata(indir+'nuv_on_sfh.fits')
ha = pyfits.getdata(indir+'ha_on_sfh.fits')
i24 = pyfits.getdata(indir+'mips24_on_sfh.fits')
jdebv = pyfits.getdata(indir+'jd_ebv_on_sfh.fits')

## Get the star formation history cube.
sfh, sfh_hdr = pyfits.getdata(indir+'sfh_cube.fits', 0, header=True)

## Make the time axis and note the size of the cube
v = np.arange(sfh_hdr['NAXIS3'])
vdif = v - (sfh_hdr['CRPIX3'] - 1)
taxis = (vdif * sfh_hdr['CDELT3']+ sfh_hdr['CRVAL3'])

sz = sfh.shape

## Figure out the pixels where we want to do the corelations
ind = (np.isfinite(co)) & (np.isfinite(hi)) & (np.isfinite(jdebv))[0]
ind2  = ind[0,:,:]

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

dosmooth = 1
kern = 1

if dosmooth:
    co = smooth_array(co, kern)
    hi = smooth_array(hi, kern)
    #ebv = smooth_array(ebv, kern)
    mdust = smooth_array(mdust, kern)
    fuv = smooth_array(fuv, kern)
    nuv = smooth_array(nuv, kern)
    ha = smooth_array(ha, kern)
    i24 = smooth_array(i24, kern)
    tg = smooth_array(tg, kern)
    ha_i24 = smooth_array(ha_i24, kern)
    fuv_i24 = smooth_array(fuv_i24, kern)
    
    #jdebv = smooth_array(jdebv, kern)

###########################
## LOOP OVER TIMES
###########################
#set_trace()
## Note number of times
ntimes = len(taxis)

## Initialize output arrays for various rank correlations
rcorr_int_co = np.zeros(ntimes)
rcorr_int_hi = np.zeros(ntimes)
rcorr_int_tg = np.zeros(ntimes)
rcorr_int_mdust = np.zeros(ntimes)
#rcorr_int_ebv = np.zeros(ntimes)
rcorr_int_ha = np.zeros(ntimes)
rcorr_int_fuv = np.zeros(ntimes)
rcorr_int_nuv = np.zeros(ntimes)
rcorr_int_i24 = np.zeros(ntimes)
rcorr_int_jdebv = np.zeros(ntimes)
rcorr_int_ha_i24 = np.zeros(ntimes)
rcorr_int_fuv_i24 = np.zeros(ntimes)

#  rcorr_int_i100 = np.zeros(ntimes)
#  rcorr_int_i250 = np.zeros(ntimes)

rcorr_delt_co = np.zeros(ntimes)
rcorr_delt_hi = np.zeros(ntimes)
rcorr_delt_tg = np.zeros(ntimes)
rcorr_delt_mdust = np.zeros(ntimes)
#rcorr_delt_ebv = np.zeros(ntimes)
rcorr_delt_ha = np.zeros(ntimes)
rcorr_delt_fuv = np.zeros(ntimes)
rcorr_delt_nuv = np.zeros(ntimes)
rcorr_delt_i24 = np.zeros(ntimes)
rcorr_delt_jdebv = np.zeros(ntimes)
rcorr_delt_jdebv = np.zeros(ntimes)
rcorr_delt_ha_i24 = np.zeros(ntimes)
rcorr_delt_fuv_i24 = np.zeros(ntimes)
#  rcorr_delt_i100 = np.zeros(ntimes)
#  rcorr_delt_i250 = np.zeros(ntimes)

for i in range(0, ntimes):
##   Calculate the SFR integrated from the time we are currently
##   thinking about to the present.
    if i == 0:
        sfr = sfh[0,:,:]
    else:
        sfr = np.sum(sfh[0:i,:,:], axis=0)/(i+1.0)

##   Calculate the SFR in a two-plane window at the current time.
    if i == 0:
        sfr_delt = np.sum(sfh[0:1,:,:], axis=0)/2.0
    else:
        sfr_delt = np.sum(sfh[i-1:i,:,:], axis=0)/2.0

##   Calculate rank correlations
    rcorr_int_co[i] = (spearmanr(co[ind2], sfr[ind2]))[0]
    rcorr_int_hi[i] = (spearmanr(hi[ind], sfr[ind2]))[0]
    rcorr_int_tg[i] = (spearmanr(tg[ind], sfr[ind2]))[0]
    rcorr_int_mdust[i] = (spearmanr(mdust[ind2], sfr[ind2]))[0]
    #rcorr_int_ebv[i] = (spearmanr(ebv[ind2], sfr[ind2]))[0]
    rcorr_int_jdebv[i] = (spearmanr(jdebv[ind2], sfr[ind2]))[0]
    
    rcorr_int_ha[i] = (spearmanr(ha[ind2], sfr[ind2]))[0]
    rcorr_int_fuv[i] = (spearmanr(fuv[ind2], sfr[ind2]))[0]
    rcorr_int_nuv[i] = (spearmanr(nuv[ind2], sfr[ind2]))[0]
    rcorr_int_i24[i] = (spearmanr(i24[ind2], sfr[ind2]))[0]
    rcorr_int_ha_i24[i] = (spearmanr(ha_i24[ind2], sfr[ind2]))[0]
    rcorr_int_fuv_i24[i] = (spearmanr(fuv_i24[ind2], sfr[ind2]))[0]
#    rcorr_int_i100[i] = (spearmanr(i100[ind], sfr[ind2]))[0]
#    rcorr_int_i250[i] = (spearmanr(i250[ind], sfr[ind2]))[0]
    
    rcorr_delt_co[i] = (spearmanr(co[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_hi[i] = (spearmanr(hi[ind], sfr_delt[ind2]))[0]
    rcorr_delt_tg[i] = (spearmanr(tg[ind], sfr_delt[ind2]))[0]
    rcorr_delt_mdust[i] = (spearmanr(mdust[ind2], sfr_delt[ind2]))[0]
    #rcorr_delt_ebv[i] = (spearmanr(ebv[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_jdebv[i] = (spearmanr(jdebv[ind2], sfr_delt[ind2]))[0]
        
    rcorr_delt_ha[i] = (spearmanr(ha[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_fuv[i] = (spearmanr(fuv[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_nuv[i] = (spearmanr(nuv[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_i24[i] = (spearmanr(i24[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_ha_i24[i] = (spearmanr(ha_i24[ind2], sfr_delt[ind2]))[0]
    rcorr_delt_fuv_i24[i] = (spearmanr(fuv_i24[ind2], sfr_delt[ind2]))[0]

#    rcorr_delt_i100[i] = (spearmanr(i100[ind], sfr_delt[ind2]))[0]
#    rcorr_delt_i250[i] = (spearmanr(i250[ind], sfr_delt[ind2]))[0]


## CORRELATIONS AMONG THE COMPARISON DATA
rcorr_i24_co = (spearmanr(co[ind2], i24[ind2]))[0]
rcorr_ha_co = (spearmanr(co[ind2], ha[ind2]))[0]
rcorr_i24_hi = (spearmanr(hi[ind], i24[ind2]))[0]
rcorr_ha_hi = (spearmanr(hi[ind], ha[ind2]))[0]



##########
## PLOT ##
##########
import os
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

p1 = True
p2 = True
p3 = True
p4 = True
dpi=300

rc('text', usetex=True)
#rc('font', family='sans-serif')
#rc('font',**{'family':'sans-serif','serif':'Times'})

lcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy','royalblue']
fcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy','royalblue']
ecol = ['purple', 'PaleVioletRed', 'darkorange', 'navy','royalblue']
sym = ['o', '^', '*', 's', 'D']
ms = [8, 8, 11, 7, 6]
lab = ['CO', 'HI', 'CO+HI', r'$\textit{Herschel}$ M$_{dust}$', 'E(B-V)']

if p1:
    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)
    ## for i in range(-10, 50):
    ##     x = np.array([-1e6, 1e6])
    ##     y = np.array([i*0.05*1, i*0.05*1])
    ##     plt.plot(x, y, ':', lw=1, color='gray')

    ## plt.plot(taxis, taxis*0+rcorr_i24_co, '--', lw=3, color='purple')
    ## plt.plot(taxis, taxis*0+rcorr_i24_hi, '--', lw=3, color='firebrick')
    ## plt.plot(taxis, taxis*0+rcorr_ha_co, '-.', lw=3, color='purple')
    ## plt.plot(taxis, taxis*0+rcorr_ha_hi, '-.', lw=3, color='firebrick')

    yarr = [rcorr_int_co, rcorr_int_hi, rcorr_int_tg, rcorr_int_mdust,
            rcorr_int_jdebv]

    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], 
                 markerfacecolor=fcol[j],
                 markeredgecolor=ecol[j], markersize=ms[j], label=lab[j],
                 linewidth=2, linestyle='-')

    plt.legend(numpoints=1, markerscale=1.5, frameon=False, loc=0)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel(r't$_{\bf{max}} \;\;$ [Myr]', fontsize=18)
    plt.ylabel(r'Rank Correlation with $\langle \,$ SFR $\,$(t $<$ t$_{\bf{max}}$) $\, \rangle$', fontsize=18)

    psfile = plotdir+'B'+brick+'_rcorr_int_sfgas.eps'
    pnfile = plotdir+'B'+brick+'_rcorr_int_sfgas.png'
    plt.savefig(psfile, format='eps')
    plt.savefig(pnfile, dpi=dpi, bbox_inches='tight')


if p2:
    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)    

    yarr = [rcorr_delt_co, rcorr_delt_hi, rcorr_delt_tg, rcorr_delt_mdust,
            rcorr_delt_jdebv]
    
    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], mfc=fcol[j],
                 mec=ecol[j], markersize=ms[j], label=lab[j], linewidth=2,
                 linestyle='-')

    plt.legend(numpoints=1, markerscale=1.5, frameon=False)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel('t  [Myr]', fontsize=18)
    plt.ylabel('Rank Correlation with SFR(t)', fontsize=18)

    psfile = plotdir+'B'+brick+'_rcorr_delt_sfgas.eps'
    pnfile = plotdir+'B'+brick+'_rcorr_delt_sfgas.png'
    plt.savefig(psfile, format='eps')
    plt.savefig(pnfile, dpi=300, bbox_inches='tight')




lcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy', 'royalblue']
fcol = ['purple', 'PaleVioletRed', 'darkorange', 'navy', 'royalblue']
ecol = ['purple', 'PaleVioletRed', 'darkorange', 'navy', 'royalblue']
sym = ['o', '^', '*', 's', 'D' ]
ms = [8, 8, 11, 7, 6]
lab = [r'H$\bf{\alpha}$', 'FUV', 'IR24', r'H$\bf{\alpha}$ + IR24',
       'FUV + IR24']

if p3:
    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)
    ## for i in range(-10, 50):
    ##     x = np.array([-1e6, 1e6])
    ##     y = np.array([i*0.05*1, i*0.05*1])
    ##     plt.plot(x, y, ':', lw=1, color='gray')

    ## plt.plot(taxis, taxis*0+rcorr_i24_co, '--', lw=3, color='purple')
    ## plt.plot(taxis, taxis*0+rcorr_i24_hi, '--', lw=3, color='firebrick')
    ## plt.plot(taxis, taxis*0+rcorr_ha_co, '-.', lw=3, color='purple')
    ## plt.plot(taxis, taxis*0+rcorr_ha_hi, '-.', lw=3, color='firebrick')

    yarr = [rcorr_int_ha, rcorr_int_fuv, rcorr_int_i24, rcorr_int_ha_i24,
            rcorr_int_fuv_i24]

    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], 
                 markerfacecolor=fcol[j],
                 markeredgecolor=ecol[j], markersize=ms[j], label=lab[j],
                 linewidth=2, linestyle='-')

    if brick == '16':
        plt.legend(numpoints=1, markerscale=1.5, frameon=False, loc=7)
    else:
        plt.legend(numpoints=1, markerscale=1.5, frameon=False)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel(r't$_{\bf{max}} \;\;$ [Myr]', fontsize=18)
    plt.ylabel(r'Rank Correlation with $\langle \,$ SFR $\,$(t $<$ t$_{\bf{max}}$) $\, \rangle$', fontsize=18)

    psfile = plotdir+'B'+brick+'_rcorr_int_sfsf.eps'
    pnfile = plotdir+'B'+brick+'_rcorr_int_sfsf.png'
    plt.savefig(psfile, format='eps')
    plt.savefig(pnfile, dpi=300, bbox_inches='tight')


if p4:
    fig = plt.figure(figsize=(8,8))
    plt.plot([-1e6, 1e6], [0,0], '-', color='gray', lw=2)    
    yarr = [rcorr_delt_ha, rcorr_delt_fuv, rcorr_delt_i24, 
            rcorr_delt_ha_i24, rcorr_delt_fuv_i24]
    for j in range(0, len(yarr)):
        plt.plot(taxis, yarr[j], sym[j], color=lcol[j], mfc=fcol[j],
                 mec=ecol[j], markersize=ms[j], label=lab[j], linewidth=2,
                 linestyle='-')

    plt.legend(numpoints=1, markerscale=1.5, frameon=False)
    plt.xlim(taxis[0], taxis[-1])
    plt.ylim(-0.2, 1.001)
    plt.xlabel('t  [Myr]', fontsize=18)
    plt.ylabel('Rank Correlation with SFR(t)', fontsize=18)

    psfile = plotdir+'B'+brick+'_rcorr_delt_sfsf.eps'
    pnfile = plotdir+'B'+brick+'_rcorr_delt_sfsf.png'
    plt.savefig(psfile, format='eps')
    plt.savefig(pnfile, dpi=300, bbox_inches='tight')



#plt.show()
