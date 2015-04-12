import numpy as np
from astropy import wcs
import astropy.io.fits as pyfits
from matplotlib.path import Path
from astrogrid import wcs as astrowcs
from pdb import set_trace
from sys import exit
import matplotlib.pyplot as plt

hi = False
co = True


KPC2CM = 3.08567758e21
D_M31 = 783 * KPC2CM

TOP_DIR = '/Users/alexialewis/research/PHAT/'

## Braun data is in units of N_HI = number of HI atoms per cm^2

if hi:
    #file1 = TOP_DIR + 'ism/working_data/b02/hi_on_sfh.fits'
    file1 = TOP_DIR + 'maps/analysis/hi_braun.fits'
    #file1 = TOP_DIR + 'test_hi_reproject.fits'
    file2 = TOP_DIR + 'm31img/m31n15v3.m0lnhr.fits'

if co:
    file1 = TOP_DIR + 'maps/analysis/co_nieten.fits'
    file2 = TOP_DIR + 'm31img/m31_co.mom0.regrid.fits'

regfile = TOP_DIR + 'm31img/phat_boundaries/phat_footprint_no1-3.reg'
#regfile = TOP_DIR + 'm31img/phat_boundaries/brick_corners_02.reg'

f = open(regfile)
lines = f.readlines()
f.close()

vertstring = lines[5].lstrip('polygon(').rstrip(') # \n # dashh=1 width=2')
verts = vertstring.split(',')
rafoot = [float(ra) for ra in verts[0::2]]
decfoot = [float(dec) for dec in verts[1::2]]
footpath = Path(zip(rafoot, decfoot))


for infile in [file1,file2]:

    data, hdr = pyfits.getdata(infile, header=True)
    if data.ndim == 3:
        data = data[0]
        indexing = 'ij'
    w = wcs.WCS(hdr, naxis=2)
    x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    xx, yy = X.flatten(), Y.flatten()
    pixels = np.array(zip(yy,xx))
    pixel_matrix = pixels.reshape(data.shape[0], data.shape[1], 2)
    world = w.wcs_pix2world((pixels), 1)
    world_matrix = world.reshape(data.shape[0], data.shape[1], 2)

    pixelfoot = w.wcs_world2pix(zip(rafoot, decfoot), 1)
    pixelfootphat = Path(zip(pixelfoot[:,0], pixelfoot[:,1]))

    #in pixel coordinates
    inphat_pixel = pixelfootphat.contains_points((pixels))
    phatpixelsel = inphat_pixel.reshape(data.shape[0], data.shape[1])
    phat_pixeldata = data[phatpixelsel]
    phat_pixeldata = phat_pixeldata[np.isfinite(phat_pixeldata)]

    
    # in world_coordinates
    inphat = footpath.contains_points(world)
    phatsel = inphat.reshape(data.shape[0], data.shape[1])
    phat_data = data[phatsel]
    phat_data = phat_data[np.isfinite(phat_data)]

    ## want to get data to # of HI atoms
    pixarea = wcs.utils.proj_plane_pixel_area(w) * (np.pi / 180.)**2
    dthetax, dthetay = np.radians(np.abs(hdr['cdelt1'])), np.radians(np.abs(hdr['cdelt2']))
    dx, dy = np.tan(dthetax) * D_M31, np.tan(dthetay) * D_M31
        
    area1 = pixarea * D_M31**2
    area2 = dx * dy

    if hi:
        phat_data_lin = 10**phat_data

        num1 = np.sum(phat_data_lin * area1)
        num2 = np.sum(phat_data_lin * area2)
    if co:
        co_data = phat_data * 2e20  #XCO conversion factor

        num1 = np.sum(co_data * area1)
        num2 = np.sum(co_data * area2)
        
    print num1, num2

    plt.figure()
    plt.imshow(data, aspect='equal')
    #cb = plt.colorbar()
    plt.plot(pixelfoot[:,0], pixelfoot[:,1], 'k', lw=2)
    #plt.scatter(pixel_matrix[phatsel,0], pixel_matrix[phatsel,1], s=1,
    #            edgecolor='red', facecolor='red', alpha=0.4)
    #plt.figure()
    #plt.hist(data.flatten(), bins=10, range=[20, 22])
    

plt.show()
