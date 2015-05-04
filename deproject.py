from astropy.coordinates import SkyCoord, Distance, Angle
from astropy import units as u
import numpy as np
from pdb import set_trace

def correct_rgc(coord, glx_ctr=SkyCoord('00h42m44.33s', '+41d16m07.5s', 
                                        'icrs'), glx_PA=Angle('37d42m54s'),
                glx_incl=Angle('77.5d'), glx_dist=Distance(785, unit=u.kpc)):
    """Computes deprojected galactocentric distance.

    Inspired by: http://idl-moustakas.googlecode.com/svn-history/
        r560/trunk/impro/hiiregions/im_hiiregion_deproject.pro

    Parameters
    ----------
    coord : :class:`astropy.coordinates.ICRS`
        Coordinate of points to compute galactocentric distance for.
        Can be either a single coordinate, or array of coordinates.
          i.e. read in my coordinate file, compute median ra and dec for each
          box to get the center, say coords = np.array(zip(ra,dec)), then
          coord = SkyCoord(coords, 'icrs', unit='deg')
    glx_ctr : :class:`astropy.coordinates.SkyCoord
        Galaxy center.
    glx_PA : :class:`astropy.coordinates.Angle`
        Position angle of galaxy disk.
    glx_incl : :class:`astropy.coordinates.Angle`
        Inclination angle of the galaxy disk.
    glx_dist : :class:`astropy.coordinates.Distance`
        Distance to galaxy.

    Returns
    -------
    obj_dist : class:`astropy.coordinates.Distance`
        Galactocentric distance(s) for coordinate point(s).
    """
    # distance from coord to glx centre
    sky_radius = glx_ctr.separation(coord)
    avg_dec = 0.5 * (glx_ctr.dec + coord.dec).radian
    x = (glx_ctr.ra - coord.ra) * np.cos(avg_dec)
    y = glx_ctr.dec - coord.dec
    
    # azimuthal angle from coord to glx -- not completely happy with this
    #  this is projected angle
    phi = glx_PA - Angle('90d') + Angle(np.arctan(y.arcsec/x.arcsec),unit=u.rad)
    
    # convert to coordinates in rotated frame, where y-axis is galaxy major
    # ax; have to convert to arcmin b/c can't do sqrt(x^2+y^2) when x and y
    # are angles
    xp = (sky_radius * np.cos(phi.radian)).arcmin
    yp = (sky_radius * np.sin(phi.radian)).arcmin

    # de-project
    ypp = yp / np.cos(glx_incl.radian)
    obj_radius = np.sqrt(xp ** 2 + ypp ** 2)  # in arcmin
    obj_dist = Distance(Angle(obj_radius, unit=u.arcmin).radian * glx_dist,
            unit=glx_dist.unit)

    # deprojected separation angle
    theta = np.degrees(np.arctan(ypp/xp))

    return obj_dist, theta



def image_coords_deproject(ra, dec, midra, middec, pa=38.5, incl=74.):
    if args.home:
        top_dir = '/Users/alexialewis/research/PHAT/'
    else:
        top_dir = '/astro/store/phat/arlewis/'
    fuvfile = top_dir + 'ism/orig_data/m31_fuv.fits'
    mywcs = wcs.WCS(fuvfile)
    x, y = mywcs.wcs_world2pix(ra, dec, 1)
    xc, yc = mywcs.wcs_world2pix(midra, middec, 1)
    xx = x-xc
    yy = y-yc
    xp = np.cos(np.radians(pa))*xx + np.sin(np.radians(pa))*yy
    yp = -np.sin(np.radians(pa))*xx + np.cos(np.radians(pa))*yy
    xnew = xp/np.cos(np.radians(incl))
    ynew = yp
    xxx = xnew + xc
    yyy = ynew + yc
    ranew, decnew = mywcs.wcs_pix2world(xxx, yyy, 1)
    radius = distances(m31_ra, m31_dec, ranew, decnew)
    tantheta = xnew/ynew
    theta = np.arctan(tantheta)
    thetadegree = np.degrees(theta)
    return radius, thetadegree
