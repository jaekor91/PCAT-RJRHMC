import time

# Matplot ticks
import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 1.
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 1.
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt

# Numpy and SciPy
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import beta as BETA

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)

def factors(num_rows, num_cols, x, y, PSF_FWHM_pix):
    """
    Test of constancy:
    dx = np.random.random() * 2 -1
    dy = np.random.random() * 2 -1

    for _ in range(100):
        print factors(64, 64, 32 + dx , 32+dy)
    """

    # Compute f, x, y gradient for each object
    lv = np.arange(0, num_rows)
    mv = np.arange(0, num_cols)
    mv, lv = np.meshgrid(lv, mv)
    var = (PSF_FWHM_pix/2.354)**2 
    PSF = gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
    PSF_sq = np.square(PSF)
    factor0 = np.sum(PSF_sq)
    factor1 = np.sum(PSF * (x - lv - 0.5)**2) / float(var **2) # sum of (dPSF/dx)^2 / PSF  
    factor2 = np.sum(PSF_sq * (x - lv - 0.5)**2) / float(var**2) # sum of (dPSF/dx)^2
    
    return factor0, factor1, factor2

def gauss_PSF(num_rows, num_cols, x, y, FWHM):
    """
    Given num_rows x num_cols of an image, generate PSF
    at location x, y.

    Note that zeroth pixel gets the value 0.5, 1st pixel 1.5, 
    and so on.
    """
    sigma = FWHM / 2.354
    xv = np.arange(0.5, num_rows)
    yv = np.arange(0.5, num_cols)
    yv, xv = np.meshgrid(xv, yv) # In my convention xv corresponds to rows and yv to columns
    PSF = np.exp(-(np.square(xv-x) + np.square(yv-y))/(2*sigma**2))/(np.pi * 2 * sigma**2)

    return PSF