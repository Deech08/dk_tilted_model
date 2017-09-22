#
#
#
# Definite dependencies 

import numpy as np
import os
from astropy import units as u
from spectral_cube import SpectralCube

# some universal functions

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return [idx]

def find_nannearest_idx(array,value):
    idx = np.nanargmin(np.abs(array-value))
    return [idx]


# Define the lookup table of values
# First create my "lookup table" for the Gaussian evaluated at an array of sigma (5 sigma)

sigma_base = np.arange(5001) / 1000 # sigma values
#gaussian_factor = norm.pdf(sigma_base)
gaussian_factor = np.exp(-1./2. * sigma_base**2)
gaussian_factor_table = np.append(gaussian_factor, [0])
#
#
# cube_creation items to populate my initial data cubes
from .cube_creation import *

# pre-prepped figures
from .figures import *

# cube manipulation functions
from .cube_functions import *
