#
#
#
import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube

from . import ensure_dir
from . import find_nannearest_idx
from . import find_nearest_idx

# Create a regrided lv plane map along the tilt axis to get symmetric cut
def tilted_lv(alpha, cube):
    cp_wcs = cube.wcs[:]
    cp_wcs.wcs.crpix[1] = 1.
    shape = np.shape(cube)
    hiv, null, null = cube.world[:,int(shape[1]/2),int(shape[2]/2)]
    null, hib, null = cube.world[int(shape[0]/2),:,int(shape[2]/2)]
    null, null, hil = cube.world[int(shape[0]/2),int(shape[1]/2),:]
    hil = coord.Angle(hil).wrap_at('180d').value
    hib = hib.value
    new_b = -1. * hil * np.tan(np.radians(alpha))
    new_grid_cube = np.zeros((shape[0], 1, shape[2])) # collapse b dimension
    for i in range(len(new_b)):
        new_grid_cube[:,0,i] = cube.unmasked_data[:,int(find_nannearest_idx(hib, new_b[i])[0]),i]
    return SpectralCube(data = new_grid_cube, wcs = cp_wcs)