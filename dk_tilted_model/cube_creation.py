# Define tilting function

#import dependencies
import numpy as np
import os
from astropy import units as u
from spectral_cube import SpectralCube
import numexpr as ne


from astropy.coordinates.representation import CylindricalRepresentation, CartesianRepresentation, CartesianDifferential

#import gala.coordinates as gc
#from gala.coordinates import cylindrical_to_cartesian, cartesian_to_cylindrical

import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph

from astropy import wcs

import scipy.interpolate
import scipy.integrate as integrate

import multiprocessing
from functools import partial

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



class TiltedDisk(coord.BaseCoordinateFrame):
    """
    A cartesian coordinate system in the frame of the tilted elliptical disk 

    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)

    x : `Quantity`, optional, must be keyword
        The x coordinate in the tilted disk coordinate system
    y : `Quantity`, optional, must be keyword
        The y cooridnate in the titled disk coordinate system
    z : `Quantity`, optional, must be keyword
        The z coordinate in the tilted disk coordinate system
        
    v_x : :class:`~astropy.units.Quantity`, optional, must be keyword
        The x component of the velocity
    v_y : :class:`~astropy.units.Quantity`, optional, must be keyword
        The y component of the velocity
    v_z : :class:`~astropy.units.Quantity`, optional, must be keyword
        The z component of the velocity
    """
    default_representation = coord.CartesianRepresentation
    default_differential = coord.CartesianDifferential
    
    frame_specific_representation_info = {
        coord.representation.CartesianDifferential: [
            coord.RepresentationMapping('d_x', 'v_x', u.km/u.s),
            coord.RepresentationMapping('d_y', 'v_y', u.km/u.s),
            coord.RepresentationMapping('d_z', 'v_z', u.km/u.s),
        ],
    }
    
    # Specify frame attributes required to fully specify the frame
    # Rotation angles
    alpha = coord.QuantityAttribute(default=0.*u.rad, unit = u.rad)
    beta = coord.QuantityAttribute(default=0.*u.rad, unit = u.rad)
    theta = coord.QuantityAttribute(default=0.*u.rad, unit = u.rad)



def get_transformation_matrix(tilteddisk_frame, inverse = False):
    alpha = tilteddisk_frame.alpha.value
    beta = tilteddisk_frame.beta.value
    theta = tilteddisk_frame.theta.value
    # Generate rotation matrix for coordinate transformation into coord.Galactocentric
    R_matrix = np.array([np.cos(beta)*np.cos(theta), np.cos(beta)*np.sin(theta), -np.sin(beta), 
                        -np.cos(theta)*np.sin(alpha)*-np.sin(beta) - np.cos(alpha)*np.sin(theta), 
                        np.cos(alpha)*np.cos(theta) + np.sin(alpha)*np.sin(beta)*np.sin(theta), 
                        np.cos(beta)*np.sin(alpha), 
                        np.cos(alpha)*np.cos(theta)*np.sin(beta) + np.sin(alpha)*np.sin(theta), 
                        -np.cos(theta)*np.sin(alpha) + np.cos(alpha)*np.sin(beta)*np.sin(theta), 
                        np.cos(alpha)*np.cos(beta)]).reshape(3,3)
    if inverse:
        return R_matrix.transpose()
    else:
        return R_matrix
    
@frame_transform_graph.transform(coord.DynamicMatrixTransform, TiltedDisk, coord.Galactocentric)
def td_to_galactocentric(tilteddisk_coord, galactocentric_frame):
    """ Compute the transformation matrix from the Tilted Disk 
        coordinates to Galactocentric coordinates.
    """
    return get_transformation_matrix(tilteddisk_coord)
    
@frame_transform_graph.transform(coord.DynamicMatrixTransform, coord.Galactocentric, TiltedDisk)
def galactocentric_to_td(galactocentric_coord, tilteddisk_frame):
    """ Compute the transformation matrix from Galactocentric coordinates to
        Tilted Disk coordinates.
    """
    return get_transformation_matrix(tilteddisk_frame, inverse = True)



def ellipse_equation(bd, el_constant1, el_constant2, bd_max, x_coord, y_coord):
    a = bd *el_constant1 + el_constant2 * bd**2 / bd_max
    result = x_coord**2 / a**2 + y_coord**2 / bd**2 - 1.
    #result = bd**2 / a**2 * (a**2 - x_coord**2) - y_coord**2
    #print(result)
    return result

def bd_equation(bd, bd_max, x_coord, y_coord):
    a = x_coord
    result = x_coord**2 / a**2 + y_coord**2 / bd**2 - 1.
    #result = bd**2 / a**2 * (a**2 - x_coord**2) - y_coord**2
    #print(result)
    return result

def bd_solver(ell, xyz, z_sigma_lim, Hz, bd_max, el_constant1, el_constant2):
        x_coord = xyz[0,ell]
        y_coord = xyz[1,ell]
        z_coord = xyz[2,ell]
        if z_coord > z_sigma_lim*Hz:
            res = bd_max+1.
        elif np.abs(y_coord) > bd_max:
            res = bd_max+1.
        elif np.abs(x_coord) > bd_max * (el_constant1 + el_constant2):
            res = bd_max+1.
        elif x_coord == 0.:
            if y_coord == 0.:
                res = 0
            else:
                res = y_coord
        elif y_coord == 0.:
            res = scipy.optimize.brenth(bd_equation, 0.000001, 1., 
                        args = (bd_max, x_coord, y_coord))
        else:
            res = scipy.optimize.brenth(ellipse_equation, 0.000001, 1., 
                        args = (el_constant1, el_constant2,bd_max, x_coord, y_coord))
            if res<0:
                print(el_constant1,el_constant2,bd_max, x_coord, y_coord )
        return res

def create_lbd_grid(resolution = (64,64,64), bd_max = 0.6, Hz = 0.1, z_sigma_lim = 3, dens0 = 0.33, q = 0.,
                   velocity_factor = 0.1, vel_0 = 360., el_constant1 = 1.6, el_constant2 = 1.5, return_all = False,
                   alpha = 0., beta = 0., theta = 0., L_range = [-10,10], B_range = [-8,8], D_range = [7,13], 
                   LSR_options={}, **kwargs):
    nx, ny, nz = resolution
    
    lbd_grid = np.mgrid[L_range[0]:L_range[1]:nx*1j,
                        B_range[0]:B_range[1]:ny*1j,
                        D_range[0]:D_range[1]:nz*1j]
    lbd = lbd_grid.T.reshape(-1,3, order = "F").transpose()
    lbd_coords = coord.Galactic(l = lbd[0,:]*u.deg, b = lbd[1,:]*u.deg, distance = lbd[2,:]*u.kpc)
    galcen_coords = lbd_coords.transform_to(coord.Galactocentric(**kwargs))
    disk_coords = galcen_coords.transform_to(TiltedDisk(alpha = alpha*u.deg, 
                                                        beta = beta*u.deg, theta = theta*u.deg))
    disk_coords_arr = np.array([disk_coords.x.value, disk_coords.y.value, disk_coords.z.value])
    xyz_grid = disk_coords_arr.T.transpose().reshape(-1,nx,ny,nz)
    
    #bd_grid = np.zeros_like(xyz_grid[0,:,:,:])
    
    partial_bd_solver = partial(bd_solver, xyz=disk_coords_arr, z_sigma_lim = z_sigma_lim, Hz = Hz, 
                        bd_max = bd_max, el_constant1 = el_constant1, el_constant2 = el_constant2)
    
    pool = multiprocessing.Pool()
    bd_vals = pool.map(partial_bd_solver, range(len(disk_coords.x.value)))
    bd_grid = np.array(bd_vals).T.transpose().reshape(nx,ny,nz)
    
    ad_grid = ne.evaluate("bd_grid * (el_constant1 + el_constant2 * bd_grid / bd_max)")
    dens_grid = np.zeros_like(bd_grid)
    z_coor = xyz_grid[2,:,:,:]
    dens_grid[(np.abs(z_coor)<(z_sigma_lim * Hz)) & (bd_grid<bd_max)] = dens0 * \
            np.exp(-0.5 * (z_coor[(np.abs(z_coor)<(z_sigma_lim * Hz)) & (bd_grid<bd_max)] / Hz)**2)

    r_x = xyz_grid[0,:,:,:]
    r_y = xyz_grid[1,:,:,:]
    
    normalizer = ne.evaluate("1 / sqrt((r_x / ad_grid**2)**2 + (r_y / bd_grid**2)**2)")
    
    xtangent = ne.evaluate("r_y / bd_grid**2 * normalizer")
    ytangent = ne.evaluate("-r_x / ad_grid**2 * normalizer")
    
    Lz_minor_axis = ne.evaluate("0. - bd_grid * vel_0 * (1. - exp(-bd_grid / velocity_factor))")  #r x v
    vel_magnitude_grid = ne.evaluate("abs(Lz_minor_axis / (r_x * ytangent - r_y * xtangent))")

    vel_xyz = np.zeros_like(xyz_grid)

    vel_xyz[0,:,:,:] = ne.evaluate("xtangent * vel_magnitude_grid")
    vel_xyz[1,:,:,:] = ne.evaluate("ytangent * vel_magnitude_grid")
    
    np.nan_to_num(vel_xyz)

    velocity_xyz = vel_xyz.T.reshape(-1,3, order = "F").transpose() * u.km/ u.s

    vel_cartesian = CartesianRepresentation(velocity_xyz)

    disk_coordinates = TiltedDisk(x = disk_coords.x, y = disk_coords.y, z = disk_coords.z,
                v_x = vel_cartesian.x, v_y = vel_cartesian.y, v_z = vel_cartesian.z, 
                alpha = alpha*u.deg, beta = beta*u.deg, theta = theta*u.deg)
    
    galcen_coords_withvel = disk_coordinates.transform_to(coord.Galactocentric(**kwargs))
    lbd_coords_withvel = galcen_coords_withvel.transform_to(coord.GalacticLSR(**LSR_options))
    
    dD = lbd_grid[2,0,0,1] - lbd_grid[2,0,0,0]
    dB = lbd_grid[1,0,1,1] - lbd_grid[1,0,0,0]
    dL = lbd_grid[0,1,0,0] - lbd_grid[0,0,0,0]
    cdelt = np.array([dL, dB, dD])

    if return_all:
        return lbd_coords_withvel, np.swapaxes(dens_grid,0,2), cdelt, disk_coordinates, \
            galcen_coords_withvel, np.swapaxes(bd_grid,0,2), np.swapaxes(vel_magnitude_grid,0,2)
    else:
        return lbd_coords_withvel, np.swapaxes(dens_grid,0,2), cdelt
    
def optical_depth_function(ell, density, gaussian):
    gaussian_slice = gaussian[:,:,:,ell]
    return ne.evaluate("gaussian_slice * density")

def create_lbv_cube(lbd_coords_withvel, density_gridin, cdelt, vel_disp = 9., vmin = -350., vmax = 350., 
                    vel_resolution = 545, T_gas = 120., species = 'hi',L_range = [-10,10], B_range = [-8,8]):
    # Define the lookup table of values
    # First create my "lookup table" for the Gaussian evaluated at an array of sigma (5 sigma)

    nz, ny, nx = density_gridin.shape
    #sigma_base = np.arange(5001) / 1000 # sigma values
    #gaussian_factor = ne.evaluate("exp(-1./2. * sigma_base**2)")
    #gaussian_factor_table = np.append(gaussian_factor, [0])

    # Define the velocity channels
    VR, dv = np.linspace(vmin,vmax,vel_resolution, retstep=True)
    vr_grid = np.swapaxes(lbd_coords_withvel.radial_velocity.value.T.transpose().reshape(nx,ny,nz),0,2)
    
    
    # Calculate my sigma values
    vr_grid_plus = vr_grid[:,:,:,None]
    #sigma_cells = ne.evaluate("abs((vr_grid_plus - VR) / vel_disp * 1000)").astype(int)
    #sigma_cells[sigma_cells > 5000] = 5001
    #sigma_cells[sigma_cells < 0] = 5001
    #gaussian_cells = gaussian_factor_table[sigma_cells]
    
    gaussian_cells = ne.evaluate("exp(-1/2. * ((vr_grid_plus - VR) / vel_disp)**2)")
    
    dist = cdelt[2]
    if species == 'hi':
        density_grid = ne.evaluate("density_gridin *33.52 / (T_gas * vel_disp)* dist *1000. / 50.")
        optical_depth = np.einsum('jkli, jkl->ijkl', gaussian_cells, density_grid).sum(axis = 1)
        emission_cube = ne.evaluate("T_gas * (1 - exp(-1.* optical_depth))") 
    if species =='ha':
        EM = ne.evaluate("density_gridin**2 * dist * 1000.")
        emission_cube = np.einsum('jkli, jkl->ijkl', gaussian_cells, EM).sum(axis = 1)
        
    # Create WCS Axes
    DBL_wcs = wcs.WCS(naxis = 3)
    DBL_wcs.wcs.crpix=[int(nx/2),int(ny/2),int(vel_resolution/2)]
    DBL_wcs.wcs.crval=[np.sum(L_range)/2, np.sum(B_range)/2, (vmax+vmin)/2]
    DBL_wcs.wcs.ctype=["GLON-CAR", "GLAT-CAR", "VRAD"]
    DBL_wcs.wcs.cunit=["deg", "deg", "km/s"]
    DBL_wcs.wcs.cdelt=np.array([cdelt[0], cdelt[1], dv])

    
    out_cube = np.swapaxes(emission_cube, 0,2)
    
    return SpectralCube(data = emission_cube, wcs = DBL_wcs)
    
def tilted_model_cube(resolution = (64,64,64), vel_resolution = 545, L_range = [-10,10], 
                        B_range = [-8,8], D_range = [7,13], bd_max = 0.6, Hz = 0.1, z_sigma_lim = 3, 
                        dens0 = 0.33, q = 0., velocity_factor = 0.1, vel_0 = 360., alpha = 13.5, beta = 20., theta = 48.5, 
                        precession = False, precession_omega = 0.0, precession_profile = 'constant', 
                        vel_disp = 9., vmin = -325, vmax = 325, species = 'hi', el_constant1 =1.6, 
                        el_constant2 =1.5, T_gas = 120., LSR_options = {},
                        return_all = False, filename = None, **kwargs):
    if return_all:
        lbd_coords, disk_density, cdelta, disk_coordinate_frame, galcen_coords, bd_grid, vel_mag_grid = \
            create_lbd_grid(resolution = resolution, bd_max =bd_max, Hz = Hz, z_sigma_lim = z_sigma_lim, 
                    dens0 = dens0, q = q, velocity_factor = velocity_factor, vel_0 = vel_0, 
                    el_constant1 = el_constant1, el_constant2 = el_constant2, return_all = return_all,
                    alpha = alpha, beta = beta, theta = theta, L_range = L_range, B_range = B_range, 
                    D_range = D_range, LSR_options=LSR_options, 
                    **kwargs)
    else:
        lbd_coords, disk_density, cdelta = \
            create_lbd_grid(resolution = resolution, bd_max =bd_max, Hz = Hz, z_sigma_lim = z_sigma_lim, 
                    dens0 = dens0, q = q, velocity_factor = velocity_factor, vel_0 = vel_0, 
                    el_constant1 = el_constant1, el_constant2 = el_constant2, return_all = return_all,
                    alpha = alpha, beta = beta, theta = theta, L_range = L_range, B_range = B_range, 
                    D_range = D_range, LSR_options=LSR_options, 
                    **kwargs)
            
    cube = create_lbv_cube(lbd_coords, disk_density, cdelta, vel_disp = vel_disp, vmin = vmin, vmax = vmax, 
                    vel_resolution = vel_resolution, T_gas = T_gas,
                    species = 'hi', L_range = L_range, B_range = B_range)
    
    if filename:
        ensure_dir(filename)
        cube.write(filename)
    
    if return_all:
        return cube, lbd_coords, disk_density, cdelta, disk_coordinate_frame, galcen_coords, bd_grid, vel_mag_grid
    else:
        return cube
