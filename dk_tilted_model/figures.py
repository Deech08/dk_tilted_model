#
#
# 
# import dependencies

import numpy as np
import os
from astropy import units as u
from spectral_cube import SpectralCube
import astropy.coordinates as coord

import aplpy

import seaborn as sns
sns.set(color_codes=True)
sns.axes_style("white")

import matplotlib.pyplot as plt

from . import ensure_dir
from . import find_nannearest_idx
from . import find_nearest_idx

import copy

# Read in CO Cloud Catalog
this_dir, this_filename = os.path.split(__file__)
catalog_file = os.path.join(this_dir, "J_ApJ_834_57", "table1.dat.gz")

#catalog_file = __file__+'/../J_A+A_599_A109.tar.gz'  #"__file__J_A+A_599_A109.tar.gz"
co_catalog = np.genfromtxt(catalog_file, 
                           unpack = True, 
                           names = ['Cloud', 'Ncomp', 'Npix', 'A', 'l', 'e_l', 'b', 'e_b', 'theta', 
                                    'WCO', 'NH2', 'Sigma', 'vcent', 'sigmav', 'Rmax', 'Rmin', 'Rang', 
                                    'Rgal', 'INF', 'Dn', 'Df', 'zn', 'zf', 'Sn', 'Sf', 'Rn', 'Rf', 'Mn', 'Mf'])

def get_co_catalog():
    return co_catalog


def tilted_lv(alpha, cube):
    cp_wcs = copy.deepcopy(cube.wcs)
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
        new_grid_cube[:,0,i] = cube.unmasked_data[:,int(tm.find_nannearest_idx(hib, new_b[i])[0]),i]
    return SpectralCube(data = new_grid_cube, wcs = cp_wcs)

def spectra(disk_cube, lon, lat, return_data = False, fontdict = None, **kwargs):

    v, b, l_n = disk_cube.world[0,:,0]
    b = b.value
    b_slice = find_nearest_idx(b, lat)
    v, b_n, l = disk_cube.world[0,0,:]
    l = coord.Angle(l).wrap_at('180d').value
    l_slice = find_nearest_idx(l, lon)
    vel, b, l = disk_cube.world[:,b_slice,l_slice]
    data = disk_cube.unmasked_data[:,b_slice,l_slice]
    vel = vel.to(u.km/u.s)
    
    lonlab = np.round(l[l_slice], 2)
    latlab = np.round(b[b_slice], 2)
    
    fig = plt.figure(figsize = (18,12))
    ax = fig.add_subplot(111)

    ax.plot(vel, data, **kwargs)
    plt.title("Spectrum at l = "+str(lonlab[0][0])+" b = "+str(latlab[0][0]), fontdict = fontdict)
    ax.set_xlabel("LSR Velocity (km/s)", fontdict = fontdict)
    
    if fontdict:
        ax.xaxis.set_tick_params(labelsize = fontdict["size"])
        ax.yaxis.set_tick_params(labelsize = fontdict["size"])


    
    if return_data:
        return vel, data

def avg_spectra(disk_cube, lon, lat, radius, return_data = False, fontdict = None, **kwargs):
    
    v, b, l = disk_cube.world[0,:,:]
    l = coord.Angle(l).wrap_at('180d').value
    b = b.value
    
    match = np.where(np.sqrt((b - lat)**2 + (l - lon)**2) < radius)
    

    b_match = b[match[0], 0]
    l_match = l[0,match[1]]
    
    #print(b_match)
    
    v, b, l_n = disk_cube.world[0,:,0]
    b = b.value
    v, b_n, l = disk_cube.world[0,0,:]
    l = coord.Angle(l).wrap_at('180d').value
    #print(b)
    b_slices = np.zeros_like(b_match)
    l_slices = np.zeros_like(l_match)
    for ell in range(len(l_match)):
        b_slices[ell] = np.where(b == b_match[ell])[0]
        l_slices[ell] = np.where(l == l_match[ell])[0]
        
    vel, b, l = disk_cube.world[:,b_slices.astype(int),l_slices.astype(int)]
    data = disk_cube.unmasked_data[:,b_slices.astype(int),l_slices.astype(int)]
    vel = vel.to(u.km/u.s)
    
    vel = vel[:,0]
    data_avg = np.mean(data, axis = 1)
    
    fig = plt.figure(figsize = (18,12))
    ax = fig.add_subplot(111)

    ax.plot(vel, data_avg, **kwargs)
    plt.title("Spectrum at l = "+str(lon)+" b = "+str(lat), fontdict = fontdict)
    ax.set_xlabel("LSR Velocity (km/s)", fontdict = fontdict)
    
    if fontdict:
        ax.xaxis.set_tick_params(labelsize = fontdict["size"])
        ax.yaxis.set_tick_params(labelsize = fontdict["size"])
    
    if return_data:
        return vel, data_avg





def burton_figure_8_sub(data_cube, disk_cube, lat = 0., figsize = (18,12), 
                        filename = None, contour_smooth = None, contour_cmap = 'Reds', 
                        contour_levels = (0.1, 0.4, 1.8, 3.8, 7., 
                                                          11., 16., 24.),
                        cmap = 'YlGnBu_r', aspect = 'auto', stretch = 'log', vmin = 0.05, vmax = 500,
                        contour_options = {}, fits_options = {},b_label = 'both', **kwargs):
    font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 20,
        }
    
    # Data cube
    v, b, l = data_cube.world[0,:,0]
    b = b.value
    data_slice = find_nearest_idx(b, lat)
    b_data = b[data_slice]
    
    # Disk Cube
    v, bd, l = disk_cube.world[0,:,0]
    bd = bd.value
    disk_slice = find_nearest_idx(bd, lat)
    b_disk = bd[disk_slice]
    
    fig = plt.figure(figsize = (18,12))
    F = aplpy.FITSFigure(data_cube.hdu, dimensions = [2,0], slices = data_slice, figure = fig, **fits_options)
    ax = fig.gca()
    ax.invert_yaxis()
    lims = F.world2pixel([-310*1000., 310*1000.], [-12., 14.])
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.set_xticklabels(('-300','-200','-100','0','100','200', '300'), fontsize = 16)
    ax.set_yticklabels(('-10','-5','0','5','10'), fontsize = 16)
    ax.set_xlabel('VLSR (km/s)', fontsize = 16)
    ax.set_ylabel('GLON (deg)', fontsize = 16)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x, y = xlim[0]+10, ylim[1]+20
    if b_label == 'both':
        label_lat = 'b = '+str(np.round(b_data[0], 2))+' and b = '+str(np.round(b_disk[0], 2))
    else:
        label_lat = 'b = '+str(np.round(b_data[0], 2))                                                                    
    ax.text(x, y, label_lat, fontdict = font)
    F.show_colorscale(cmap = cmap, aspect = aspect, stretch = stretch, vmin = vmin, vmax = vmax, **kwargs)
    F.add_colorbar(axis_label_text = 'Temperature (K)')
    F.show_contour(disk_cube.hdu, dimensions = [2,0], slices = disk_slice, 
                   levels = contour_levels, cmap = contour_cmap, smooth = contour_smooth, **contour_options)
    if filename:
        plt.savefig(filename)
        
def burton_figure_10_sub(data_cube, disk_cube, lon = 0., figsize = (18,12), 
                        filename = None, contour_smooth = 1, contour_cmap = 'Reds', 
                        contour_levels = (0.1, 0.4, 1.8, 3.8, 7., 
                                                          11., 16., 24.),
                        cmap = 'YlGnBu_r', aspect = 'auto', stretch = 'log', vmin = 0.05, vmax = 500,
                        contour_options = {}, fits_options = {},l_label = 'both', **kwargs):
    font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 20,
        }
    
    if lon < 0:
        lon = 360. + lon
    
    # Data cube
    v, b, l = data_cube.world[0,0,:]
    l = l.value
    data_slice = find_nearest_idx(l, lon)
    l_data = l[data_slice]
    
    # Disk Cube
    v, b, l = disk_cube.world[0,0,:]
    l = l.value
    disk_slice = find_nearest_idx(l, lon)
    l_disk = l[disk_slice]
    
    fig = plt.figure(figsize = (18,12))
    F = aplpy.FITSFigure(data_cube.hdu, dimensions = [2,1], slices = data_slice, figure = fig, **fits_options)
    ax = fig.gca()
    ax.invert_yaxis()
    lims = F.world2pixel([-310*1000., 310*1000.], [-10., 10.])
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.set_xticklabels(('-300','-200','-100','0','100','200', '300'), fontsize = 16)
    ax.set_yticklabels(('-8','-4','0','4','8'), fontsize = 16)
    ax.set_xlabel('VLSR (km/s)', fontsize = 16)
    ax.set_ylabel('GLAT (deg)', fontsize = 16)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x, y = xlim[0]+10, ylim[1]-20
    if l_label == 'both':
        label_lat = 'l = '+str(np.round(l_data[0], 2))+' and l = '+str(np.round(l_disk[0], 2))
    else:
        label_lat = 'l = '+str(np.round(l_data[0], 2))                                                                    
    ax.text(x, y, label_lat, fontdict = font)
    F.show_colorscale(cmap = cmap, aspect = aspect, stretch = stretch, vmin = vmin, vmax = vmax, **kwargs)
    F.add_colorbar(axis_label_text = 'Temperature (K)')
    F.show_contour(disk_cube.hdu, dimensions = [2,1], slices = disk_slice, 
                   levels = contour_levels, cmap = contour_cmap, smooth = contour_smooth, **contour_options)
    if filename:
        plt.savefig(filename)
        
def burton_figure_8_sub_clouds(data_cube, disk_cube, lat = 0., figsize = (18,12), 
                        filename = None, contour_smooth = 1, contour_cmap = 'Reds', 
                        contour_levels = (0.1, 0.4, 1.8, 3.8, 7., 
                                                          11., 16., 24.),
                        cmap = 'YlGnBu_r', aspect = 'auto', stretch = 'log', vmin = 0.05, vmax = 500, 
                        clouds_l = co_catalog['l'], clouds_b = co_catalog['b'], clouds_v=co_catalog['vcent'],
                        contour_options = {}, fits_options = {},b_label = 'both', **kwargs):
    font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 20,
        }
    
    # Data cube
    v, b, l = data_cube.world[0,:,0]
    b = b.value
    data_slice = find_nearest_idx(b, lat)
    b_data = b[data_slice]
    
    # Disk Cube
    v, b, l = disk_cube.world[0,:,0]
    b = b.value
    disk_slice = find_nearest_idx(b, lat)
    b_disk = b[disk_slice]
    
    fig = plt.figure(figsize = (18,12))
    F = aplpy.FITSFigure(data_cube.hdu, dimensions = [2,0], slices = data_slice, figure = fig, **fits_options)
    ax = fig.gca()
    ax.invert_yaxis()
    lims = F.world2pixel([-310*1000., 310*1000.], [-13., 13.])
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.set_xticklabels(('-300','-200','-100','0','100','200', '300'), fontsize = 16)
    ax.set_yticklabels(('-10','-5','0','5','10'), fontsize = 16)
    ax.set_xlabel('VLSR (km/s)', fontsize = 16)
    ax.set_ylabel('GLON (deg)', fontsize = 16)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x, y = xlim[0]+10, ylim[1]+20
    
    if len(clouds_l) != 0:
        b_up = b[disk_slice[0]+2]
        b_down = b[disk_slice[0]-2]
        clouds_l_frame = clouds_l[(clouds_l > -13.) & (clouds_l < 13.) & \
                                  (clouds_b < b_up) & (clouds_b > b_down)]
        clouds_b_frame = clouds_b[(clouds_l > -13) & (clouds_l < 13) & \
                                  (clouds_b < b_up) & (clouds_b > b_down)]
        clouds_v_frame = clouds_v[(clouds_l > -13) & (clouds_l < 13) & \
                                  (clouds_b < b_up) & (clouds_b > b_down)]
        if len(clouds_l_frame) != 0:
            clouds_v_pix, clouds_l_pix = F.world2pixel(clouds_v_frame * 1000., clouds_l_frame)
            ax.scatter(clouds_v_pix, clouds_l_pix, color = 'red')
        
    if b_label == 'both':
        label_lat = 'b = '+str(np.round(b_data[0], 2))+' and b = '+str(np.round(b_disk[0], 2))
    else:
        label_lat = 'b = '+str(np.round(b_data[0], 2))                                                                    
    ax.text(x, y, label_lat, fontdict = font)
    F.show_colorscale(cmap = cmap, aspect = aspect, stretch = stretch, vmin = vmin, vmax = vmax, **kwargs)
    F.add_colorbar(axis_label_text = 'Temperature (K)')
    F.show_contour(disk_cube.hdu, dimensions = [2,0], slices = disk_slice, 
                   levels = contour_levels, cmap = contour_cmap, smooth = contour_smooth, **contour_options)
    if filename:
        plt.savefig(filename)
        
def burton_figure_8(data_cube, disk_cube, figsize = (18,12), 
                        filename = None, contour_smooth = 3, contour_cmap = 'Reds', 
                        contour_levels = (0.1, 0.4, 1.8, 3.8, 7., 
                                                          11., 16., 24.),
                        cmap = 'YlGnBu_r', aspect = 'auto', stretch = 'log', vmin = 0.05, vmax = 500, 
                        clouds_l = co_catalog['l'], clouds_b = co_catalog['b'], clouds_v=co_catalog['vcent'],
                        contour_options = {}, fits_options = {}, b_label = 'both', **kwargs):
    if filename:
        ensure_dir(filename)
        filename = filename[:-4]+'_a'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = 4.5, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_b'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = 3, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_c'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = 2.5, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_d'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = 0.5, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_e'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = 0, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_f'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = -.5, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_g'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = -1.5, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_h'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = -2.5, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_i'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = -3, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    if filename:
        filename = filename[:-6]+'_j'+filename[-4:]
    burton_figure_8_sub_clouds(data_cube, disk_cube, lat = -4.5, figsize = figsize, filename = filename, contour_smooth = contour_smooth,
                       contour_cmap = contour_cmap, contour_levels = contour_levels, cmap = cmap, aspect = aspect, 
                       stretch = stretch, vmin = vmin, vmax = vmax, contour_options = contour_options, 
                       fits_options = fits_options, b_label = b_label,
                       clouds_l = clouds_l, clouds_b = clouds_b, clouds_v = clouds_v, **kwargs)
    