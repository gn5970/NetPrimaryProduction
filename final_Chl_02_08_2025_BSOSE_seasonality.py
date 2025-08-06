from scipy import io
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib import ticker
from cmcrameri import cm
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.stats.mstats import gmean
from collections import namedtuple
from scipy.stats import t as stu
from scipy.stats import gaussian_kde
import os
from eofs.xarray import Eof
# Extract variables X and y

import h5py
import xarray as xr
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import h5py
import statsmodels.api as sm
from scipy.stats.mstats import gmean
from collections import namedtuple
from scipy.stats import t as stu
from scipy.stats import gaussian_kde
import matplotlib.ticker as mticker

import numpy as np
import random

random.seed(42)
np.random.seed(42)

def red_noise(N, M, g):
    red = np.zeros(shape=(N, M))
    red[0, :] = np.random.randn(1, M)
    for i in np.arange(1, N):
        red[i, :] = g * red[i-1, :] + np.random.randn(1, M)
    return red

def ar1_fit(y):
    from statsmodels.tsa.arima.model import ARIMA
    ar1_mod = ARIMA(y, order=(1, 0, 0), missing='drop', trend='ct').fit()
    g = ar1_mod.params[2]
    if g > 1:
        eps = np.spacing(1.0)
        g = 1.0 - eps**(1/4)
    return g

def isopersistent_rn(X, M):
    N = np.size(X)
    mu = np.mean(X)
    sig = np.std(X, ddof=1)

    g = ar1_fit(X)
    red = red_noise(N, M, g)
    m = np.mean(red)
    s = np.std(red, ddof=1)

    red_n = (red - m) / s
    red_z = red_n * sig + mu

    return red_z, g

def corr_isopersist(x, y, alpha=0.05, nsim=1000):
    A = np.corrcoef(x, y)
    r = A[1][0]
    ra = np.abs(r)

    x_red, g1 = isopersistent_rn(x, nsim)
    y_red, g2 = isopersistent_rn(y, nsim)

    rs = np.zeros(nsim)
    for i in np.arange(nsim):
        B = np.corrcoef(x_red[:, i], y_red[:, i])
        rs[i] = B[1][0]

    rsa = np.abs(rs)
    xi = np.linspace(0, 1.1*np.max([ra, np.max(rsa)]), 200)
    kde = gaussian_kde(rsa)
    prob = kde(xi).T

    diff = np.abs(ra - xi)
    pos = np.argmin(diff)
    pval = np.trapz(prob[pos:], xi[pos:])
    rcrit = np.percentile(rsa, 100*(1-alpha))
    signif = ra >= rcrit
    return r, rcrit, pval




def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def remove_time_mean(x):
    return x - x.mean(dim='time',skipna=True)

data_thetao = xr.open_dataset("/projects/CDEUTSCH/DATA/Theta_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
print('data_thetao',data_thetao)
#data_thetao = convert_0_360_to_neg180_180(data_thetao)

data_thetao = data_thetao.where(data_thetao.Z >= -100, drop=True).mean('Z')

time_mean_THETA_clim = data_thetao.THETA.groupby('time.month').mean(dim='time',skipna=True)
time_mean_THETA_anom = data_thetao.THETA.groupby('time.month')-time_mean_THETA_clim
time_mean_THETA_anom=detrend_dim(time_mean_THETA_anom,dim='time')
time_mean_THETA_anom=time_mean_THETA_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_THETA_anom2=time_mean_THETA_anom/time_mean_THETA_anom.std()
time_mean_THETA_anom2=time_mean_THETA_anom2.coarsen(XC=16, boundary="pad").mean()




data_ssh = xr.open_dataset("/projects/CDEUTSCH/DATA/SSH_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_ssh = convert_0_360_to_neg180_180(data_ssh)
print('data_ssh',data_ssh)

time_mean_SSH_clim = data_ssh.ETAN.groupby('time.month').mean(dim='time',skipna=True)
time_mean_SSH_anom = data_ssh.ETAN.groupby('time.month')-time_mean_SSH_clim
time_mean_SSH_anom=detrend_dim(time_mean_SSH_anom,dim='time')
time_mean_SSH_anom=time_mean_SSH_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_SSH_anom2=time_mean_SSH_anom/time_mean_SSH_anom.std()
time_mean_SSH_anom2=time_mean_SSH_anom2.coarsen(XC=16, boundary="pad").mean()

data_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_salt = convert_0_360_to_neg180_180(data_salt)
data_salt = data_salt.where(data_salt.Z >= -100, drop=True).mean('Z')

time_mean_SALT_clim = data_salt.SALT.groupby('time.month').mean(dim='time',skipna=True)
time_mean_SALT_anom = data_salt.SALT.groupby('time.month')-time_mean_SALT_clim
time_mean_SALT_anom=time_mean_SALT_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_SALT_anom2=time_mean_SALT_anom/time_mean_SALT_anom.std()
time_mean_SALT_anom2=time_mean_SALT_anom2.coarsen(XC=16, boundary="pad").mean()


data_SIArea = xr.open_dataset("/projects/CDEUTSCH/DATA/SIArea_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_SIArea = convert_0_360_to_neg180_180(data_SIArea)


time_mean_SIArea_clim = data_SIArea.SIarea.groupby('time.month').mean(dim='time',skipna=True)
time_mean_SIArea_anom = data_SIArea.SIarea.groupby('time.month')-time_mean_SIArea_clim
time_mean_SIArea_anom=detrend_dim(time_mean_SIArea_anom,dim='time')
time_mean_SIArea_anom=time_mean_SIArea_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_SIArea_anom2=time_mean_SIArea_anom/time_mean_SIArea_anom.std()
time_mean_SIArea_anom2=time_mean_SIArea_anom2.coarsen(XC=16, boundary="pad").mean()

data_NO3 = xr.open_dataset("/projects/CDEUTSCH/DATA/NO3_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")

def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_O2 = convert_0_360_to_neg180_180(data_O2)
data_NO3 = data_NO3.where(data_NO3.Z >= -100, drop=True).mean('Z')

print(data_thetao)
print(data_ssh)
print(data_salt)

time_mean_TRAC04_clim = data_NO3.TRAC04.groupby('time.month').mean(dim='time',skipna=True)
time_mean_TRAC04_anom = data_NO3.TRAC04.groupby('time.month')-time_mean_TRAC04_clim
time_mean_TRAC04_anom=detrend_dim(time_mean_TRAC04_anom,dim='time')
time_mean_TRAC04_anom=time_mean_TRAC04_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_TRAC04_anom2=time_mean_TRAC04_anom/time_mean_TRAC04_anom.std()
time_mean_TRAC04_anom2=time_mean_TRAC04_anom2.coarsen(XC=16, boundary="pad").mean()

data_irris = xr.open_dataset("/projects/CDEUTSCH/DATA/irris_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_O2 = convert_0_360_to_neg180_180(data_O2)
data_irris = data_irris.where(data_irris.Z >= -100, drop=True).mean('Z')

print('data_O2',data_irris)

time_mean_irris_clim = data_irris.BLGIRRIS.groupby('time.month').mean(dim='time',skipna=True)
time_mean_irris_anom = data_irris.BLGIRRIS.groupby('time.month')-time_mean_irris_clim
time_mean_irris_anom=detrend_dim(time_mean_irris_anom,dim='time')
time_mean_irris_anom=time_mean_irris_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_irris_anom2=time_mean_irris_anom/time_mean_irris_anom.std()
time_mean_irris_anom2=time_mean_irris_anom2.coarsen(XC=16, boundary="pad").mean()



data_Fe = xr.open_dataset("/projects/CDEUTSCH/DATA/Fe_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_Fe = convert_0_360_to_neg180_180(data_Fe)
data_Fe = data_Fe.where(data_Fe.Z >= -100, drop=True).mean('Z')

print('data_Fe',data_Fe)
time_mean_TRAC06_clim = data_Fe.TRAC06.groupby('time.month').mean(dim='time',skipna=True)
time_mean_TRAC06_anom = data_Fe.TRAC06.groupby('time.month')-time_mean_TRAC06_clim
time_mean_TRAC06_anom=detrend_dim(time_mean_TRAC06_anom,dim='time')
time_mean_TRAC06_anom=time_mean_TRAC06_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_TRAC06_anom2=time_mean_TRAC06_anom/time_mean_TRAC06_anom.std()
time_mean_TRAC06_anom2=time_mean_TRAC06_anom2.coarsen(XC=16, boundary="pad").mean()


data_NPP = xr.open_dataset("/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_NPP = convert_0_360_to_neg180_180(data_NPP)
data_NPP = data_NPP.where(data_NPP.Z >= -100, drop=True).mean('Z')
print('data_NPP',data_NPP)

for i in range(1):
    
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    
    levels = np.linspace(-1.5*10**(-8), 1.5*10**(-8), 40)
    fill = ax.contourf(
    np.array(data_NPP.XC),
    np.array(data_NPP.YC),
    np.array(data_NPP.BLGNPP.mean('time')).squeeze(),
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
    )
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    cb.set_label('NPP (mol C/m3/s)', fontsize=20, labelpad=15)  
    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("NPP.png")








print('data_NPP',data_NPP)

time_mean_NPP_clim = data_NPP.BLGNPP.groupby('time.month').mean(dim='time',skipna=True)
time_mean_NPP_anom = data_NPP.BLGNPP.groupby('time.month')-time_mean_NPP_clim
time_mean_NPP_anom=detrend_dim(time_mean_NPP_anom,dim='time')
time_mean_NPP_anom=time_mean_NPP_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_NPP_anom2=time_mean_NPP_anom/time_mean_NPP_anom.std()
time_mean_NPP_anom2=time_mean_NPP_anom2.coarsen(XC=16, boundary="pad").mean()

from eofs.xarray import Eof

# Assume ds_anom_interpolated is a DataArray with dims ('time', 'lat', 'lon')
# Step 1: Subset to DataArray if needed
# or ds_anom_interpolated['varname'] if it's a Dataset

# Step 2: Weight by latitude (important for spatial EOFs)
coslat = np.cos(np.deg2rad(time_mean_NPP_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_NPP_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=1)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std(dim='time')  # shape: (mode,)
eofs_normalized = eofs / pcs_std  # broadcast std across spatial dims
# Step 5: Explained variance
variance_fractions = solver.varianceFraction()

print('eofs',np.array(eofs.isel(mode=0)))


for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-0.05,0.05, 40, endpoint=True)
    
    # Contour plot with PlateCarree projection
    levels = np.linspace(-0.1*(10**(-2)), 0.1*(10**(-2)), 40)
    fill = ax.contourf(
    np.array(eofs_normalized.XC),
    np.array(eofs_normalized.YC),
    np.array(eofs_normalized.isel(mode=0)).squeeze(),
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
    )
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    cb.set_label('EOF NPP (mol C/m3/s)', fontsize=20, labelpad=15)  # Add padding to the label for clarity
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("eof_NPP.png")

variance_fractions = solver.varianceFraction()

# Convert to NumPy array (if not already)
variance_array = np.array(variance_fractions)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, len(variance_array[:15])+1), variance_array[:15] * 100, color='skyblue', edgecolor='k')
plt.xlabel('EOF Mode', fontsize=10)
plt.ylabel('Explained Variance (%)', fontsize=10)
plt.title('Explained Variance by EOF Mode', fontsize=12)
plt.xticks(np.arange(1, len(variance_array[:15])+1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("variance_NPP.png")






data_NCP = xr.open_dataset("/projects/CDEUTSCH/DATA/NCP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
#data_NCP = convert_0_360_to_neg180_180(data_NCP)
data_NCP = data_NCP.where(data_NCP.Z >= -100, drop=True).mean('Z')
print('data_NCP',data_NCP)

for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 0.3*(10**(-8)), 40, endpoint=True)
    vmin, vmax = 0, 0.3*(10**(-8))
    # Contour plot with PlateCarree projection
    levels = np.linspace(0, 0.3e-8, 40)
    fill = ax.contourf(
    np.array(data_NCP.XC),
    np.array(data_NCP.YC),
    np.array(data_NCP.BLGNCP.mean('time')).squeeze(),
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
    ) 
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    cb.set_label('NCP (mol C/m3/s)', fontsize=20, labelpad=15)  # Add padding to the label for clarity
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("NCP.png")

time_mean_NCP_clim = data_NCP.BLGNCP.groupby('time.month').mean(dim='time',skipna=True)
time_mean_NCP_anom = data_NCP.BLGNCP.groupby('time.month')-time_mean_NCP_clim
time_mean_NCP_anom=detrend_dim(time_mean_NCP_anom,dim='time')
time_mean_NCP_anom=time_mean_NCP_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_NCP_anom2=time_mean_NCP_anom/time_mean_NCP_anom.std()
time_mean_NCP_anom2=time_mean_NCP_anom2.coarsen(XC=16, boundary="pad").mean()

from eofs.xarray import Eof

# Assume ds_anom_interpolated is a DataArray with dims ('time', 'lat', 'lon')
# Step 1: Subset to DataArray if needed
# or ds_anom_interpolated['varname'] if it's a Dataset

# Step 2: Weight by latitude (important for spatial EOFs)
coslat = np.cos(np.deg2rad(time_mean_NCP_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_NCP_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=1)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized = eofs / pcs_std  # broadcast std across spatial dims
pcs_normalized = pcs * pcs_std  # broadcast std across spatial dims
print('np.array(eofs_normalized.isel(mode=0))',pcs_normalized)
# Step 5: Explained variance
variance_fractions = solver.varianceFraction()


for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-0.05,0.05, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    levels = np.linspace(-10**(-4),10**(-4), 40)
    fill = ax.contourf(
    np.array(eofs_normalized.XC),
    np.array(eofs_normalized.YC),
    np.array(eofs_normalized.isel(mode=0)).squeeze(),
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
    )
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    cb.set_label('EOF NCP (mol C/m3/s)', fontsize=20, labelpad=15)  # Add padding to the label for clarity
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("eof_NCP.png")

import matplotlib.pyplot as plt
import numpy as np

# Get the explained variance for each mode
variance_fractions = solver.varianceFraction()

# Convert to NumPy array (if not already)
variance_array = np.array(variance_fractions)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, len(variance_array)+1), variance_array * 100, color='skyblue', edgecolor='k')
plt.xlabel('EOF Mode', fontsize=10)
plt.ylabel('Explained Variance (%)', fontsize=10)
plt.title('Explained Variance by EOF Mode', fontsize=12)
plt.xticks(np.arange(1, len(variance_array)+1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("variance_NCP.png")



correlation_map = xr.corr(time_mean_THETA_anom2, time_mean_NPP_anom2, dim='time')
print('correlation_map',correlation_map)

correlation_map2 = xr.corr(time_mean_THETA_anom2, time_mean_NCP_anom2, dim='time')


for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-0.6, 0.6, 40, endpoint=True)
    levels = np.linspace(-0.6,0.6, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), np.array(correlation_map).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("correlation_NPP_THETAO.png")

for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map2.XC), np.array(correlation_map2.YC), np.array(correlation_map2).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #cb.set_label('NPP (mol C/m3/s)', fontsize=20, labelpad=15)  # Add padding to the label for clarity

    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("correlation_NCP_THETAO.png")


correlation_map = xr.corr(time_mean_irris_anom2, time_mean_NPP_anom2, dim='time')
for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-0.8, 0.8, 40, endpoint=True)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), np.array(correlation_map).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("correlation_NPP_irris.png")

data_TAUX = xr.open_dataset("/projects/CDEUTSCH/DATA/oceTAUX_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XG'].attrs
    if (_ds['XG'].min() >= 0) and (_ds['XG'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XG'] = xr.where(_ds['XG'] > 180, _ds['XG'] - 360, _ds['XG'])
        _ds = _ds.sortby('XG')
    return _ds

# Apply the conversion to your dataset
data_TAUX = convert_0_360_to_neg180_180(data_TAUX)


data_TAUY = xr.open_dataset("/projects/CDEUTSCH/DATA/oceTAUY_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")



def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
data_TAUY = convert_0_360_to_neg180_180(data_TAUY)

print('data_TAUY',data_TAUY)

print('data_TAUX',data_TAUX)

dx=(2*np.pi)/360
dy=(2*np.pi)/360

def div_4pt_xr(U, V):
    """
    POP stencil operator for divergence
    using xarray
    """
    #U_at_lat_t = U + U.roll(lat=-1, roll_coords=False)  # avg U in y
    dUdx = U.roll(XG=-1, roll_coords=False) - U.roll(XG=1, roll_coords=False)  # dU/dx
    #V_at_lon_t = V + V.roll(lon=-1, roll_coords=False)  # avg V in x
    dVdy = V.roll(YG=-1, roll_coords=False) - V.roll(YG=1, roll_coords=False)  # dV/dy
    return dUdx,dVdy

def z_curl_xr(U, V, dx, dy, lat_wsc):
    """
    xr based
    """
    R = 6413 * (10 ** 3)
    dcos = np.cos(np.deg2rad(lat_wsc))  # Ensure positive value for cosine
    const2 = 1/ (R * (dcos * dcos))
    v = 0.5 * V * dy * dcos
    u = 0.5 * U * dx * dcos
    Vdx, Udy = div_4pt_xr(v, u)
    zcurl = (const2 *(Vdx - dcos * Udy)) / (dx * dy)

    # Adjust sign in the southern hemisphere
    #southern_hemisphere = lat_wsc < 0
    #zcurl[southern_hemisphere] *= -1

    return zcurl, Udy, Vdx,dcos

# Get the 1D lat/lon arrays
lat = data_TAUY['YG']
lon = data_TAUX['XG']

# Create 2D lat-lon meshgrid using xarray broadcasting
lon2d, lat2d = xr.broadcast(lon, lat)  # Each will now have shape (YG, XG)


lat_wsc=lat2d
print('lat_wsc',lat_wsc.shape)
#zcurl, Udy, Vdx,dcos = z_curl_xr(data_TAUX.oceTAUX, data_TAUY.oceTAUY, dx, dy, lat_wsc)

data_MLD = xr.open_dataset("/projects/CDEUTSCH/DATA/MLD_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

print('data_MLD',data_MLD)
# Apply the conversion to your dataset
#data_MLD = convert_0_360_to_neg180_180(data_MLD)
#data_MLD = data_MLD.where(data_MLD.Z >= -100, drop=True).mean('Z')
time_mean_MLD_clim = data_MLD.BLGMLD.groupby('time.month').mean(dim='time',skipna=True)
time_mean_MLD_anom = data_MLD.BLGMLD.groupby('time.month')-time_mean_MLD_clim
time_mean_MLD_anom=detrend_dim(time_mean_MLD_anom,dim='time')
time_mean_MLD_anom=time_mean_MLD_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_MLD_anom2=time_mean_MLD_anom/time_mean_MLD_anom.std()
time_mean_MLD_anom2=time_mean_MLD_anom2.coarsen(XC=16, boundary="pad").mean()


correlation_map = xr.corr(time_mean_MLD_anom2, time_mean_NPP_anom2, dim='time')
print('correlation_map',time_mean_MLD_anom2)
print('correlation_map',time_mean_NPP_anom2)


for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), np.array(correlation_map).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("correlation_NPP_MLD.png")

correlation_map2 = xr.corr(time_mean_MLD_anom2, time_mean_NCP_anom2, dim='time')
print('correlation_map',correlation_map)

for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map2.XC), np.array(correlation_map2.YC), np.array(correlation_map2).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("correlation_NCP_MLD.png")




correlation_map = xr.corr(time_mean_SALT_anom2, time_mean_NPP_anom2, dim='time')
correlation_map2 = xr.corr(time_mean_SALT_anom2, time_mean_NCP_anom2, dim='time')

for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map2.XC), np.array(correlation_map2.YC), np.array(correlation_map2).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
                                                  
plt.savefig("correlation_NCP_SALT.png")





for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), np.array(correlation_map).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("correlation_NPP_SALT.png")


correlation_map = xr.corr(time_mean_SIArea_anom2, time_mean_NPP_anom2, dim='time')
for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), np.array(correlation_map).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

plt.savefig("correlation_NPP_SiArea.png")


correlation_map = xr.corr(time_mean_TRAC04_anom2, time_mean_NPP_anom2, dim='time')
correlation_map2 = xr.corr(time_mean_TRAC04_anom2, time_mean_NCP_anom2, dim='time')


for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), np.array(correlation_map).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

plt.savefig("correlation_NPP_TRAC04.png")




correlation_map = xr.corr(time_mean_TRAC06_anom2, time_mean_NPP_anom2, dim='time')
correlation_map2 = xr.corr(time_mean_TRAC06_anom2, time_mean_NCP_anom2, dim='time')

for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map2.XC), np.array(correlation_map2.YC), np.array(correlation_map2).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

plt.savefig("correlation_NCP_TRAC06.png")

for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), np.array(correlation_map).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

plt.savefig("correlation_NPP_TRAC06.png")


correlation_map2 = xr.corr(time_mean_SSH_anom2, time_mean_NPP_anom2, dim='time')

for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-0.8,0.8, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map2.XC), np.array(correlation_map2.YC), np.array(correlation_map2).squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

plt.savefig("correlation_NPP_SSH.png")














#########################################3 predicting the pc



lags = range(20)
persistence = np.zeros((37*135,20))
longitude = np.arange(-180, 180, 1)
time_mean_NPP_anom2 = time_mean_NPP_anom2.values
time_mean_NPP_anom2=time_mean_NPP_anom2.reshape(132,37*135)

time_mean_NCP_anom2 = time_mean_NCP_anom2.values
time_mean_NCP_anom2=time_mean_NCP_anom2.reshape(132,37*135)

import statsmodels.api as sm
for j in range(37*135):
    series = time_mean_NPP_anom2[:, j]

    # Skip constant series to avoid NaNs
    if np.std(series) == 0:
        persistence[j, :] = 0  # Fill with zero for constant series
        continue

    # Compute autocorrelation and handle NaNs
    acorr = sm.tsa.acf(series, nlags=len(lags) - 1, missing="drop")
    persistence[j, :] = np.nan_to_num(acorr)  # Replace NaNs with zero

persistence=persistence.reshape(37,135,20)
for i in range(1,20,6):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)
    levels = np.linspace(-1,1, 40)
    # Contour plot with PlateCarree projection
    fill = ax.contourf(np.array(correlation_map.XC), np.array(correlation_map.YC), persistence[:,:,i].squeeze(), levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree())
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    plt.savefig("persistence_NPP_"+str(i)+".png")

lon1=np.array(correlation_map.XC)
lat1=np.array(correlation_map.YC)


import statsmodels.api as sm
persistence = np.zeros((37*135,20))
for j in range(37*135):
    series = time_mean_NCP_anom2[:, j]

    # Skip constant series to avoid NaNs
    if np.std(series) == 0:
        persistence[j, :] = 0  # Fill with zero for constant series
        continue

    # Compute autocorrelation and handle NaNs
    acorr = sm.tsa.acf(series, nlags=len(lags) - 1, missing="drop")
    persistence[j, :] = np.nan_to_num(acorr)  # Replace NaNs with zero

persistence=persistence.reshape(37,135,20)
for i in range(1,20,6):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    levels = np.linspace(-1, 1, 40)
    lon = lon1.squeeze()
    lat = lat1.squeeze()
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    fill = ax.contourf(
        lon2d,
        lat2d,
        persistence[:, :, i].squeeze(),
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
    )

    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    cb.ax.tick_params(labelsize=14)
    cb.set_label('Correlation', fontsize=16)


    plt.savefig("persistence_NCP_"+str(i)+".png")


def div_4pt_xr(U, V):
    """
    POP stencil operator for divergence using xarray
    Assumes dimensions (YC, XC) or (time, YC, XC)
    """
    dUdx = U.roll(XG=-1, roll_coords=False) - U.roll(XG=1, roll_coords=False)
    dVdy = V.roll(YC=-1, roll_coords=False) - V.roll(YC=1, roll_coords=False)
    return dUdx, dVdy


def z_curl_xr(U, V, dx, dy, lat_wsc):
    """
    Compute z-curl of wind stress using a POP-like stencil and cosine latitude correction.

    Parameters:
    - U, V: DataArrays with dims (time, YC, XC)
    - dx, dy: grid spacing in radians
    - lat_wsc: 1D array of latitudes (deg) matching YC

    Returns:
    - zcurl: DataArray with dims (time, YC, XC)
    """
    R = 6413e3  # Earth radius [m]
    dcos = np.cos(np.deg2rad(lat_wsc))
    const2 = 1 / (R * dcos**2)

    # Wrap as DataArray
    dcos_da = xr.DataArray(dcos, coords={"YC": U.coords["YC"]}, dims=["YC"])
    const2_da = xr.DataArray(const2, coords={"YC": U.coords["YC"]}, dims=["YC"])

    # Broadcast over full (time, YC, XC)
    dcos_3d = dcos_da.broadcast_like(U)
    const2_3d = const2_da.broadcast_like(U)

    u = 0.5 * U * dx * dcos_3d
    v = 0.5 * V * dy * dcos_3d

    dVdx, dUdy = div_4pt_xr(v, u)

    zcurl = (const2_3d * (dVdx - dcos_3d * dUdy)) / (dx * dy)
    zcurl.name = "wind_stress_curl"

    return zcurl

def compute_zcurl_with_time(U, V, dx, dy, lat_wsc):
    """
    Compute z-curl across all time steps and return (time, YC, XC).
    """
    zcurl_list = []

    for t in range(U.sizes["time"]):
        zcurl_t = z_curl_xr(U.isel(time=t), V.isel(time=t), dx, dy, lat_wsc)
        zcurl_t = zcurl_t.expand_dims(time=[U.time[t].values])
        zcurl_list.append(zcurl_t)

    zcurl_all = xr.concat(zcurl_list, dim="time")
    return zcurl_all

lat = data_TAUY['YG']
lon = data_TAUX['XG']

dx = dy = (2 * np.pi) / 360  # degrees to radians
lat_wsc = data_TAUX.YC.values

zcurl = compute_zcurl_with_time(
    data_TAUX.oceTAUX,
    data_TAUY.oceTAUY.interp(YG=data_TAUX.YC, XC=data_TAUX.XG),
    dx, dy,
    data_TAUX.YC.values
)


time_mean_zcurl_clim = zcurl.groupby('time.month').mean(dim='time',skipna=True)
time_mean_zcurl_anom = zcurl.groupby('time.month')-time_mean_zcurl_clim
time_mean_zcurl_anom=detrend_dim(time_mean_zcurl_anom,dim='time')
time_mean_zcurl_anom=time_mean_zcurl_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_zcurl_anom2=time_mean_zcurl_anom/time_mean_zcurl_anom.std()
time_mean_zcurl_anom2=time_mean_zcurl_anom2.coarsen(XG=16, boundary="pad").mean()

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Reshape and Stack Inputs
# ---------------------------------------------------
time_mean_MLD_anom2 = time_mean_MLD_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_SSH_anom2 = time_mean_SSH_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_SALT_anom2 = time_mean_SALT_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_THETA_anom2 = time_mean_THETA_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_TRAC04_anom2 = time_mean_TRAC04_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_TRAC06_anom2 = time_mean_TRAC06_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_SIArea_anom2=time_mean_SIArea_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_irris_anom2=time_mean_irris_anom2.values.reshape(132, -1).astype(np.float32)
time_mean_zcurl_anom2=time_mean_zcurl_anom2.values.reshape(132, -1).astype(np.float32)

time_mean_NPP_anom2 = time_mean_NPP_anom2.reshape(132, -1).astype(np.float32)
time_mean_NCP_anom2 = time_mean_NCP_anom2.reshape(132, -1).astype(np.float32)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- Parameters ---


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import numpy as np

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from joblib import Parallel, delayed

def sliding_windows(data_X, data_Y, seq_length, lead_time):
    X_windows, y_windows = [], []
    for i in range(data_X.shape[0] - seq_length - lead_time):
        X_windows.append(data_X[i:i + seq_length].flatten())
        y_windows.append(data_Y[i + seq_length + lead_time])
    return np.array(X_windows), np.array(y_windows)

# Correlation computation function
def compute_corr(yt, yp):
    return pearsonr(yt, yp)[0] if np.std(yt) > 0 else np.nan


# Parameters
seq_length = 12
max_lead_time = 20
train_size_ratio = 0.7

# Stack predictors and targets
X = np.concatenate((
    time_mean_MLD_anom2, time_mean_SSH_anom2, time_mean_SIArea_anom2,
    time_mean_THETA_anom2, time_mean_TRAC04_anom2, time_mean_TRAC06_anom2,time_mean_irris_anom2,time_mean_zcurl_anom2
), axis=1)

io.savemat('X_anomalies.mat', {'X': X})

# Y: shape (time, n_targets)
valid_targets = ~np.isnan(X).all(axis=0)  # True for targets that are NOT all-NaN over time
X = X[:, valid_targets]             # Keep only valid target columns




Y = np.concatenate((time_mean_NPP_anom2, time_mean_NCP_anom2), axis=1)
Y=Y.reshape(132,4995*2)

io.savemat('Y_anomalies.mat', {'Y': Y})

Y1=Y[:,:4995]
Y2=Y[:,4995:]
# Y: shape (time, n_targets)


n_targets = Y.shape[1]


# 3. Create meshgrid
# Create 2D meshgrid (standard lat-lon layout)
nlat, nlon = 37, 135
ngrids = nlat * nlon
seq_length = 12
max_lead_time = 20
train_size_ratio = 0.7

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from joblib import Parallel, delayed

# Custom correlation scorer for GridSearchCV
def corr_scorer(y_true, y_pred):
    return np.mean([pearsonr(y_true[:, i], y_pred[:, i])[0] if np.std(y_true[:, i]) > 0 else 0 for i in range(y_true.shape[1])])

correlation_scorer = make_scorer(corr_scorer, greater_is_better=True)

def compute_corr(yt, yp):
    return pearsonr(yt, yp)[0] if np.std(yt) > 0 else np.nan

# Skill matrix init
skills_matrix_total = np.zeros((max_lead_time, n_targets))

# Grid for hyperparameter search
param_grid = {
    "alpha": [1e-2, 1e-1, 1],
    "kernel": ["rbf", "poly"],
    "gamma": [1e-3, 1e-2]  # only relevant for rbf/poly
}

param_grid = [
{"kernel": ["poly"], "alpha": [1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "degree": [2, 3, 4]},  # optionally include degree
{"kernel": ["rbf"], "alpha": [1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "gamma": [1e-3, 1e-2, 1e-1]},
{"kernel": ["linear"], "alpha": [1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
{"kernel": ["sigmoid"], "alpha": [1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "gamma": [1e-3, 1e-2, 1e-1], "coef0": [0, 1]},
]

# Loop over lead times
for k in range(1):

  valid_targets = ~np.isnan(X).all(axis=0)  # True for targets that are NOT all-NaN over time
  X = X[:, valid_targets]
  if k==0:
     Y=Y1
  if k==1:
     Y=Y2
  Y=Y.reshape(132,4995)
  valid_targets = ~np.isnan(Y).all(axis=0)  # True for targets that are NOT all-NaN over time
  Y = Y[:, valid_targets]             # Keep only valid target columns
  scaler_X = StandardScaler()
  scaler_Y = StandardScaler()
  train_size = int(train_size_ratio * X.shape[0])
  scaler_X.fit(X[:train_size])
  scaler_Y.fit(Y[:train_size])

  X_scaled = scaler_X.transform(X)
  Y_scaled = scaler_Y.transform(Y)

  n_targets = Y.shape[1]
  skills_matrix_total = np.zeros((max_lead_time, n_targets))
  residual_matrix2=[]
  for lead_time in range(1, max_lead_time + 1):
    print(f"🔁 Processing lead time {lead_time}...")

    X_seq, y_seq = sliding_windows(X_scaled, Y_scaled, seq_length, lead_time)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # Grid search for best Kernel Ridge model across all outputs together
    krr = KernelRidge()
    search = GridSearchCV(krr, param_grid, cv=9, scoring=correlation_scorer, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = scaler_Y.inverse_transform(y_pred)
    y_test = scaler_Y.inverse_transform(y_test)
    # Parallel correlation skill
    skills_matrix_total[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_corr)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
    )
    print('y_test',y_test.shape)
    residual_matrix = np.zeros((y_test.shape[0], n_targets))
    skills = []
    pvals = []
    significances = []

    for i in range(n_targets):
        #r, rcrit, pval = corr_isopersist(np.nan_to_num(y_test[:, i]).flatten(),np.nan_to_num(y_pred[:, i]).flatten())
        #skills.append(r)
        #pvals.append(pval)
        #significances.append(1 if pval <= 0.05 else 0)
        for j in range(y_test.shape[0]):
            residual_matrix[j,i]=y_pred[j,i]- y_test[j,i]
    residual_matrix=np.nanmean(residual_matrix,axis=0)        
    residual_matrix2.append(residual_matrix)
    #significances2.append(significances)    
  
  #significances2=np.array(significances)
  residual_matrix2=np.array(residual_matrix2)
  print('significances2',residual_matrix2.shape)
  print('y_pred_kernel',y_pred.shape) 
  B = ['NPP', 'NCP']
  # 1. Convert lon from [0, 360] to [-180, 180] if needed

  skills_matrix_total=skills_matrix_total.reshape(37,135,20)
  # 3. Create meshgrid
  # Create 2D meshgrid (standard lat-lon layout)
  nlat, nlon = 37, 135
  ngrids = nlat * nlon

  # Create a full skill matrix with NaNs
  full_skills = np.full((max_lead_time, ngrids), np.nan)

  # skills_matrix should be shape (12, n_valid_targets)
  # valid_targets should be shape (4995,)
  full_skills[:, valid_targets] = skills_matrix_total.reshape(20,37*135)

  # Reshape to (nlat, nlon, lead_time)
  full_skills = full_skills.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)
  
  for i in range(1,20,6):  # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-1, 1, 40)
      lon = lon1.squeeze()
      lat = lat1.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_skills[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=14)
      cb.set_label('Correlation', fontsize=16)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("kernel_regression_"+str(i)+"_"+str(B[k])+".png")
      
   
  residual_matrix2=residual_matrix2.reshape(37,135,20)
  # Create 2D meshgrid (standard lat-lon layout)
  nlat, nlon = 37, 135
  ngrids = nlat * nlon

  # Create a full skill matrix with NaNs
  full_residual = np.full((max_lead_time, ngrids), np.nan)

  # skills_matrix should be shape (12, n_valid_targets)
  # valid_targets should be shape (4995,)
  full_residual[:, valid_targets] =  residual_matrix2.reshape(20,37*135)

  # Reshape to (nlat, nlon, lead_time)
  full_residual =  full_residual.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)
  for i in range(1,20,6):  # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-1, 1, 40)
      lon = lon1.squeeze()
      lat = lat1.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_residual[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=14)
      cb.set_label('Residual', fontsize=16)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("kernel_regression_residual_"+str(i)+"_"+str(B[k])+".png")




# sensitivity


# === Seasonal Sensitivity Analysis: Spring/Summer vs Fall/Winter ===
# Define seasonal indices
months = np.tile(np.arange(12), int(X.shape[0] / 12))  # [0,1,...,11] repeated
spring_summer_indices = np.where((months >= 3) & (months <= 8))[0]  # April–September
fall_winter_indices = np.where((months <= 2) | (months >= 9))[0]    # October–March

print('X_ss',X[spring_summer_indices].shape)

print('Y_fw',X[fall_winter_indices].shape)

# Apply to inputs and targets

# Helper: seasonal correlation skill
def seasonal_skill(X_seq, y_seq, model, scaler_Y):
    y_pred = model.predict(X_seq)
    y_pred = scaler_Y.inverse_transform(y_pred)
    y_true = scaler_Y.inverse_transform(y_seq)
    return np.array([compute_corr(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

# Loop over lead times for both seasonal sets
# Prepare containers for the two seasonal skill matrices

seasonal_skill_matrix_ss = None
seasonal_skill_matrix_fw = None
months = np.tile(np.arange(12), int(X.shape[0] / 12))  # [0,1,...,11] repeated
spring_summer_indices = np.where((months >= 3) & (months <= 8))[0]  # April–September
X_ss = X[spring_summer_indices]
Y_ss = Y[spring_summer_indices]

train_size_ratio = 0.7
train_size = int(train_size_ratio * X_ss.shape[0])


scaler_X_ss = StandardScaler().fit(X_ss[:train_size])
scaler_Y_ss = StandardScaler().fit(Y_ss[:train_size])
X_ss_scaled = scaler_X_ss.transform(X_ss)
Y_ss_scaled = scaler_Y_ss.transform(Y_ss)

fall_winter_indices = np.where((months <= 2) | (months >= 9))[0]    # October–March
X_fw = X[fall_winter_indices]
Y_fw = Y[fall_winter_indices]
train_size_ratio = 0.7
train_size = int(train_size_ratio * X_fw.shape[0])


scaler_X_fw = StandardScaler().fit(X_fw[:train_size])
scaler_Y_fw = StandardScaler().fit(Y_fw[:train_size])
X_fw_scaled = scaler_X_fw.transform(X_fw)
Y_fw_scaled = scaler_Y_fw.transform(Y_fw)


max_lead_time=6

for season_name, X_season, Y_season, scaler_Y in [
    ("SpringSummer", X_ss_scaled, Y_ss_scaled, scaler_Y_ss),
    ("FallWinter", X_fw_scaled, Y_fw_scaled, scaler_Y_fw)
]:

    seasonal_skill_matrix = np.zeros((max_lead_time, n_targets))
    print(f"\n📅 Seasonal Skill Evaluation: {season_name}")
    for lead_time in range(1, max_lead_time + 1):
        X_seq, y_seq = sliding_windows(X_season, Y_season, seq_length, lead_time)
        
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        # Grid search for best Kernel Ridge model across all outputs together
        krr = KernelRidge()
        search = GridSearchCV(krr, param_grid, cv=9, scoring=correlation_scorer, n_jobs=-1)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred = scaler_Y.inverse_transform(y_pred)
        y_test = scaler_Y.inverse_transform(y_test)

        seasonal_skill_values = Parallel(n_jobs=-1)(
            delayed(compute_corr)(y_pred[:, i], y_test[:, i]) for i in range(y_test.shape[1])
        )
        seasonal_skill_matrix[lead_time - 1] = seasonal_skill_values
        print(f"Lead time {lead_time:2d}: Mean Correlation ({season_name}) = {np.nanmean(seasonal_skill_values):.3f}")
    
    seasonal_skill_matrix = seasonal_skill_matrix.reshape(6, 37 * 135)
    if season_name == "SpringSummer":
        seasonal_skill_matrix_ss = seasonal_skill_matrix.copy()
    else:
        seasonal_skill_matrix_fw = seasonal_skill_matrix.copy()

    # Create 2D meshgrid (standard lat-lon layout)
    nlat, nlon = 37, 135
    ngrids = nlat * nlon

    # Create a full skill matrix with NaNs
    full_skills = np.full((max_lead_time, ngrids), np.nan)

    # skills_matrix should be shape (12, n_valid_targets)
    # valid_targets should be shape (4995,)
    full_skills[:, valid_targets] = seasonal_skill_matrix.reshape(6,37*135)

    # Reshape to (nlat, nlon, lead_time)
    full_skills = full_skills.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)
    for i in range(1,3,6):  # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-1, 1, 40)
      lon = lon1.squeeze()
      lat = lat1.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_skills[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=14)
      cb.set_label('Correlation', fontsize=16)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

      plt.savefig(f"kernel_regression_{season_name}_{i}_{B[k]}.png")    

full_skills_ss=seasonal_skill_matrix_ss
full_skills_fw=seasonal_skill_matrix_fw
# ---------------------------------------------
# 0.  House‑keeping
# ---------------------------------------------
nlat, nlon   = 37, 135
ngrids       = nlat * nlon
max_lead_time = full_skills_ss.shape[0]          # 6

lon_vec = lon1.squeeze()      # shape (135,)
lat_vec = lat1.squeeze()      # shape (37,)

lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)   # shapes (37,135)
lon_flat = lon2d.ravel()                       # (4995,)
lat_flat = lat2d.ravel()                       # (4995,)

# ---------------------------------------------
# 1.  Build 1‑D masks for each basin
# ---------------------------------------------
sector_defs = {
    # wrap‑around (300 → 360) ∪ (0 → 20)
    "Weddell":                  dict(lonmin=300, lonmax=20),
    # unchanged
    "Indian":                   dict(lonmin=20,  lonmax=90),
    "West Pacific":             dict(lonmin=90,  lonmax=160),
    # Ross Sea band (no wrap in 0…360)
    "Ross":                     dict(lonmin=160, lonmax=230),
    # Amundsen–Bellingshausen (no wrap)
    "Amundsen–Bellingshausen":  dict(lonmin=230, lonmax=300),
    # whole belt
    "Pan–Antarctic":            dict(lonmin=0,   lonmax=360)
}

import numpy as np
import matplotlib.pyplot as plt

# 1) build your flat masks once:
def lon_mask(lon, lonmin, lonmax):
    if lonmin < lonmax:
        return (lon >= lonmin) & (lon < lonmax)
    else:
        # wrap case
        return (lon >= lonmin) | (lon < lonmax)

sector_masks_flat = {}
for name, sd in sector_defs.items():
    m_lon = lon_mask(lon_flat, sd["lonmin"], sd["lonmax"])
    m_lat = (lat_flat <= -30)          # or <= -50 if you prefer
    sector_masks_flat[name] = m_lon & m_lat   # 1D boolean array

# 2) compute per‐basin skill curves:
leads = np.arange(1, full_skills_ss.shape[0] + 1)

skill_SS = {}    # will hold arrays of length n_lead
skill_FW = {}

for basin, mask in sector_masks_flat.items():
    skill_SS[basin] = np.nanmean(full_skills_ss[:, mask], axis=1)
    skill_FW[basin] = np.nanmean(full_skills_fw[:, mask], axis=1)

# 3) plot each basin in its own figure:
style = {
    "Spring–Summer": dict(color="tab:red",   marker="o", linestyle="-",  label="Spring–Summer"),
    "Fall–Winter":   dict(color="tab:blue",  marker="s", linestyle="--", label="Fall–Winter"),
}

for basin in sector_defs.keys():  # in insertion order
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(leads, skill_SS[basin], **style["Spring–Summer"])
    ax.plot(leads, skill_FW[basin], **style["Fall–Winter"])
    i#ax.set_title(f"NPP skill · {basin}")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("Mean correlation")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    fname = f"kernel_regression_{basin.replace(' ','_').replace('–','-')}_both_seasons.png"
    plt.savefig(fname, dpi=300)
    plt.show()


### sensitivity

# Create 2D meshgrid (standard lat-lon layout)
nlat, nlon = 37, 135
ngrids = nlat * nlon
max_lead_time = 20

labels = ['Ocean physics', 'Ocean BGC/nutrients (NO3 & Fe)', 'Atmospheric variables: Surface Irradiance and WSC']  # Example sensitivity labels
full_skills_sensitivity_list = []  # Store sensitivity maps
full_skills_sensitivity_list_total=[]

for k in range(3):
    # Construct X with variable removal
    X = np.concatenate((
        time_mean_MLD_anom2,
        time_mean_SSH_anom2,
        time_mean_THETA_anom2,
        time_mean_SIArea_anom2,
        time_mean_TRAC06_anom2,
        time_mean_TRAC04_anom2,
        time_mean_irris_anom2,
        time_mean_zcurl_anom2
    ), axis=1)

    if k == 0:
        X = X[:, :(4995*4)]
    elif k == 1:
        X = X[:, (4995*4):(4995*6)]
    elif k == 2:
        X = X[:, (4995*6):]

    valid_targets = ~np.isnan(X).all(axis=0)
    X_filtered = X[:, valid_targets]

    Y = Y1
    
    Y = Y.reshape(132, 4995)
    valid_targets = ~np.isnan(Y).all(axis=0)
    Y_filtered = Y[:, valid_targets]

    # Normalize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    train_size = int(train_size_ratio * X_filtered.shape[0])

    scaler_X.fit(X_filtered[:train_size])
    scaler_Y.fit(Y_filtered[:train_size])

    X_scaled = scaler_X.transform(X_filtered)
    Y_scaled = scaler_Y.transform(Y_filtered)

    n_targets = Y_filtered.shape[1]
    skills_matrix = np.zeros((max_lead_time, n_targets))

    for lead_time in range(1, max_lead_time + 1):
        print(f"🔁 Processing lead time {lead_time} for {labels[k]}...")

        X_seq, y_seq = sliding_windows(X_scaled, Y_scaled, seq_length, lead_time)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        krr = KernelRidge()
        search = GridSearchCV(krr, param_grid, cv=9, scoring=correlation_scorer, n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_pred = scaler_Y.inverse_transform(y_pred)
        y_test = scaler_Y.inverse_transform(y_test)

        # Compute correlation skill for each target
        skills_matrix[lead_time - 1] = Parallel(n_jobs=-1)(
            delayed(compute_corr)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
        )

    # --- Compute skill sensitivity
    skills_matrix2 = skills_matrix.reshape(max_lead_time, nlat, nlon)

    # Fill valid target locations
    full_skills = np.full((max_lead_time, ngrids), np.nan)
    full_skills[:, valid_targets] = skills_matrix2.reshape(max_lead_time, -1)

    # Reshape to (nlat, nlon, lead_time)
    full_skills = full_skills.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)

    # Append to the list for later use
    full_skills_sensitivity_list.append(full_skills.copy())
    

    skills_matrix2_total = skills_matrix_total.reshape(max_lead_time, nlat, nlon)

    # Fill valid target locations
    full_skills_total = np.full((max_lead_time, ngrids), np.nan)
    full_skills_total[:, valid_targets] = skills_matrix2_total.reshape(max_lead_time, -1)

    # Reshape to (nlat, nlon, lead_time)
    full_skills_total = full_skills_total.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)

    # Append to the list for later use
    full_skills_sensitivity_list_total.append(full_skills_total.copy())




    for i in range(1,20,6):  # for lag = 1, 7
        
      data_list = [full_skills[:, :, i], full_skills_total[:, :, i]]
      title_list = ['Skill (Sensitivity)', 'Skill (Baseline)']

      for k, (data, title) in enumerate(zip(data_list, title_list)):
        fig = plt.figure(figsize=(10, 10), dpi=300)  
        ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

        ax.coastlines(resolution='110m')
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        levels = np.linspace(-1, 1, 40)
        lon = lon1.squeeze()
        lat = lat1.squeeze()
        lon2d, lat2d = np.meshgrid(lon, lat)

        cf = ax.contourf(
            lon2d, lat2d, data,
            levels=levels,
            cmap=plt.cm.RdBu_r,
            extend='both',
            transform=ccrs.PlateCarree()
        )
        ax.set_title(f'{title} (Lead Time = {i+1})', fontsize=18)

        cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
        cb.ax.tick_params(labelsize=18)
        cb.set_label('Correlation', fontsize=18)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        plt.savefig("kernel_regression_sensitivity2_"+str(k)+"_"+str(i)+".png")
    
    





nlat, nlon   = 37, 135
ngrids       = nlat * nlon
max_lead_time = 20          # 6

lon_vec = lon1.squeeze()      # shape (135,)
lat_vec = lat1.squeeze()      # shape (37,)

lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)   # shapes (37,135)
lon_flat = lon2d.ravel()                       # (4995,)
lat_flat = lat2d.ravel()                       # (4995,)

# ---------------------------------------------
# 1.  Build 1‑D masks for each basin
# ---------------------------------------------
sector_defs = {
# wrap‑around (300 → 360) ∪ (0 → 20)
"Weddell":                  dict(lonmin=300, lonmax=20),
# unchanged
"Indian":                   dict(lonmin=20,  lonmax=90),
"West Pacific":             dict(lonmin=90,  lonmax=160),
# Ross Sea band (no wrap in 0…360)
"Ross":                     dict(lonmin=160, lonmax=230),
# Amundsen–Bellingshausen (no wrap)
"Amundsen–Bellingshausen":  dict(lonmin=230, lonmax=300),
# whole belt
"Pan–Antarctic":            dict(lonmin=0,   lonmax=360)
}

import numpy as np
import matplotlib.pyplot as plt

# 1) build your flat masks once:
def lon_mask(lon, lonmin, lonmax):
    if lonmin < lonmax:
        return (lon >= lonmin) & (lon < lonmax)
    else:
        # wrap case
        return (lon >= lonmin) | (lon < lonmax)
    
# 1) build your flat masks once:
def lon_mask(lon, lonmin, lonmax):
    if lonmin < lonmax:
        return (lon >= lonmin) & (lon < lonmax)
    else:
        # wrap case
        return (lon >= lonmin) | (lon < lonmax)

sector_masks_flat = {}
for name, sd in sector_defs.items():
    m_lon = lon_mask(lon_flat, sd["lonmin"], sd["lonmax"])
    m_lat = (lat_flat <= -30)          # or <= -50 if you prefer
    sector_masks_flat[name] = m_lon & m_lat   # 1D boolean array


leads = np.arange(1, 20 + 1)
for basin, mask in sector_masks_flat.items():
    fig, ax = plt.subplots(figsize=(6, 4))

    for k, full_sens in enumerate(full_skills_sensitivity_list):
        flat = full_sens.reshape(nlat * nlon, max_lead_time).T
        skill_sens = np.nanmean(flat[:, mask], axis=1)

        flat = full_skills_total.reshape(nlat * nlon, max_lead_time).T
        skill_sens_total = np.nanmean(flat[:, mask], axis=1)

        ax.plot(leads, skill_sens, linestyle="--", marker="s", label=f"Sensitivity {k+1}")
        ax.plot(leads, skill_sens_total, linestyle="--", marker="s", label="Total")

    ax.set_xlabel("Lead time (months)", fontsize=16)
    ax.set_ylabel("Mean correlation", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(0.1, 0.65)
    ax.grid(alpha=0.3)
    #ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"kernel_regression_sensitivity_all_{basin.replace(' ', '_')}.png", dpi=300)
    plt.show()

