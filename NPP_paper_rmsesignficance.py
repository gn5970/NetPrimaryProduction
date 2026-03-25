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
from sklearn.metrics import mean_squared_error

import h5py
import xarray as xr
import numpy as np
import numpy as np
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset
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
import xesmf as xe
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
#time_mean_THETA_anom=time_mean_THETA_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_THETA_anom2=time_mean_THETA_anom/time_mean_THETA_anom.std()
time_mean_THETA_anom=time_mean_THETA_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_THETA_anom=time_mean_THETA_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_THETA_anom2=time_mean_THETA_anom/time_mean_THETA_anom.std()



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
#time_mean_SSH_anom=time_mean_SSH_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_SSH_anom2=time_mean_SSH_anom/time_mean_SSH_anom.std()
time_mean_SSH_anom=time_mean_SSH_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_SSH_anom=time_mean_SSH_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_SSH_anom2=time_mean_SSH_anom/time_mean_SSH_anom.std()

#data_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
data_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")

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
#time_mean_SALT_anom=time_mean_SALT_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_SALT_anom2=time_mean_SALT_anom/time_mean_SALT_anom.std()
time_mean_SALT_anom=time_mean_SALT_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_SALT_anom=time_mean_SALT_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_SALT_anom2=time_mean_SALT_anom/time_mean_SALT_anom.std()

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
#time_mean_SIArea_anom=time_mean_SIArea_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_SIArea_anom2=time_mean_SIArea_anom/time_mean_SIArea_anom.std()
time_mean_SIArea_anom=time_mean_SIArea_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_SIArea_anom=time_mean_SIArea_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_SIArea_anom2=time_mean_SIArea_anom/time_mean_SIArea_anom.std()

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
#data_NO3 = data_NO3.where(data_NO3.Z >= -100, drop=True).mean('Z')
data_NO3 = data_NO3.TRAC04

#data_O2 = convert_0_360_to_neg180_180(data_O2)
#data_irris = data_irris.where(data_irris.Z >= -100, drop=True).mean('Z')


# Step 2: Mask top 100 m
Z_top = data_NO3.Z.where(data_NO3.Z >= -100, drop=True)
data_NO3 = data_NO3.sel(Z=Z_top)


drF = data_NO3.drF  # (Z), vertical cell thickness in m
hFacC = data_NO3.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data_NO3.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_NO3 = (data_NO3 * drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]


print(data_thetao)
print(data_ssh)
print(data_salt)

time_mean_TRAC04_clim = data_NO3.groupby('time.month').mean(dim='time',skipna=True)
time_mean_TRAC04_anom = data_NO3.groupby('time.month')-time_mean_TRAC04_clim
time_mean_TRAC04_anom=detrend_dim(time_mean_TRAC04_anom,dim='time')
#time_mean_TRAC04_anom=time_mean_TRAC04_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_TRAC04_anom2=time_mean_TRAC04_anom/time_mean_TRAC04_anom.std()
time_mean_TRAC04_anom=time_mean_TRAC04_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_TRAC04_anom=time_mean_TRAC04_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_TRAC04_anom2=time_mean_TRAC04_anom/time_mean_TRAC04_anom.std()

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
data_irris = data_irris.BLGIRRIS

#data_O2 = convert_0_360_to_neg180_180(data_O2)
#data_irris = data_irris.where(data_irris.Z >= -100, drop=True).mean('Z')


# Step 2: Mask top 100 m
Z_top = data_irris.Z.where(data_irris.Z >= -100, drop=True)
data_irris = data_irris.sel(Z=Z_top)


drF = data_irris.drF  # (Z), vertical cell thickness in m
hFacC = data_irris.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data_irris.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_irris = (data_irris * drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]



#data_irris = data_irris.where(data_irris.Z >= -100, drop=True).mean('Z')

print('data_O2',data_irris)

time_mean_irris_clim = data_irris.groupby('time.month').mean(dim='time',skipna=True)
time_mean_irris_anom = data_irris.groupby('time.month')-time_mean_irris_clim
time_mean_irris_anom=detrend_dim(time_mean_irris_anom,dim='time')
#time_mean_irris_anom=time_mean_irris_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_irris_anom2=time_mean_irris_anom/time_mean_irris_anom.std()
time_mean_irris_anom=time_mean_irris_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_irris_anom=time_mean_irris_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_irris_anom2=time_mean_irris_anom/time_mean_irris_anom.std()


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
#data_Fe = data_Fe.where(data_Fe.Z >= -100, drop=True).mean('Z')
data_Fe = data_Fe.TRAC06

#data_O2 = convert_0_360_to_neg180_180(data_O2)
#data_irris = data_irris.where(data_irris.Z >= -100, drop=True).mean('Z')


# Step 2: Mask top 100 m
Z_top = data_Fe.Z.where(data_Fe.Z >= -100, drop=True)
data_Fe = data_Fe.sel(Z=Z_top)


drF = data_Fe.drF  # (Z), vertical cell thickness in m
hFacC = data_Fe.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data_Fe.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_Fe = (data_Fe * drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]



print('data_Fe',data_Fe)
time_mean_TRAC06_clim = data_Fe.groupby('time.month').mean(dim='time',skipna=True)
time_mean_TRAC06_anom = data_Fe.groupby('time.month')-time_mean_TRAC06_clim
time_mean_TRAC06_anom=detrend_dim(time_mean_TRAC06_anom,dim='time')
#time_mean_TRAC06_anom=time_mean_TRAC06_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_TRAC06_anom2=time_mean_TRAC06_anom/time_mean_TRAC06_anom.std()
time_mean_TRAC06_anom=time_mean_TRAC06_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_TRAC06_anom=time_mean_TRAC06_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_TRAC06_anom2=time_mean_TRAC06_anom/time_mean_TRAC06_anom.std()

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
#data_NPP = data_NPP.where(data_NPP.Z >= -100, drop=True).mean('Z')
seconds_per_year =12.011*1000*86400  # 31,536,000
data_NPP = data_NPP * seconds_per_year  # Now in molC/m³/year
npp = data_NPP.BLGNPP  # Assuming this is the variable name

# Step 2: Mask top 100 m
Z_top = data_NPP.Z.where(data_NPP.Z >= -100, drop=True)
data_NPP= data_NPP.sel(Z=Z_top)


drF = data_NPP.drF  # (Z), vertical cell thickness in m
hFacC = data_NPP.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data_NPP.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_NPP = (npp * drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]
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
    np.array(data_NPP.mean('time')).squeeze(),
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

time_mean_NPP_clim = data_NPP.groupby('time.month').mean(dim='time',skipna=True)
time_mean_NPP_anom = data_NPP.groupby('time.month')-time_mean_NPP_clim
time_mean_NPP_anom=detrend_dim(time_mean_NPP_anom,dim='time')
#time_mean_NPP_anom=time_mean_NPP_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_NPP_anom2=time_mean_NPP_anom/time_mean_NPP_anom.std()
time_mean_NPP_anom=time_mean_NPP_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_NPP_anom=time_mean_NPP_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_NPP_anom2=time_mean_NPP_anom/time_mean_NPP_anom.std()
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
#time_mean_MLD_anom=time_mean_MLD_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_MLD_anom2=time_mean_MLD_anom/time_mean_MLD_anom.std()
time_mean_MLD_anom=time_mean_MLD_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_MLD_anom=time_mean_MLD_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_MLD_anom2=time_mean_MLD_anom/time_mean_MLD_anom.std()




correlation_map = xr.corr(time_mean_SALT_anom2, time_mean_NPP_anom2, dim='time')
#correlation_map2 = xr.corr(time_mean_SALT_anom2, time_mean_NCP_anom2, dim='time')










#########################################3 predicting the pc



lags = range(20)
persistence = np.zeros((37*135,20))
longitude = np.arange(-180, 180, 1)
time_mean_NPP_anom2 = time_mean_NPP_anom2.values
time_mean_NPP_anom2=time_mean_NPP_anom2.reshape(132,37*135)

#time_mean_NCP_anom2 = time_mean_NCP_anom2.values
#time_mean_NCP_anom2=time_mean_NCP_anom2.reshape(132,37*135)


import statsmodels.api as sm


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

#lat = data_TAUY['YG']
#lon = data_TAUX['XG']

dx = dy = (2 * np.pi) / 360  # degrees to radians
#lat_wsc = data_TAUX.YC.values

# Step 1: Interpolate TAUY from (YG, XC) to (YC, XG)
#TAUY_interp = data_TAUY.oceTAUY.interp(YG=data_TAUX.YC, XC=data_TAUX.XG)

# Step 2: Drop any conflicting coordinates (e.g., YC or XG if they exist already)
#TAUY_interp = TAUY_interp.drop_vars([v for v in ['YG', 'XC', 'YC', 'XG'] if v in TAUY_interp.coords])

# Step 3: Rename dims (not coords!)
#TAUY_interp = TAUY_interp.rename({'YG': 'YC', 'XC': 'XG'})

# Step 4: Rename variables (only if needed)
#TAUY_interp = TAUY_interp.rename({'YG': 'YC', 'XC': 'XG'})

# Step 5: Transpose to match TAUX
#TAUY_interp = TAUY_interp.transpose('time', 'YC', 'XG')

#print("TAUX dims:", data_TAUX.oceTAUX.dims)
#print("TAUY_interp dims:", TAUY_interp.dims)

#zcurl = compute_zcurl_with_time(
#    data_TAUX.oceTAUX,
#    TAUY_interp,
#    dx, dy,
#    data_TAUX.YC.values
#)


############################## new addition
##########################################
#########################################


def div_4pt_xr(U, V):
    """
    POP stencil operator for divergence
    using xarray
    """
    #U_at_lat_t = U + U.roll(lat=-1, roll_coords=False)  # avg U in y
    dUdx = U.roll(XC=-1, roll_coords=False) - U.roll(XC=1, roll_coords=False)  # dU/dx
    #V_at_lon_t = V + V.roll(lon=-1, roll_coords=False)  # avg V in x
    dVdy = V.roll(YC=-1, roll_coords=False) - V.roll(YC=1, roll_coords=False)  # dV/dy
    return dUdx,dVdy


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



target_grid = {"lon": data_TAUY["XC"], "lat": data_TAUX["YC"]}

# --- Regrid UVEL: (YG, XC) → (YC, XC)
regridder_TAUX = xe.Regridder(
    data_TAUX.oceTAUX, target_grid, method="bilinear", periodic=True, reuse_weights=False
)
TAUX_on_tracer = regridder_TAUX(data_TAUX.oceTAUX)

# --- Regrid VVEL: (YC, XG) → (YC, XC)
regridder_TAUY = xe.Regridder(
    data_TAUY.oceTAUY, target_grid, method="bilinear", periodic=True, reuse_weights=False
)
TAUY_on_tracer = regridder_TAUY(data_TAUY.oceTAUY)

zcurl = compute_zcurl_with_time(
    TAUX_on_tracer,
    TAUY_on_tracer,
    dx, dy,
    TAUY_on_tracer.YC.values
)

#####################################################3
#####################################################3





time_mean_zcurl_clim = zcurl.groupby('time.month').mean(dim='time',skipna=True)
time_mean_zcurl_anom = zcurl.groupby('time.month')-time_mean_zcurl_clim
time_mean_zcurl_anom=detrend_dim(time_mean_zcurl_anom,dim='time')
#time_mean_zcurl_anom=time_mean_zcurl_anom.coarsen(YC=16, boundary="pad").mean()
#time_mean_zcurl_anom2=time_mean_zcurl_anom/time_mean_zcurl_anom.std()
time_mean_zcurl_anom=time_mean_zcurl_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_zcurl_anom=time_mean_zcurl_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_zcurl_anom2=time_mean_zcurl_anom/time_mean_zcurl_anom.std()

time1=np.array(time_mean_zcurl_anom2.time)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Reshape and Stack Inputs
# ---------------------------------------------------
lon1=np.array(time_mean_MLD_anom2.XC)
lat1=np.array(time_mean_MLD_anom2.YC)

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
#time_mean_NCP_anom2 = time_mean_NCP_anom2.reshape(132, -1).astype(np.float32)

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




Y = np.concatenate((time_mean_NPP_anom2, time_mean_NPP_anom2), axis=1)
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

def compute_rmse(yt, yp):
    """Compute RMSE between true and predicted values using scikit-learn."""
    if np.isnan(yt).any() or np.isnan(yp).any():
        # Mask out NaNs to avoid computation errors
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[mask], yp[mask]

    return np.sqrt(mean_squared_error(yt, yp)) if len(yt) > 0 else np.nan


# Custom correlation scorer for GridSearchCV
def corr_scorer(y_true, y_pred):
    return np.mean([pearsonr(y_true[:, i], y_pred[:, i])[0] if np.std(y_true[:, i]) > 0 else 0 for i in range(y_true.shape[1])])

correlation_scorer = make_scorer(corr_scorer, greater_is_better=True)

def compute_corr(yt, yp):
    return pearsonr(yt, yp)[0] if np.std(yt) > 0 else np.nan



nlat, nlon = 37, 135
ngrids = nlat * nlon
seq_length = 12
max_lead_time = 20
train_size_ratio = 0.7
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
  skills_lin = np.zeros((max_lead_time, n_targets))
  skills_lin_rmse=np.zeros((max_lead_time, n_targets))
  # Sliding window function

  for lead_time in range(20): #max_lead_time + 1):
    print(f"Processing lead time {lead_time}...")

    X_seq, y_seq = sliding_windows(X_scaled, Y_scaled, seq_length, lead_time)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # Vectorized training across all targets
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred = scaler_Y.inverse_transform(y_pred)
    y_test = scaler_Y.inverse_transform(y_test)
    # Parallel skill computation
    skills_lin[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_corr)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
    )
    skills_lin_rmse[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_rmse)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
    )
  print("✅ All lead times processed. Skill matrix shape:", skills_lin.shape)
  import matplotlib.pyplot as plt
  import cartopy.crs as ccrs
  import numpy as np

  B = ['NPP', 'NCP']
  # 1. Convert lon from [0, 360] to [-180, 180] if needed

  #skills_matrix=skills_matrix.reshape(37,135,20)
  # 3. Create meshgrid
  # Create 2D meshgrid (standard lat-lon layout)
  nlat, nlon = 37, 135
  ngrids = nlat * nlon

  # Create a full skill matrix with NaNs
  full_skills = np.full((max_lead_time, ngrids), np.nan)

  # skills_matrix should be shape (12, n_valid_targets)
  # valid_targets should be shape (4995,)
  full_skills[:, valid_targets] = skills_lin.reshape(20,37*135)

  # Reshape to (nlat, nlon, lead_time)
  full_skills = full_skills.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)

  for i in range(1, max_lead_time + 1,6):  # for lag = 1, 7
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
      cb.ax.tick_params(labelsize=20)
      cb.set_label('Correlation', fontsize=20)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("linear_regression_"+str(i)+"_"+str(B[k])+".png")
   
  full_skills_rmse = np.full((max_lead_time, ngrids), np.nan)

  # skills_matrix should be shape (12, n_valid_targets)
  # valid_targets should be shape (4995,)
  full_skills_rmse[:, valid_targets] = skills_lin_rmse.reshape(20,37*135)

  # Reshape to (nlat, nlon, lead_time)
  full_skills_rmse = full_skills_rmse.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)

  for i in range(1, max_lead_time + 1,6):  # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-1.5, 1.5, 40)
      lon = lon1.squeeze()
      lat = lat1.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_skills_rmse[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=20)
      cb.set_label('RMSE', fontsize=20)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("linear_regression_rmse_"+str(i)+"_"+str(B[k])+".png")













# Skill matrix init
skills_kkr_total = np.zeros((max_lead_time, n_targets))

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
  skills_kkr_total = np.zeros((max_lead_time, n_targets))
  skills_kkr_total_rmse = np.zeros((max_lead_time, n_targets))
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
    skills_kkr_total[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_corr)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
    )
    skills_kkr_total_rmse[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_rmse)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
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
    
    pred_mean = np.nanmean(y_pred, axis=1)
    true_mean = np.nanmean(y_test, axis=1)
    pred_std = np.nanstd(y_pred, axis=1)
    true_std = np.nanstd(y_test, axis=1)
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(time1[-y_pred.shape[0]:],pred_mean, color='tab:blue', linewidth=2)
    ax.plot(time1[-y_pred.shape[0]:],true_mean, color='tab:orange', linestyle='--', linewidth=2)
    ax.fill_between(
    time1[-y_pred.shape[0]:],
    pred_mean - pred_std,
    pred_mean + pred_std,
    color="tab:blue",
    alpha=0.2,
    )
    #ax.set_title(f"Mean NPP prediction")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("NPP anomalies", fontsize=12)
    ax.set_ylim(-0.75, 0.75)
    ax.grid(alpha=0.2)
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)
    #ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    #fname = f"kernel_regression_timeseries_{lead_time}.pdf"
    #plt.savefig(fname, dpi=300)

  #significances2=np.array(significances)
  residual_matrix2=np.array(residual_matrix2)
  print('significances2',residual_matrix2.shape)
  print('y_pred_kernel',y_pred.shape) 
  B = ['NPP', 'NCP']
  # 1. Convert lon from [0, 360] to [-180, 180] if needed

  skills_kkr_total=skills_kkr_total.reshape(37,135,20)
  # 3. Create meshgrid
  # Create 2D meshgrid (standard lat-lon layout)
  nlat, nlon = 37, 135
  ngrids = nlat * nlon

  # Create a full skill matrix with NaNs
  full_skills2 = np.full((max_lead_time, ngrids), np.nan)

  # skills_matrix should be shape (12, n_valid_targets)
  # valid_targets should be shape (4995,)
  full_skills2[:, valid_targets] = skills_kkr_total.reshape(20,37*135)

  # Reshape to (nlat, nlon, lead_time)
  full_skills2 = full_skills2.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)
  
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
        full_skills2[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=14)
      cb.set_label('Correlation', fontsize=16)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      #plt.savefig("kernel_regression_"+str(i)+"_"+str(B[k])+".png")
      
  full_skills_rmse = np.full((max_lead_time, ngrids), np.nan)

  # skills_matrix should be shape (12, n_valid_targets)
  # valid_targets should be shape (4995,)
  full_skills_rmse[:, valid_targets] = skills_kkr_total_rmse
  full_skills_rmse = full_skills_rmse.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)

  for i in range(1,20,6):  # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-1.5, 1.5, 40)
      lon = lon1.squeeze()
      lat = lat1.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_skills_rmse[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=18)
      cb.set_label('RMSE', fontsize=18)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      #plt.savefig("kernel_regression_rmse_"+str(i)+"_inputsequence_"+str(seq_length[ii])+".png")
      #plt.savefig("kernel_regression_rmse_"+str(i)+"_"+str(B[k])+".png")  

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

      levels = np.linspace(-1.2, 1.2, 40)
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
      cb.ax.tick_params(labelsize=18)
      cb.set_label('Residual', fontsize=16)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      #plt.savefig("kernel_regression_residual_"+str(i)+"_"+str(B[k])+".png")
  
  for i in range(1,20,6):  # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-1.2, 1.2, 40)
      lon = lon1.squeeze()
      lat = lat1.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_skills2[:, :, i]-full_skills[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=18)
      cb.set_label('Residual', fontsize=16)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("Supplemental_skill_difference_"+str(i)+"_"+str(B[k])+".png")



# -----------------------------
# Significance of KRR vs Linear (bootstrap + paired tests)
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
import warnings
warnings.filterwarnings("ignore")

def paired_bootstrap_diff(rmse_lin, rmse_krr, n_boot=1000, random_seed=42):
    """
    rmse_lin, rmse_krr: arrays shape (n_leads, n_targets)
    returns:
      mean_diff (n_leads,)
      ci_low (n_leads,), ci_high (n_leads,)  (95% percentile bootstrap)
      mean_rel (n_leads,)  # mean relative improvement (lin - krr)/lin
      rel_ci_low, rel_ci_high
    Bootstrap resamples across targets (columns), preserving pairing.
    """
    rng = np.random.RandomState(random_seed)
    L, N = rmse_krr.shape
    diffs = rmse_lin - rmse_krr            # positive => KRR better (lower RMSE)
    mean_diff = np.nanmean(diffs, axis=1)
    # relative improvement: (RMSE_lin - RMSE_krr) / RMSE_lin
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(rmse_lin != 0, (diffs) / rmse_lin, np.nan)

    mean_rel = np.nanmean(rel, axis=1)

    boot_mean = np.zeros((n_boot, L))
    boot_mean_rel = np.zeros((n_boot, L))

    for b in range(n_boot):
        idx = rng.randint(0, N, N)  # sample targets with replacement
        d_sample = diffs[:, idx]
        boot_mean[b, :] = np.nanmean(d_sample, axis=1)
        r_sample = rel[:, idx]
        boot_mean_rel[b, :] = np.nanmean(r_sample, axis=1)

    ci_low = np.nanpercentile(boot_mean, 2.5, axis=0)
    ci_high = np.nanpercentile(boot_mean, 97.5, axis=0)
    rel_ci_low = np.nanpercentile(boot_mean_rel, 2.5, axis=0)
    rel_ci_high = np.nanpercentile(boot_mean_rel, 97.5, axis=0)

    return mean_diff, ci_low, ci_high, mean_rel, rel_ci_low, rel_ci_high

def paired_tests_across_targets(rmse_lin, rmse_krr):
    """
    Paired t-test and Wilcoxon across targets
    returns arrays of p-values shape (n_leads,)
    """
    L, N = rmse_lin.shape
    p_t = np.ones(L) * np.nan
    p_w = np.ones(L) * np.nan
    for i in range(L):
        a = rmse_lin[i, :]
        b = rmse_krr[i, :]
        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) < 5:
            continue
        try:
            tstat, pvalt = ttest_rel(a[mask], b[mask], nan_policy='omit')
            p_t[i] = pvalt
        except Exception:
            p_t[i] = np.nan
        # wilcoxon requires nonzero differences in some versions; handle carefully
        try:
            wstat, pw = wilcoxon(a[mask], b[mask], zero_method='wilcox', alternative='two-sided', mode='approx')
            p_w[i] = pw
        except Exception:
            p_w[i] = np.nan
    return p_t, p_w

# Example: use your arrays (make sure names match)
# skills_linear_rmse  -> shape (n_leads, n_targets)
# skills_matrix_total_rmse -> shape (n_leads, n_targets)
rmse_lin = skills_lin_rmse.copy()
rmse_krr = skills_kkr_total_rmse.copy()

# Sanity: shapes equal
#assert rmse_lin.shape == rmse_krr.shape, "RMSE arrays must have same shape (leads, targets)."

# Compute bootstrap + paired tests
mean_diff, ci_low, ci_high, mean_rel, rel_ci_low, rel_ci_high = paired_bootstrap_diff(rmse_lin, rmse_krr, n_boot=1000)
p_t, p_w = paired_tests_across_targets(rmse_lin, rmse_krr)

# Prepare summary table
n_leads = rmse_lin.shape[0]
lead_idx = np.arange(1, n_leads+1)

import pandas as pd
summary = pd.DataFrame({
    'lead_months': lead_idx,
    'mean_delta_RMSE': mean_diff,
    'ci_low': ci_low,
    'ci_high': ci_high,
    'mean_rel_reduction': mean_rel,  # fraction (e.g., 0.10 = 10% RMSE reduction)
    'rel_ci_low': rel_ci_low,
    'rel_ci_high': rel_ci_high,
    'p_paired_t': p_t,
    'p_wilcoxon': p_w
})

# Save CSV for the reviewer
summary.to_csv("krr_vs_lin_rmse_summary_by_lead.csv", index=False)
print("Saved summary: krr_vs_lin_rmse_summary_by_lead.csv")

# Quick print for human-readable numbers (round)
pd.options.display.float_format = '{:.4f}'.format
print(summary[['lead_months','mean_delta_RMSE','ci_low','ci_high','mean_rel_reduction','p_paired_t']])

# Plot mean ΔRMSE with 95% CI and significance markers
plt.figure(figsize=(7,4))
plt.fill_between(lead_idx, ci_low, ci_high, color='lightgrey', label='95% bootstrap CI')
plt.plot(lead_idx, mean_diff, marker='o', color='C0', label='mean ΔRMSE (lin - KRR)')
# mark significance where paired t-test p < 0.05
sig_idx = np.where(p_t < 0.05)[0] + 1
plt.scatter(sig_idx, mean_diff[sig_idx-1], marker='*', color='red', s=120, label='paired t p<0.05')
plt.axhline(0, color='k', linewidth=0.7, linestyle='--')
plt.xlabel('Lead (months)')
plt.ylabel('ΔRMSE (linear - KRR)')
plt.title('KRR vs Linear: ΔRMSE by lead (positive => KRR better)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("Supplemental_krr_vs_lin_deltaRMSE_by_lead.png", dpi=300)
print("Saved figure: krr_vs_lin_deltaRMSE_by_lead.png")


