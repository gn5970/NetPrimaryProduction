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
from scipy.stats import linregress

import numpy as np
import random
from scipy import stats


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
#data_thetao = data_thetao.where(data_thetao.maskC == 1)
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


coslat = np.cos(np.deg2rad(time_mean_THETA_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_THETA_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_THETA = eofs # broadcast std across spatial dims
pcs_normalized_THETA = pcs # broadcast std across spatial dims

eofs_normalized_THETA *= -1
pcs_normalized_THETA *= -1
# Step 5: Explained variance
variance_fractions = solver.varianceFraction()
fig = plt.figure(figsize=(12, 4), dpi=300)

plt.plot(pcs_normalized_THETA.time, pcs_normalized_THETA.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 1", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_THETA_timeseries.png")
plt.show()


data_ssh = xr.open_dataset("/projects/CDEUTSCH/DATA/SSH_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_ssh = data_ssh.where(data_ssh.maskInC == 1)
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

coslat = np.cos(np.deg2rad(time_mean_SSH_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_SSH_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_SSH = eofs # broadcast std across spatial dims
pcs_normalized_SSH = pcs # broadcast std across spatial dims

eofs_normalized_SSH *= -1
pcs_normalized_SSH *= -1


for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    # Add coastlines and gridlines
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Set contour levels
    levels = np.linspace(-2e-3, 2e-3, 40)

    # Contourf plot
    fill = ax.contourf(
        np.array(eofs_normalized_SSH.XC),
        np.array(eofs_normalized_SSH.YC),
        np.array(eofs_normalized_SSH.isel(mode=0)).squeeze(),
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
    )

    ax.set_aspect('equal')

    # Colorbar formatting
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    cb.ax.tick_params(labelsize=20)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))  # Scientific with 1 decimal
    cb.set_label('First EOF SSH', fontsize=20, labelpad=15)

    # Final extent (redundant here, but safe)
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

# Save figure
plt.savefig("eof_SSH.pdf", bbox_inches='tight')


# Step 5: Explained variance
variance_fractions = solver.varianceFraction()
fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(pcs_normalized_SSH.time, pcs_normalized_SSH.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 1", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_SSH_timeseries.png")
plt.show()

data_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_salt = data_salt.where(data_salt.maskC == 1)
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

coslat = np.cos(np.deg2rad(time_mean_SALT_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_SALT_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_SALT = eofs # broadcast std across spatial dims
pcs_normalized_SALT = pcs # broadcast std across spatial dim

eofs_normalized_SALT *= -1
pcs_normalized_SALT *= -1
# Step 5: Explained variance
fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(pcs_normalized_SALT.time, pcs_normalized_SALT.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.tight_layout()
plt.savefig("pc1_SALT_timeseries.png")
plt.show()

data_SIArea = xr.open_dataset("/projects/CDEUTSCH/DATA/SIArea_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")

#data_SIArea = data_SIArea.where(data_SIArea.maskInC == 1)
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

coslat = np.cos(np.deg2rad(time_mean_SIArea_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_SIArea_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_SIArea = eofs # broadcast std across spatial dims
pcs_normalized_SIArea = pcs # broadcast std across spatial dims

eofs_normalized_SIArea *= -1
pcs_normalized_SIArea *= -1


for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    # Add coastlines and gridlines
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Set contour levels
    levels = np.linspace(-2.5e-3, 2.5e-3, 40)

    # Contourf plot
    fill = ax.contourf(
        np.array(eofs_normalized_SIArea.XC),
        np.array(eofs_normalized_SIArea.YC),
        np.array(eofs_normalized_SIArea.isel(mode=0)).squeeze(),
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
    )

    ax.set_aspect('equal')

    # Colorbar formatting
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    cb.ax.tick_params(labelsize=20)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))  # Scientific with 1 decimal
    cb.set_label('First EOF SIArea', fontsize=20, labelpad=15)

    # Final extent (redundant here, but safe)
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

# Save figure
plt.savefig("eof_SIArea.pdf", bbox_inches='tight')




# Step 5: Explained variance
fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(pcs_normalized_SIArea.time, pcs_normalized_SIArea.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 1", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_SIArea_timeseries.png")
plt.show()



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
#data_Fe = convert_0_360_to_neg180_180(data_Fe)
#data_Fe= data_Fe.where(data_irris.maskC == 1)
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
time_mean_TRAC04_anom=time_mean_TRAC04_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_TRAC04_anom2=time_mean_TRAC04_anom/time_mean_TRAC04_anom.std()
time_mean_TRAC04_anom2=time_mean_TRAC04_anom2.coarsen(XC=16, boundary="pad").mean()

coslat = np.cos(np.deg2rad(time_mean_TRAC04_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_TRAC04_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series


fig = plt.figure(figsize=(12, 4), dpi=300)
pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_TRAC04 = eofs / pcs_std  # broadcast std across spatial dims
pcs_normalized_TRAC04 = pcs * pcs_std  # broadcast std across spatial dims
# Step 5: Explained variance
plt.plot(pcs_normalized_TRAC04.time, pcs_normalized_TRAC04.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 1", fontsize=18)
plt.xlabel("Time", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_TRAC04_timeseries.png")
plt.show()


data_irris = xr.open_dataset("/projects/CDEUTSCH/DATA/irris_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_irris= data_irris.where(data_irris.maskC == 1)
data_irris = data_irris.BLGIRRIS 
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



print('data_O2',data_irris)

time_mean_irris_clim = data_irris.groupby('time.month').mean(dim='time',skipna=True)
time_mean_irris_anom = data_irris.groupby('time.month')-time_mean_irris_clim
time_mean_irris_anom=detrend_dim(time_mean_irris_anom,dim='time')
time_mean_irris_anom=time_mean_irris_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_irris_anom2=time_mean_irris_anom/time_mean_irris_anom.std()
time_mean_irris_anom2=time_mean_irris_anom2.coarsen(XC=16, boundary="pad").mean()

coslat = np.cos(np.deg2rad(time_mean_irris_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_irris_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=1)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

coslat = np.cos(np.deg2rad(time_mean_irris_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_irris_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_irris = eofs # broadcast std across spatial dims
pcs_normalized_irris = pcs # broadcast std across spatial dims

eofs_normalized_irris *= -1
pcs_normalized_irris *= -1


for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    # Add coastlines and gridlines
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Set contour levels
    levels = np.linspace(-1.5e-3, 1.5e-3, 40)

    # Contourf plot
    fill = ax.contourf(
        np.array(eofs_normalized_irris.XC),
        np.array(eofs_normalized_irris.YC),
        np.array(eofs_normalized_irris.isel(mode=0)).squeeze(),
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
    )

    ax.set_aspect('equal')

    # Colorbar formatting
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    cb.ax.tick_params(labelsize=20)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))  # Scientific with 1 decimal
    cb.set_label('First EOF Irradiance', fontsize=20, labelpad=15)

    # Final extent (redundant here, but safe)
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

# Save figure
plt.savefig("eof_irris.pdf", bbox_inches='tight')


fig = plt.figure(figsize=(12, 4), dpi=300)
pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_irris = eofs # broadcast std across spatial dims
pcs_normalized_irris = pcs   # broadcast std across spatial dims
# Step 5: Explained variance
plt.plot(pcs_normalized_irris.time, pcs_normalized_irris.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 1", fontsize=18)
plt.xlabel("Time", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_irris_timeseries.png")
plt.show()


data_Fe = xr.open_dataset("/projects/CDEUTSCH/DATA/Fe_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_Fe = data_Fe.where(data_Fe.maskC == 1)
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
#data_Fe= data_Fe.where(data_irris.maskC == 1)
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


#data_Fe = data_Fe.where(data_Fe.Z >= -100, drop=True).mean('Z')

print('data_Fe',data_Fe)
time_mean_TRAC06_clim = data_Fe.groupby('time.month').mean(dim='time',skipna=True)
time_mean_TRAC06_anom = data_Fe.groupby('time.month')-time_mean_TRAC06_clim
time_mean_TRAC06_anom=detrend_dim(time_mean_TRAC06_anom,dim='time')
time_mean_TRAC06_anom=time_mean_TRAC06_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_TRAC06_anom2=time_mean_TRAC06_anom/time_mean_TRAC06_anom.std()
time_mean_TRAC06_anom2=time_mean_TRAC06_anom2.coarsen(XC=16, boundary="pad").mean()

coslat = np.cos(np.deg2rad(time_mean_TRAC06_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_TRAC06_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std()  # shape: (mode,)
eofs_normalized_TRAC06=eofs
pcs_normalized_TRAC06=pcs

eofs_normalized_TRAC06 *= -1
pcs_normalized_TRAC06 *= -1

# Step 5: Explained variance
for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    # Add coastlines and gridlines
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Set contour levels
    levels = np.linspace(-1.5e-3, 1.5e-3, 40)

    # Contourf plot
    fill = ax.contourf(
        np.array(eofs_normalized_TRAC06.XC),
        np.array(eofs_normalized_TRAC06.YC),
        np.array(eofs_normalized_TRAC06.isel(mode=0)).squeeze(),
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
    )

    ax.set_aspect('equal')

    # Colorbar formatting
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    cb.ax.tick_params(labelsize=20)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))  # Scientific with 1 decimal
    cb.set_label('First EOF Fe', fontsize=20, labelpad=15)

    # Final extent (redundant here, but safe)
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

# Save figure
plt.savefig("eof_TRAC06.pdf", bbox_inches='tight')
plt.close()


fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(pcs_normalized_TRAC06.time, pcs_normalized_TRAC06.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 1", fontsize=18)
plt.xlabel("Time", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_TRAC06_timeseries.png")
plt.show()


data_NPP = xr.open_dataset("/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000

# Multiply all data variables by seconds per year
data_NPP = data_NPP * seconds_per_year


#data_NPP= data_NPP.where(data_NPP.maskC == 1)
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
#dz = np.abs(np.gradient(data_NPP.Z.values))  # dz in meters

# Broadcast to match the data shape
#dz_xr = xr.DataArray(dz, dims=["Z"], coords={"Z": data_NPP.Z})


#data_NPP = data_NPP.where(data_NPP.Z >= -100, drop=True)
#data_NPP = (data_NPP * dz_xr).sum(dim="Z")
data = xr.open_dataset("/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
data = data * seconds_per_year  # Now in molC/m³/year
npp = data.BLGNPP  # Assuming this is the variable name

# Step 2: Mask top 100 m
Z_top = data.Z.where(data_NPP.Z >= -100, drop=True)
data = data.sel(Z=Z_top)


drF = data.drF  # (Z), vertical cell thickness in m
hFacC = data.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data.rA

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
    
    levels = np.linspace(-17, 17, 40)
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
    cb.set_label('NPP (mol C/m²/year)', fontsize=20, labelpad=15)  
    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("NPP.pdf")

from scipy.stats import linregress

def linear_trend(y, time):
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, _, _, pval, _ = linregress(time[mask], y[mask])
    return slope, pval

# Time in numeric form (e.g., year)
time_num = data_NPP.time.dt.year.values

# Apply across lat-lon
trend, pval = xr.apply_ufunc(
    linear_trend,
    data_NPP,
    time_num,
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[[], []],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float, float]
)

trend_masked = trend.where(pval < 0.05)

print('trend',trend)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# Trend color levels
levels = np.linspace(-0.55, 0.55, 40)

# Filled contour: trend values (masked where not significant)
fill = ax.contourf(
    data_NPP.XC,
    data_NPP.YC,
    trend.squeeze(),
    levels=levels,
    cmap=plt.cm.RdBu_r,
    extend="both",
    transform=ccrs.PlateCarree()
)

# Hatching for regions with *insignificant* trend (p >= 0.05)
significant = (pval <= 0.05)
#ax.contourf(
#    data_NPP.XC,
#    data_NPP.YC,
#    significant.squeeze(),
#    levels=[0.5, 1.5],
#    hatches=["////"],
#    colors="none",
#    transform=ccrs.PlateCarree()
#)
#lon, lat = np.meshgrid(data_NPP.XC, data_NPP.YC)
#lon_sig = lon[significant]
#lat_sig = lat[significant]

# Add small black dots
#ax.scatter(
#    lon_sig, lat_sig,
#    s=3, color='black', marker='.',  # small dots
#    transform=ccrs.PlateCarree(),
#    label='Significant (p ≤ 0.05)'
#)

# Colorbar
cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=16)
cb.set_label('NPP Trend (mol C/m²/year²)', fontsize=18, labelpad=15)
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

# Save
plt.savefig("NPP_trend_significance.pdf", bbox_inches='tight')



mean_npp = data_NPP.mean(dim=['XC', 'YC'])

# 3. Plot the time series (e.g. at a given depth level, or integrated)
#    Here we’ll pick the surface depth (Z=0) as an example
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(data_NPP.time, mean_npp, color='black', linewidth=2)
plt.xlabel("Time", fontsize=18)
plt.ylabel("NPP (mol C/m²/year)", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.grid(True)
plt.tight_layout()
plt.savefig('mean_NPP_time_series.pdf', dpi=300)
plt.show()




print('data_NPP',data_NPP)

time_mean_NPP_clim = data_NPP.groupby('time.month').mean(dim='time',skipna=True)
time_mean_NPP_anom = data_NPP.groupby('time.month')-time_mean_NPP_clim
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
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std(dim='time')  # shape: (mode,)
eofs_normalized = eofs   # broadcast std across spatial dims
pcs_normalized = pcs   # broadcast std across spatial dims


# Step 5: Explained variance
variance_fractions = solver.varianceFraction()

print('eofs',np.array(eofs.isel(mode=0)))

from matplotlib.ticker import ScalarFormatter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# EOF plot loop
for i in range(1):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    # Add coastlines and gridlines
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Set contour levels
    levels = np.linspace(-9e-1, 9e-1, 40)

    # Contourf plot
    fill = ax.contourf(
        np.array(eofs_normalized.XC),
        np.array(eofs_normalized.YC),
        np.array(eofs_normalized.isel(mode=0)).squeeze(),
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
    )

    ax.set_aspect('equal')

    # Colorbar formatting
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    cb.ax.tick_params(labelsize=20)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))  # Scientific with 1 decimal
    cb.set_label('First EOF NPP', fontsize=20, labelpad=15)

    # Final extent (redundant here, but safe)
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

# Save figure
plt.savefig("eof_NPP.pdf", bbox_inches='tight')
plt.close()


import matplotlib.pyplot as plt

# Create a separate figure for PC1
fig = plt.figure(figsize=(12, 4), dpi=300)

plt.plot(pcs_normalized.time, pcs_normalized.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 1", fontsize=18)
plt.xlabel("Time", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_NPP_timeseries.pdf")
plt.show()



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
    levels = np.linspace(-2*(10**(-3)),2*(10**(-3)), 40)
    fill = ax.contourf(
    np.array(eofs_normalized_THETA.XC),
    np.array(eofs_normalized_THETA.YC),
    np.array(eofs_normalized_THETA.isel(mode=0)).squeeze(),
    cmap=plt.cm.RdBu_r,
    levles=levels,
    transform=ccrs.PlateCarree()
    )
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    cb.set_label('EOF THETA', fontsize=20, labelpad=15)  # Add padding to the label for clarity
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("eof_THETA.png")


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
    levels = np.linspace(-3*(10**(-3)),3*(10**(-3)), 40)
    fill = ax.contourf(
    np.array(eofs_normalized_SSH.XC),
    np.array(eofs_normalized_SSH.YC),
    np.array(eofs_normalized_SSH.isel(mode=0)).squeeze(),
    cmap=plt.cm.RdBu_r,
    levels=levels,
    transform=ccrs.PlateCarree()
    )
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    cb.set_label('First EOF SSH', fontsize=20, labelpad=15)  # Add padding to the label for clarity
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("eof_SSH.pdf")


from eofs.xarray import Eof



correlation_map = xr.corr(time_mean_THETA_anom2, time_mean_NPP_anom2, dim='time')
print('correlation_map',correlation_map)


for i in range(1):
    # Degrees of freedom (e.g., number of time steps - 2)
    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed

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
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')
    cb.set_label('Correlation', fontsize=20, labelpad=15)
    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #cb.set_label('NPP (mol C/m3/s)', fontsize=20, labelpad=15)  # Add padding to the label for clarity

    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("correlation_NPP_THETAO.png")

from scipy import stats



correlation_map = xr.corr(time_mean_irris_anom2, time_mean_NPP_anom2, dim='time')
for i in range(1):

    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed

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
    cb.set_label('Correlation', fontsize=20, labelpad=15)
    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
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


coslat = np.cos(np.deg2rad(time_mean_MLD_anom2.YC))
wgts = np.sqrt(coslat)
da_weighted = time_mean_MLD_anom2 * wgts

# Step 3: Initialize EOF solver
solver = Eof(da_weighted)

# Step 4: Get EOFs and PCs
eofs = solver.eofs(neofs=3,eofscaling=2)      # spatial patterns
pcs = solver.pcs(npcs=3, pcscaling=1)          # time series

pcs_std = pcs.std(dim='time')  # shape: (mode,)
eofs_normalized_MLD = eofs   # broadcast std across spatial dims
pcs_normalized_MLD=pcs

eofs_normalized_MLD *= -1
pcs_normalized_MLD *= -1


variance_fractions = solver.varianceFraction()

fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(pcs_normalized_MLD.time, pcs_normalized_MLD.isel(mode=0), color='black', linewidth=2)
#plt.title("Principal Component 1", fontsize=16)

plt.ylabel("Principal Component 1", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("pc1_MLD_timeseries.png")
plt.show()


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
    levels = np.linspace(-0.0015,0.0015, 40)
    fill = ax.contourf(
    np.array(eofs_normalized_MLD.XC),
    np.array(eofs_normalized_MLD.YC),
    np.array(eofs_normalized_MLD.isel(mode=0)).squeeze(),
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree(),
    levels=levels
    )
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    cb.set_label('First EOF MLD', fontsize=20, labelpad=15)  # Add padding to the label for clarity
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("eof_MLD.pdf")





correlation_map = xr.corr(time_mean_MLD_anom2, time_mean_NPP_anom2, dim='time')
print('correlation_map',time_mean_MLD_anom2)
print('correlation_map',time_mean_NPP_anom2)


for i in range(1):
    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed
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
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )
    cb.set_label('Correlation', fontsize=20, labelpad=15)
    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("correlation_NPP_MLD.png")





correlation_map = xr.corr(time_mean_SALT_anom2, time_mean_NPP_anom2, dim='time')




for i in range(1):
    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed

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
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )

    cb.set_label('Correlation', fontsize=20, labelpad=15)
    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("correlation_NPP_SALT.png")


correlation_map = xr.corr(time_mean_SIArea_anom2, time_mean_NPP_anom2, dim='time')
for i in range(1):
    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed

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
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )

    cb.set_label('Correlation', fontsize=20, labelpad=15)
    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("correlation_NPP_SiArea.png")


correlation_map = xr.corr(time_mean_TRAC04_anom2, time_mean_NPP_anom2, dim='time')

for i in range(1):
    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed
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
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )
    cb.set_label('Correlation', fontsize=20, labelpad=15)
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("correlation_NPP_TRAC04.png")




correlation_map = xr.corr(time_mean_TRAC06_anom2, time_mean_NPP_anom2, dim='time')

for i in range(1):
    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed

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
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )
    cb.set_label('Correlation', fontsize=20, labelpad=15)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("correlation_NPP_TRAC06.png")


correlation_map2 = xr.corr(time_mean_SSH_anom2, time_mean_NPP_anom2, dim='time')

for i in range(1):
    df = len(np.array(time_mean_NPP_anom2.time))-2

    # Compute t-values from correlation map (xarray-native)
    t_val = correlation_map * ((df - 2) / (1 - correlation_map ** 2)) ** 0.5
    # Compute two-tailed critical t thresholds
    t90 = stats.t.ppf(1 - 0.05, df - 2)   # p < 0.10 two-tailed
    t95 = stats.t.ppf(1 - 0.025, df - 2)  # p < 0.05 two-tailed
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
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
    t_val.plot.contourf(
    ax=ax,
    levels=[-1 * t95, -1 * t90, t90, t95],
    colors='none',
    hatches=['..', None, None, '..'],
    extend='both',
    add_colorbar=False,
    transform=ccrs.PlateCarree()
    )
    cb.set_label('Correlation', fontsize=20, labelpad=15)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("correlation_NPP_SSH.png")















#########################################3 predicting the pc


time_mean_NPP_anom2 = time_mean_NPP_anom2.values.reshape(132, -1).astype(np.float32)
print('time_mean_NPP_anom2',time_mean_NPP_anom2.shape)
print('time_mean_NPP_anom2',time_mean_NPP_anom2)

import numpy as np
import statsmodels.api as sm
lags = range(1, 21)  # lags 1 to 20
n_time, n_grid = 132, 37 * 135  # Should match your reshape
persistence = np.zeros((n_grid, len(lags)))

# Reshape flattened anomaly fields
time_mean_NPP_anom2 = time_mean_NPP_anom2.reshape(n_time, n_grid)

import numpy as np
import statsmodels.api as sm


for j in range(n_grid):
    series = time_mean_NPP_anom2[:, j]

    # Remove NaNs before computing std and ACF
    valid_series = series[~np.isnan(series)]

    # Skip if all NaNs or constant
    if len(valid_series) == 0 or np.std(valid_series) == 0:
        persistence[j, :] = 0
        continue

    try:
        acorr = sm.tsa.acf(valid_series, nlags=max(lags), fft=True, missing="drop")
        persistence[j, :] = np.nan_to_num(acorr[1:len(lags)+1])
    except Exception as e:
        print(f"⚠️ Skipping index {j} due to error: {e}")
        persistence[j, :] = 0


print("✅ Autocorrelation computed for shape:", persistence.shape)

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
    cb.ax.tick_params(labelsize=20)
    cb.set_label('Correlation', fontsize=20)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    plt.savefig("persistence_NPP_"+str(i)+".png")

lon1=np.array(correlation_map.XC)
lat1=np.array(correlation_map.YC)
time1=np.array(data_MLD.time)


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

# Step 1: Interpolate TAUY from (YG, XC) to (YC, XG)
TAUY_interp = data_TAUY.oceTAUY.interp(YG=data_TAUX.YC, XC=data_TAUX.XG)

# Step 2: Drop any conflicting coordinates (e.g., YC or XG if they exist already)
TAUY_interp = TAUY_interp.drop_vars([v for v in ['YG', 'XC', 'YC', 'XG'] if v in TAUY_interp.coords])

# Step 3: Rename dims (not coords!)
TAUY_interp = TAUY_interp.rename({'YG': 'YC', 'XC': 'XG'})

# Step 4: Rename variables (only if needed)
#TAUY_interp = TAUY_interp.rename({'YG': 'YC', 'XC': 'XG'})

# Step 5: Transpose to match TAUX
TAUY_interp = TAUY_interp.transpose('time', 'YC', 'XG')

print("TAUX dims:", data_TAUX.oceTAUX.dims)
print("TAUY_interp dims:", TAUY_interp.dims)



# Now safely compute zcurl
zcurl = compute_zcurl_with_time(
    data_TAUX.oceTAUX,
    TAUY_interp,
    dx, dy,
    data_TAUX.YC.values
)


#zcurl = compute_zcurl_with_time(
#    data_TAUX.oceTAUX,
#    data_TAUY.oceTAUY.interp(YG=data_TAUX.YC, XC=data_TAUX.XG),
#    dx, dy,
#    data_TAUX.YC.values
#)


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
#time_mean_TRAC01_anom2=time_mean_TRAC01_anom2.values.reshape(132, -1).astype(np.float32)
#time_mean_TRAC02_anom2=time_mean_TRAC02_anom2.values.reshape(132, -1).astype(np.float32)


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
    time_mean_THETA_anom2, time_mean_TRAC04_anom2, time_mean_TRAC06_anom2,time_mean_irris_anom2,time_mean_zcurl_anom2), axis=1)
io.savemat('X_anomalies.mat', {'X': X})

# Y: shape (time, n_targets)
#valid_targets = ~np.isnan(X).all(axis=0)  # True for targets that are NOT all-NaN over time
#X = X[:, valid_targets]             # Keep only valid target columns




Y = time_mean_NPP_anom2
Y=Y.reshape(132,4995)

io.savemat('Y_anomalies.mat', {'Y': Y})

Y1=Y[:,:]

# Y: shape (time, n_targets)


n_targets = Y.shape[1]


# 3. Create meshgrid
# Create 2D meshgrid (standard lat-lon layout)
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
  skills_matrix = np.zeros((max_lead_time, n_targets))

  # Sliding window function

  for lead_time in range(1,20,6): #max_lead_time + 1):
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
    skills_matrix[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_corr)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
    )

  print("✅ All lead times processed. Skill matrix shape:", skills_matrix.shape)

  import matplotlib.pyplot as plt
  import cartopy.crs as ccrs
  import numpy as np

  B = ['NPP', 'NCP']
  # 1. Convert lon from [0, 360] to [-180, 180] if needed

  skills_matrix=skills_matrix.reshape(37,135,20)
  # 3. Create meshgrid
  # Create 2D meshgrid (standard lat-lon layout)
  nlat, nlon = 37, 135
  ngrids = nlat * nlon

  # Create a full skill matrix with NaNs
  full_skills = np.full((max_lead_time, ngrids), np.nan)

  # skills_matrix should be shape (12, n_valid_targets)
  # valid_targets should be shape (4995,)
  full_skills[:, valid_targets] = skills_matrix.reshape(20,37*135)

  # Reshape to (nlat, nlon, lead_time)
  full_skills = full_skills.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)

  for i in range(1):  # for lag = 1, 7
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

full_skills_LR=full_skills
# kernel ridge regression
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
  prediction=[]
  true=[]
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
    y_true = scaler_Y.inverse_transform(y_test)
    # Parallel correlation skill
    skills_matrix_total[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_corr)(y_true[:, i], y_pred[:, i]) for i in range(n_targets)
    )
    print('y_test',y_test.shape)
    residual_matrix = np.zeros((y_test.shape[0], n_targets))
    pvals = []
    significances = []

    for i in range(n_targets):
        #r, rcrit, pval = corr_isopersist(np.nan_to_num(y_test[:, i]).flatten(),np.nan_to_num(y_pred[:, i]).flatten())
        #skills.append(r)
        #pvals.append(pval)
        #significances.append(1 if pval <= 0.05 else 0)
        for j in range(y_test.shape[0]):
            residual_matrix[j,i]=y_pred[j,i]- y_true[j,i]
    residual_matrix=np.nanmean(residual_matrix,axis=0)        
    residual_matrix2.append(residual_matrix)
    skills = skills_matrix_total[lead_time - 1]  # shape: (n_targets,)
  
    high_skill_mask = skills > 0.6

    # Step 3: Filter y_pred and y_true using the mask
    y_pred_high_skill = y_pred[:, high_skill_mask]
    y_true_high_skill = y_true[:, high_skill_mask]

    pred_mean = np.nanmean(y_pred, axis=1)
    true_mean = np.nanmean(y_true, axis=1)
    pred_std = np.nanstd(y_pred, axis=1)
    true_std = np.nanstd(y_true, axis=1)
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
    ax.set_title(f"Mean NPP prediction")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("NPP anomalies", fontsize=12)
    ax.set_ylim(-0.75, 0.75)
    ax.grid(alpha=0.2)
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)
    #ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    fname = f"kernel_regression_timeseries_{lead_time}.pdf"
    plt.savefig(fname, dpi=300)

  #significances2=np.array(significances)
  residual_matrix2=np.array(residual_matrix2)
 
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
      cb.ax.tick_params(labelsize=18)
      cb.set_label('Correlation', fontsize=18)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("kernel_regression_"+str(i)+"_"+str(B[k])+".png")
    
      # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-1, 1, 40)
      lon = lon.squeeze()
      lat = lat.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_skills[:, :, i]-full_skills_LR[:,:,i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=20)
      cb.set_label('Correlation', fontsize=20)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("kernel_regression_difference_"+str(i)+"_"+str(B[k])+".png")
  
      
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
      cb.ax.tick_params(labelsize=20)
      cb.set_label('Residual (mol C/m2/year)', fontsize=20)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
      plt.savefig("kernel_regression_residual_"+str(i)+"_"+str(B[k])+".png")
    
prediction=np.array(prediction)
print('prediction',prediction.shape)


# sensitivity
labels=['MLD','SSH','SIArea','THETA','NO3','Fe','Irradiance','WSC']


for k in range(8):

    if k==0:
       X=X[:,:(4995)*8-(4995)]
    if k==1:
       X1=X[:,:((4995)*8-(4995)*2)]
       X2=X[:,((4995)*7):]
       X=np.concatenate((X1,X2),axis=1)
    if k==2:
       X1=X[:,:((4995)*8-(4995)*3)]
       X2=X[:,((4995)*6):]
    if k==3:
       X1=X[:,:((4995)*8-(4995)*4)]
       X2=X[:,((4995)*5):]
       X=np.concatenate((X1,X2),axis=1)
    if k==4:
       X1=X[:,:((4995)*8-(4995)*5)]
       X2=X[:,-(4995)*4:]
       X=np.concatenate((X1,X2),axis=1)
    if k==5:
       X1=X[:,:((4995)*8-(4995)*6)]
       X2=X[:,((4995)*3):]
       X=np.concatenate((X1,X2),axis=1)
    if k==6:
       X1=X[:,:((4995)*8-(4995*4)*7)]
       X2=X[:,((4995)*2):]
       X=np.concatenate((X1,X2),axis=1)
    if k==7:
       X=X[:,(4995):]
    valid_targets = ~np.isnan(X).all(axis=0)
    X_filtered = X[:, valid_targets]

    Y = Y1
    
    Y = Y.reshape(132, (4995))
    valid_targets = ~np.isnan(Y).all(axis=0)
    Y_filtered = Y[:, valid_targets]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    train_size = int(train_size_ratio * X_filtered.shape[0])

    scaler_X.fit(X_filtered[:train_size])
    scaler_Y.fit(Y_filtered[:train_size])

    X_scaled = scaler_X.transform(X_filtered)
    Y_scaled = scaler_Y.transform(Y_filtered)

    n_targets = Y_filtered.shape[1]
    skills_matrix = np.zeros((max_lead_time, n_targets))
    skills_matrix2 = np.zeros((max_lead_time, n_targets))
    import shap
    import numpy as np

    shap2 = []  # store SHAP values for each lead_time
    all_importances = np.zeros((max_lead_time, n_targets, n_targets*7))

    import shap
    from joblib import Parallel, delayed
    from sklearn.gaussian_process.kernels import RBF

    def extract_rbf_lengthscales(kernel):
        """Recursively extract RBF lengthscales from a composite kernel."""
        if isinstance(kernel, RBF):
            return kernel.length_scale

        if hasattr(kernel, 'k1') and hasattr(kernel, 'k2'):
            # Search both sub-kernels
            result_k1 = extract_rbf_lengthscales(kernel.k1)
            if result_k1 is not None:
                return result_k1
            result_k2 = extract_rbf_lengthscales(kernel.k2)
            if result_k2 is not None:
                return result_k2

        return None  # No RBF kernel found

    for lead_time in range(1, max_lead_time + 1):
        print(f"🔁 Processing lead time {lead_time}...")

        # Prepare data
        X_seq, y_seq = sliding_windows(X_scaled, Y_scaled, seq_length, lead_time)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        # Grid search for best Kernel Ridge model across all outputs together
        krr = KernelRidge()
        search = GridSearchCV(krr, param_grid, cv=3, scoring=correlation_scorer, n_jobs=-1)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred = scaler_Y.inverse_transform(y_pred)
        y_test = scaler_Y.inverse_transform(y_test)
        # Parallel correlation skill
        skills_matrix[lead_time - 1] = Parallel(n_jobs=-1)(
        delayed(compute_corr)(y_test[:, i], y_pred[:, i]) for i in range(n_targets)
        )

        print('✅ X_test shape:', X_test.shape)
        print('✅ X_test shape:', X_test.shape)

        # Efficient SHAP Calculation for all targets at once
        
        # Subsample a small reference dataset (100 samples for efficiency)

        # Never exceeds available samples
        from sklearn.inspection import permutation_importance
        from joblib import Parallel, delayed

    # After the loop, convert to array:

    # Regrid to map
    B = ['NPP', 'NCP']
    skills_matrix2 = skills_matrix_total.reshape(max_lead_time, 37, 135)-skills_matrix.reshape(max_lead_time, 37, 135)

    # Grid setup
    nlat, nlon = 37, 135
    ngrids = nlat * nlon

    full_skills_sensitivity = np.full((max_lead_time, ngrids), np.nan)
    full_skills_sensitivity[:, valid_targets] = skills_matrix2.reshape(max_lead_time, -1)

    # Reshape for map plotting
    full_skills_sensitivity = full_skills_sensitivity.reshape((max_lead_time, nlat, nlon)).transpose(1, 2, 0)
    for i in range(1,20,6):  # for lag = 1, 7
      fig = plt.figure(figsize=(10, 10), dpi=300)
      ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
      ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

      ax.coastlines(resolution='110m')
      gl = ax.gridlines(draw_labels=True)
      gl.top_labels = False
      gl.right_labels = False

      levels = np.linspace(-0.45, 0.45, 40)
      lon = lon1.squeeze()
      lat = lat1.squeeze()
      lon2d, lat2d = np.meshgrid(lon, lat)

      fill = ax.contourf(
        lon2d,
        lat2d,
        full_skills_sensitivity[:, :, i],
        levels=levels,
        cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree()
      )

      cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
      cb.ax.tick_params(labelsize=20)
      cb.set_label('Correlation', fontsize=20)
      cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
      plt.savefig("kernel_regression_sensitivity_"+str(k)+"_"+str(i)+".png")

    
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

    sector_masks_flat = {}
    for name, sd in sector_defs.items():
        m_lon = lon_mask(lon_flat, sd["lonmin"], sd["lonmax"])
        m_lat = (lat_flat <= -30)          # or <= -50 if you prefer
        sector_masks_flat[name] = m_lon & m_lat   # 1D boolean array

    # 2) compute per‐basin skill curves:
    leads = np.arange(1, max_lead_time + 1)

    skill_sensitivity = {}    # will hold arrays of length n_lead
    skill_total = {} 
    
    full_skills_sensitivity_flat = full_skills_sensitivity.reshape(nlat * nlon, max_lead_time).T
    # Shape becomes (max_lead_time, 4995)
    full_skills_flat = full_skills.reshape(nlat * nlon, max_lead_time).T

    for basin, mask in sector_masks_flat.items():
        skill_sensitivity[basin] = np.nanmean(full_skills_sensitivity_flat[:, mask], axis=1)
        skill_total[basin] = np.nanmean(full_skills_flat[:, mask], axis=1)

    # 3) plot each basin in its own figure:
    style = {
    "Total": dict(color="tab:red",   marker="o", linestyle="-",  label="Total"),
    f"No {labels[k]}": dict(color="tab:blue",  marker="s", linestyle="--", label=f"No {labels[k]}"),
    }

    for basin in sector_defs.keys():  # in insertion order
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(leads, skill_total[basin], **style["Total"])
        ax.plot(leads, skill_sensitivity[basin], **style[f"No {labels[k]}"])
        ax.set_title(f"NPP skill · {basin}")
        ax.set_xlabel("Lead time (months)")
        ax.set_ylabel("Mean correlation")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="upper left")
        plt.tight_layout()
        fname = f"kernel_regression_difference_{basin.replace(' ', '_')}_no_{labels[k].replace(' ','_')}.png"
        plt.savefig(fname, dpi=300)
        plt.show()


# 1) build your flat masks once:
