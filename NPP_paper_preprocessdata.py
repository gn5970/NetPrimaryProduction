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
import tqdm
import h5py
import xarray as xr
import numpy as np
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
import xesmf as xe
import numpy as np
import random
from scipy import stats


random.seed(42)
np.random.seed(42)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.stats as sstats
from scipy.stats import gaussian_kde
from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# 1️⃣ Define helper functions
# ============================================================

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm


import numpy as np
from scipy.stats import t as tdist

import numpy as np
from scipy.stats import t as tdist
import numpy as np

from scipy.stats import t as tdist
import numpy as np

from scipy.stats import t as tdist
import numpy as np

import numpy as np
from scipy.stats import t as tdist


def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    # Support both DataArray and Dataset with possible 'lon' or 'XC' coordinate names
    if 'lon' in _ds.coords:
        lon_key = 'lon'
    elif 'XC' in _ds.coords:
        lon_key = 'XC'
    else:
        return _ds  # nothing to do

    if (_ds[lon_key].min() >= 0) and (_ds[lon_key].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords[lon_key] = xr.where(_ds[lon_key] > 180, _ds[lon_key] - 360, _ds[lon_key])
        _ds = _ds.sortby(lon_key)
    return _ds

def pc_trend_rednoise(time, x):
    """
    Linear trend significance accounting for AR(1) residuals
    Returns slope (per year) and p-value
    """

    x = np.asarray(x, dtype=float)
    time = np.asarray(time)

    mask = np.isfinite(x)
    x = x[mask]
    time = time[mask]

    if len(x) < 8:
        return np.nan, np.nan

    # convert time → years safely
    t = (time - time[0]) / np.timedelta64(1, 'D') / 365.25
    t = t.astype(float)

    # linear regression
    slope, intercept = np.polyfit(t, x, 1)
    fit = slope*t + intercept
    res = x - fit

    # ---- unbiased AR(1) estimate ----
    r1 = np.sum(res[1:] * res[:-1]) / np.sum(res**2)
    r1 = np.clip(r1, -0.99, 0.99)

    N = len(x)

    # effective sample size (Bretherton 1999)
    Neff = N * (1 - r1) / (1 + r1)
    Neff = max(Neff, 3)

    # variance inflation due to red noise
    Sxx = np.sum((t - t.mean())**2)
    sigma2 = np.sum(res**2) / (N - 2)

    se = np.sqrt(sigma2 / Sxx) * np.sqrt((1 + r1)/(1 - r1))

    # t statistic
    tstat = slope / se
    p = 2 * (1 - tdist.cdf(abs(tstat), df=Neff-2))

    return slope, p

def red_noise(N, M, g):
    """Generate M AR(1) red-noise realizations of length N."""
    red = np.zeros((N, M))
    red[0, :] = np.random.randn(M)
    for i in range(1, N):
        red[i, :] = g * red[i-1, :] + np.random.randn(M)
    return red

def ar1_fit(y):
    """Fit AR(1) coefficient g using statsmodels ARIMA."""
    y = np.array(y)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return 0.0
    try:
        ar1_mod = ARIMA(y, order=(1, 0, 0), trend='c', missing='drop').fit()
        g = ar1_mod.params[1]  # AR(1) coefficient
        return np.clip(g, -0.9999, 0.9999)
    except Exception:
        return 0.0

def corr_rednoise_levels(x, y, nsim=1000, alpha_levels=[90, 95, 99]):
    """Compute observed r and critical r values for AR(1) red-noise surrogates."""
    x = np.array(x)
    y = np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 10:
        return np.nan, (np.nan, np.nan, np.nan)

    r_obs = np.corrcoef(x, y)[0, 1]

    gx, gy = ar1_fit(x), ar1_fit(y)
    N = len(x)
    rx = red_noise(N, nsim, gx)
    ry = red_noise(N, nsim, gy)
    rs = np.array([np.corrcoef(rx[:, i], ry[:, i])[0, 1] for i in range(nsim)])
    crit = np.percentile(np.abs(rs), alpha_levels)
    return r_obs, crit


def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def remove_time_mean(x):
    return x - x.mean(dim='time',skipna=True)



seconds_per_year=12.011*1000*86400
data = xr.open_dataset("/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")

seconds_per_year =12.011*1000*86400

#data = data.coarsen(YC=16, boundary="pad").mean()
#data = data.coarsen(XC=16, boundary="pad").mean()
#data = data #* seconds_per_year  # Now in molC/m³/year
npp = data.BLGNPP*seconds_per_year  # Assuming this is the variable name



# Step 2: Mask top 100 m
Z_top = data.Z.where(npp.Z >= -100, drop=True)
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
data_NPP = (npp * drF_top * hFacC_top).sum(dim="Z") # [mol C / m² / yr]

fig = plt.figure(figsize=(22, 12), dpi=600)

gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1, 1.8],   # second column wider
    height_ratios=[1, 1],
    wspace=0.25,
    hspace=0.25
)

# -------------------------
# 1) Original NPP map
# -------------------------
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.SouthPolarStereo(central_longitude=0))
ax1.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
ax1.coastlines(resolution="110m")

gl1 = ax1.gridlines(draw_labels=True)
gl1.top_labels = False
gl1.right_labels = False

levels1 = np.linspace(-550, 550, 40)

fill1 = ax1.contourf(
    np.array(data_NPP.XC),
    np.array(data_NPP.YC),
    np.array(data_NPP.mean("time")).squeeze(),
    levels=levels1,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
)
ax1.text(0.02, 0.95, 'a', transform=ax1.transAxes,
         fontsize=22, fontweight='bold', va='top')
ax1.set_title("NPP BSOSE", fontsize=20)
ax1.set_aspect("equal")

cb1 = plt.colorbar(fill1, ax=ax1, orientation="vertical", shrink=0.75)
cb1.set_label("NPP (mgC m$^{-2}$ day$^{-1}$)", fontsize=16)
cb1.ax.tick_params(labelsize=14)
cb1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))



mean_npp = data_NPP.mean(dim=['XC', 'YC'])

# 3. Plot the time series (e.g. at a given depth level, or integrated)
#    Here we’ll pick the surface depth (Z=0) as an example
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ax2 = fig.add_subplot(gs[0, 1])

mean_npp_original = data_NPP.mean(dim=["XC", "YC"])
ax2.text(0.02, 0.95, 'b', transform=ax2.transAxes,
         fontsize=22, fontweight='bold', va='top')
ax2.plot(data_NPP.time, mean_npp_original, color="black", linewidth=2)
ax2.set_title("Spatial mean NPP BSOSE", fontsize=20)
ax2.set_xlabel("Time", fontsize=16)
ax2.set_ylabel("NPP (mgC m$^{-2}$ day$^{-1}$)", fontsize=16)
ax2.tick_params(axis="both", labelsize=14)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax2.grid(True)



data = xr.open_dataset("/projects/CDEUTSCH/DATA/npp_coarsened.nc", engine="h5netcdf")
print('npp',data)
npp_coarsened = data.npp
print("After time alignment:")
#print("npp time:", data_NPP.time.min().values, "to", data_NPP.time.max().values)
#print("npp_coarsened time:", npp_coarsened.time.min().values, "to", npp_coarsened.time.max().values)


lon, lat=np.meshgrid(npp_coarsened.lon,npp_coarsened.lat)

# -------------------------
# 3) Coarsened NPP map
# -------------------------
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.SouthPolarStereo(central_longitude=0))
ax3.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
ax3.coastlines(resolution="110m")
ax3.text(0.02, 0.95, 'c', transform=ax3.transAxes,
         fontsize=22, fontweight='bold', va='top')

gl3 = ax3.gridlines(draw_labels=True)
gl3.top_labels = False
gl3.right_labels = False

levels2 = np.linspace(-550, 550, 40)

fill2 = ax3.contourf(
    np.array(npp_coarsened.lon),
    np.array(npp_coarsened.lat),
    np.array(npp_coarsened.mean("time")).squeeze(),
    levels=levels2,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
)

ax3.set_title("NPP Eppley-VGPM", fontsize=20)
ax3.set_aspect("equal")

cb2 = plt.colorbar(fill2, ax=ax3, orientation="vertical", shrink=0.75)
cb2.set_label("NPP (mgC m$^{-2}$ day$^{-1}$)", fontsize=16)
cb2.ax.tick_params(labelsize=14)
cb2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))


ax4 = fig.add_subplot(gs[1, 1])

mean_npp_coarsened = npp_coarsened.mean(dim=["lon", "lat"])
ax4.text(0.02, 0.95, 'd', transform=ax4.transAxes,
         fontsize=22, fontweight='bold', va='top')
ax4.plot(npp_coarsened.time, mean_npp_coarsened, color="black", linewidth=2)
ax4.set_title("Spatial mean NPP Eppley-VGPM", fontsize=20)
ax4.set_xlabel("Time", fontsize=16)
ax4.set_ylabel("NPP (mgC m$^{-2}$ day$^{-1}$)", fontsize=16)
ax4.tick_params(axis="both", labelsize=14)
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax4.grid(True)


plt.tight_layout()
plt.savefig("Figure1.pdf", dpi=600,
    bbox_inches="tight",
    pad_inches=0.05
)








import xesmf as xe
import numpy as np
import xarray as xr

data = xr.open_dataset("/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
data=convert_0_360_to_neg180_180(data)
data['time'] = pd.date_range("2013-01-01", periods=132, freq="M")

seconds_per_year =12.011*1000*86400

data = data.coarsen(YC=16, boundary="pad").mean()
data = data.coarsen(XC=16, boundary="pad").mean()
#data = data #* seconds_per_year  # Now in molC/m³/year
npp = data.BLGNPP*seconds_per_year  # Assuming this is the variable name



# Step 2: Mask top 100 m
Z_top = data.Z.where(npp.Z >= -100, drop=True)
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
data_NPP = (npp * drF_top * hFacC_top).sum(dim="Z") # [mol C / m² / yr]


# --- Ensure data_NPP has lon/lat coordinate names xESMF expects ---
# Many MOM6/BSOSE outputs use 'XC' and 'YC' for horizontal coords. If so,
# rename them to 'lon' and 'lat' so the regridder grids and data align cleanly.
# --- Ensure data_NPP has lon/lat coordinate names xESMF expects ---
# Many MOM6/BSOSE outputs use 'XC' and 'YC' for horizontal coords. If so,
# rename them to 'lon' and 'lat' so the regridder grids and data align cleanly.
if ('lon' not in data_NPP.coords) and ('XC' in data_NPP.coords):
    data_NPP = data_NPP.rename({'XC': 'lon', 'YC': 'lat'})


data = xr.open_dataset("/projects/CDEUTSCH/DATA/npp_coarsened.nc", engine="h5netcdf")
npp_coarsened = data.npp


print('npp_coarsed',npp_coarsened)

# Similarly ensure the satellite product has 'lon'/'lat' coords
if ('lon' not in npp_coarsened.coords) and ('XC' in npp_coarsened.coords):
    npp_coarsened = npp_coarsened.rename({'XC': 'lon', 'YC': 'lat'})

# Print coordinate info for debugging
print("Model grid coordinates:")
print("Longitude range:", data_NPP.lon.min().values, "to", data_NPP.lon.max().values)
print("Latitude range:", data_NPP.lat.min().values, "to", data_NPP.lat.max().values)
print("\nSatellite grid coordinates:")
print("Longitude range:", npp_coarsened.lon.min().values, "to", npp_coarsened.lon.max().values)
print("Latitude range:", npp_coarsened.lat.min().values, "to", npp_coarsened.lat.max().values)

# === 1️⃣ Define your grids explicitly as 1D coordinate arrays ===
# xESMF expects grid dicts containing 1D lon/lat arrays (in degrees).
grid_model = {
    'lon': data_NPP['lon'].values,
    'lat': data_NPP['lat'].values
}
grid_satellite = {
    'lon': npp_coarsened['lon'].values,
    'lat': npp_coarsened['lat'].values
}

# === 2️⃣ Create both regridders ONCE ===
# Use reuse_weights=True when rerunning interactively to avoid recomputing weights unnecessarily.
regridder_model_to_satellite = xe.Regridder(
    grid_model,
    grid_satellite,
    method='bilinear',
    periodic=True,
)

regridder_satellite_to_model = xe.Regridder(
    grid_satellite,
    grid_model,
    method='bilinear',
    periodic=True,
)

data_NPP = data_NPP.transpose('time', 'lat', 'lon')
npp_on_satellite = regridder_model_to_satellite(data_NPP)

print('npp_coarsened',npp_coarsened)
print('npp_on_satellite',npp_on_satellite)
correlation_map2 = xr.corr(npp_coarsened, npp_on_satellite, dim='time')

# ============================================================
# AR(1) SIGNIFICANCE FOR CORRELATION MAP
# ============================================================

import numpy as np
import xarray as xr
from scipy.stats import t as student_t

def lag1_autocorr(ts):
    ts = ts[np.isfinite(ts)]
    if len(ts) < 3:
        return np.nan
    return np.corrcoef(ts[:-1], ts[1:])[0,1]

def corr_sig_ar1(ts1, ts2, alpha=0.01):

    mask = np.isfinite(ts1) & np.isfinite(ts2)
    ts1 = ts1[mask]
    ts2 = ts2[mask]

    if len(ts1) < 10:
        return np.nan, False

    r = np.corrcoef(ts1, ts2)[0,1]

    r1 = lag1_autocorr(ts1)
    r2 = lag1_autocorr(ts2)

    if np.isnan(r1) or np.isnan(r2):
        return r, False

    n = len(ts1)

    # Bretherton effective DOF
    neff = n * (1 - r1*r2) / (1 + r1*r2)
    neff = max(neff, 3)

    tval = r * np.sqrt((neff - 2) / (1 - r**2))
    pval = 2 * (1 - student_t.cdf(abs(tval), df=neff-2))

    return r, (pval < alpha)

# ============================================================
# COMPUTE SIGNIFICANCE GRIDPOINT BY GRIDPOINT
# ============================================================

lat = correlation_map2.lat.values
lon = correlation_map2.lon.values

sig_99 = np.zeros((len(lat), len(lon)), dtype=bool)

sat_np   = npp_coarsened.values
model_np = npp_on_satellite.values

for i in range(len(lat)):
    for j in range(len(lon)):

        ts1 = sat_np[:, i, j]
        ts2 = model_np[:, i, j]

        _, sig = corr_sig_ar1(ts1, ts2, alpha=0.01)
        sig_99[i, j] = sig

sig_99 = xr.DataArray(sig_99, coords=correlation_map2.coords, dims=correlation_map2.dims)

corr_sig = correlation_map2.where(sig_99)

# === 3️⃣ Regrid model -> satellite grid ===
print('\nPre-regridding info:')
print('data_NPP dims, coords:', data_NPP.dims, list(data_NPP.coords))
print('data_NPP shape:', data_NPP.shape)

# Ensure data is in correct order (time, lat, lon)
data_NPP = data_NPP.transpose('time', 'lat', 'lon')


fig = plt.figure(figsize=(16, 7), dpi=600)
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=0))

# =====================================================
# (a) Correlation
# =====================================================
ax1.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
ax1.coastlines(resolution="110m")

gl1 = ax1.gridlines(draw_labels=True)
gl1.top_labels = False
gl1.right_labels = False

levels = np.linspace(-1, 1, 41)

ax1.contourf(
    correlation_map2.lon,
    correlation_map2.lat,
    correlation_map2,
    levels=levels,
    cmap="Greys",
    alpha=0.3,
    transform=ccrs.PlateCarree()
)

fill1 = ax1.contourf(
    correlation_map2.lon,
    correlation_map2.lat,
    corr_sig,
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
)

ax1.contour(
    correlation_map2.lon,
    correlation_map2.lat,
    sig_99,
    levels=[0.5],
    colors="black",
    linewidths=0.8,
    transform=ccrs.PlateCarree()
)

ax1.set_title("Correlation", fontsize=18)
ax1.text(
    0.02, 0.96, "a",
    transform=ax1.transAxes,
    fontsize=24,
    fontweight="bold",
    va="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)

cb1 = plt.colorbar(fill1, ax=ax1, orientation="vertical", shrink=0.75)
cb1.set_label("Correlation (r)", fontsize=16)
cb1.ax.tick_params(labelsize=14)
cb1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


# =====================================================
# (b) Error
# =====================================================
error = npp_on_satellite - npp_coarsened

ax2.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
ax2.coastlines(resolution="110m")

gl2 = ax2.gridlines(draw_labels=True)
gl2.top_labels = False
gl2.right_labels = False

levels_error = np.linspace(-600, 600, 30)
ticks_error = np.arange(-600, 601, 200)

fill2 = ax2.contourf(
    error.lon,
    error.lat,
    error.mean("time").squeeze(),
    levels=levels_error,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
)

ax2.set_title("Bias", fontsize=18)
ax2.text(
    0.02, 0.96, "b",
    transform=ax2.transAxes,
    fontsize=24,
    fontweight="bold",
    va="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)

cb2 = plt.colorbar(
    fill2,
    ax=ax2,
    orientation="vertical",
    shrink=0.75,
    ticks=ticks_error
)
cb2.set_label("NPP difference (mgC m$^{-2}$ day$^{-1}$)", fontsize=16)
cb2.ax.tick_params(labelsize=14)
cb2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

plt.tight_layout()

plt.savefig(
    "Figure2.pdf",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05
)



data_thetao = xr.open_dataset("/projects/CDEUTSCH/DATA/Theta_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_thetao = data_thetao.coarsen(YC=8, boundary="pad").mean()
#data_thetao = data_thetao.coarsen(XC=8, boundary="pad").mean()
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
time_mean_THETA_anom=time_mean_THETA_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_THETA_anom2=time_mean_THETA_anom/time_mean_THETA_anom.std()
#time_mean_THETA_anom2=time_mean_THETA_anom2.coarsen(XC=16, boundary="pad").mean()



time_mean_THETA_anom2_1=time_mean_THETA_anom2.isel(time=slice(0,36)).mean('time')
time_mean_THETA_anom2_2=time_mean_THETA_anom2.isel(time=slice(132-36,132)).mean('time')

#print('data_Fe',data_Fe)
#print('data_Fe',np.array(data_Fe2-data_Fe1).shape)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.5,0.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(time_mean_THETA_anom2.XC), np.array(time_mean_THETA_anom2.YC), np.array(time_mean_THETA_anom2_2-time_mean_THETA_anom2_1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Supplemental_Thetadifference.pdf")




data_ssh = xr.open_dataset("/projects/CDEUTSCH/DATA/SSH_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_ssh = data_ssh.coarsen(YC=8, boundary="pad").mean()
#data_ssh = data_ssh.coarsen(XC=8, boundary="pad").mean()
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
time_mean_SSH_anom=time_mean_SSH_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_SSH_anom2=time_mean_SSH_anom/time_mean_SSH_anom.std()
#time_mean_SSH_anom2=time_mean_SSH_anom2.coarsen(XC=16, boundary="pad").mean()


time_mean_SSH_anom2_1=time_mean_SSH_anom2.isel(time=slice(0,36)).mean('time')
time_mean_SSH_anom2_2=time_mean_SSH_anom2.isel(time=slice(132-36,132)).mean('time')

#print('data_Fe',data_Fe)
#print('data_Fe',np.array(data_Fe2-data_Fe1).shape)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.5,0.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(time_mean_SSH_anom2.XC), np.array(time_mean_SSH_anom2.YC), np.array(time_mean_SSH_anom2_2-time_mean_SSH_anom2_1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Supplemental_SSHdifference.pdf")



data_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
#data_salt = data_salt.coarsen(YC=8, boundary="pad").mean()
#data_salt = data_salt.coarsen(XC=8, boundary="pad").mean()
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
data_salt = data_salt.where(data_salt.Z >= -40, drop=True).mean('Z')

time_mean_SALT_clim = data_salt.SALT.groupby('time.month').mean(dim='time',skipna=True)
time_mean_SALT_anom = data_salt.SALT.groupby('time.month')-time_mean_SALT_clim
time_mean_SALT_anom=time_mean_SALT_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_SALT_anom=time_mean_SALT_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_SALT_anom2=time_mean_SALT_anom/time_mean_SALT_anom.std()
#time_mean_SALT_anom2=time_mean_SALT_anom2.coarsen(XC=16, boundary="pad").mean()

data_SIArea = xr.open_dataset("/projects/CDEUTSCH/DATA/SIArea_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_SIArea = data_SIArea.coarsen(YC=8, boundary="pad").mean()
#data_SIArea = data_SIArea.coarsen(XC=8, boundary="pad").mean()
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
time_mean_SIArea_anom=time_mean_SIArea_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_SIArea_anom2=time_mean_SIArea_anom/time_mean_SIArea_anom.std()
#time_mean_SIArea_anom2=time_mean_SIArea_anom2.coarsen(XC=16, boundary="pad").mean()



time_mean_SIArea_anom2_1=time_mean_SIArea_anom2.isel(time=slice(0,36)).mean('time')
time_mean_SIArea_anom2_2=time_mean_SIArea_anom2.isel(time=slice(132-36,132)).mean('time')

#print('data_Fe',data_Fe)
#print('data_Fe',np.array(data_Fe2-data_Fe1).shape)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.5,0.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(time_mean_SIArea_anom2.XC), np.array(time_mean_SIArea_anom2.YC), np.array(time_mean_SIArea_anom2_2-time_mean_SIArea_anom2_1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Supplemental_Siareadifference.pdf")



data_MLD = xr.open_dataset("/projects/CDEUTSCH/DATA/MLD_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_MLD= data_MLD.coarsen(YC=8, boundary="pad").mean()
#data_MLD= data_MLD.coarsen(XC=8, boundary="pad").mean()

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
time_mean_MLD_anom=time_mean_MLD_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_MLD_anom2=time_mean_MLD_anom/time_mean_MLD_anom.std()



time_mean_MLD_anom2_1=time_mean_MLD_anom2.isel(time=slice(0,36)).mean('time')
time_mean_MLD_anom2_2=time_mean_MLD_anom2.isel(time=slice(132-36,132)).mean('time')

#print('data_Fe',data_Fe)
#print('data_Fe',np.array(data_Fe2-data_Fe1).shape)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.5,0.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(time_mean_MLD_anom2.XC), np.array(time_mean_MLD_anom2.YC), np.array(time_mean_MLD_anom2_2-time_mean_MLD_anom2_1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Supplemental_MLDdifference.pdf")




data_NO3 = xr.open_dataset("/projects/CDEUTSCH/DATA/NO3_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
#data_NO3 =data_NO3.coarsen(YC=8, boundary="pad").mean()
#data_NO3 =data_NO3.coarsen(XC=8, boundary="pad").mean()
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
time_mean_TRAC04_anom=time_mean_TRAC04_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_TRAC04_anom2=time_mean_TRAC04_anom/time_mean_TRAC04_anom.std()
#time_mean_TRAC04_anom2=time_mean_TRAC04_anom2.coarsen(XC=16, boundary="pad").mean()


data_irris = xr.open_dataset("/projects/CDEUTSCH/DATA/irris_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_irris =data_irris.coarsen(YC=8, boundary="pad").mean()
#data_irris =data_irris.coarsen(XC=8, boundary="pad").mean()
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
time_mean_irris_anom=time_mean_irris_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_irris_anom2=time_mean_irris_anom/time_mean_irris_anom.std()
#time_mean_irris_anom2=time_mean_irris_anom2.coarsen(XC=16, boundary="pad").mean()


time_mean_irris_anom2_1=time_mean_irris_anom2.isel(time=slice(0,36)).mean('time')
time_mean_irris_anom2_2=time_mean_irris_anom2.isel(time=slice(132-36,132)).mean('time')

#print('data_Fe',data_Fe)
#print('data_Fe',np.array(data_Fe2-data_Fe1).shape)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.5,0.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(time_mean_irris_anom2.XC), np.array(time_mean_irris_anom2.YC), np.array(time_mean_irris_anom2_2-time_mean_irris_anom2_1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Supplemental_irrisdifference.pdf")



data_Fe = xr.open_dataset("/projects/CDEUTSCH/DATA/Fe_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_Fe=data_Fe.coarsen(YC=8, boundary="pad").mean()
#data_Fe=data_Fe.coarsen(XC=8, boundary="pad").mean()
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
time_mean_TRAC06_anom=time_mean_TRAC06_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_TRAC06_anom2=time_mean_TRAC06_anom/time_mean_TRAC06_anom.std()
#time_mean_TRAC06_anom2=time_mean_TRAC06_anom2.coarsen(XC=16, boundary="pad").mean()



time_mean_TRAC06_anom2_1=time_mean_TRAC06_anom2.isel(time=slice(0,36)).mean('time')
time_mean_TRAC06_anom2_2=time_mean_TRAC06_anom2.isel(time=slice(132-36,132)).mean('time')

#print('data_Fe',data_Fe)
#print('data_Fe',np.array(data_Fe2-data_Fe1).shape)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.5,0.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(time_mean_TRAC06_anom2.XC), np.array(time_mean_TRAC06_anom2.YC), np.array(time_mean_TRAC06_anom2_2-time_mean_TRAC06_anom2_1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Supplemental_Fedifference.pdf")






NO3_Fe_ratio = data_NO3 / data_Fe
time_mean_NO3_Fe_ratio_clim = NO3_Fe_ratio.groupby('time.month').mean(dim='time',skipna=True)
time_mean_NO3_Fe_ratio_anom = NO3_Fe_ratio.groupby('time.month')-time_mean_NO3_Fe_ratio_clim
time_mean_NO3_Fe_ratio_anom=detrend_dim(time_mean_NO3_Fe_ratio_anom,dim='time')
time_mean_NO3_Fe_ratio_anom=time_mean_NO3_Fe_ratio_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_NO3_Fe_ratio_anom=time_mean_NO3_Fe_ratio_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_NO3_Fe_ratio_anom2=time_mean_NO3_Fe_ratio_anom/time_mean_NO3_Fe_ratio_anom.std()






print('data_NPP',data_NPP)
data = xr.open_dataset("/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")

seconds_per_year =12.011*1000*86400

#data = data.coarsen(YC=16, boundary="pad").mean()
#data = data.coarsen(XC=16, boundary="pad").mean()
#data = data #* seconds_per_year  # Now in molC/m³/year
npp = data.BLGNPP*seconds_per_year  # Assuming this is the variable name



# Step 2: Mask top 100 m
Z_top = data.Z.where(npp.Z >= -100, drop=True)
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
data_NPP = (npp * drF_top * hFacC_top).sum(dim="Z") # [mol C / m² / yr]

time_mean_NPP_clim = data_NPP.groupby('time.month').mean(dim='time',skipna=True)
time_mean_NPP_anom = data_NPP.groupby('time.month')-time_mean_NPP_clim
time_mean_NPP_anom=detrend_dim(time_mean_NPP_anom,dim='time')
time_mean_NPP_anom=time_mean_NPP_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_NPP_anom=time_mean_NPP_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_NPP_anom2=time_mean_NPP_anom/time_mean_NPP_anom.std()
#time_mean_NPP_anom2=time_mean_NPP_anom2.coarsen(XC=16, boundary="pad").mean()

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

# ---------------------------------------------------------------
# Compute trends/p-values for both PCs (your existing function)
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

# ---------------------------------------------------------------
# Compute trends/p-values for both PCs
# ---------------------------------------------------------------
pc1 = pcs_normalized.isel(mode=0)
pc2 = pcs_normalized.isel(mode=1)

slope1, pval1 = pc_trend_rednoise(pc1.time.values, pc1.values)
slope2, pval2 = pc_trend_rednoise(pc2.time.values, pc2.values)

print("PC1 trend (per year):", slope1, "p =", pval1)
print("PC2 trend (per year):", slope2, "p =", pval2)

# ---------------------------------------------------------------
# Build the 2x2 figure (mixed projections)
# ---------------------------------------------------------------
proj = ccrs.SouthPolarStereo(central_longitude=0)

fig = plt.figure(figsize=(16, 12), dpi=300)
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.0, 1.4],
    hspace=0.30,
    wspace=0.20,
    left=0.06, right=0.92, top=0.94, bottom=0.08,
)

panel_letters = [['a', 'b'],
                 ['c', 'd']]

eof_levels = np.linspace(-9e-1, 9e-1, 40)
eof_data   = [eofs_normalized.isel(mode=0), eofs_normalized.isel(mode=1)]
eof_titles = ['First EOF NPP', 'Second EOF NPP']

pc_data    = [pc1, pc2]
pc_titles  = ['First PC NPP', 'Second PC NPP']
pc_slopes  = [slope1, slope2]
pc_pvals   = [pval1, pval2]

# ---- Column 1: EOF maps ----
for row in range(2):
    ax = fig.add_subplot(gs[row, 0], projection=proj)
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False

    fill = ax.contourf(
        np.array(eof_data[row].XC),
        np.array(eof_data[row].YC),
        np.array(eof_data[row]).squeeze(),
        levels=eof_levels,
        cmap=plt.cm.RdBu_r,
        extend='both',
        transform=ccrs.PlateCarree(),
    )
    ax.set_aspect('equal', adjustable='box')

    # title above the panel
    ax.set_title(eof_titles[row], fontsize=16)

    # colorbar without a label (info now in the title)
    cb = fig.colorbar(fill, ax=ax, orientation='vertical',
                      shrink=0.75, pad=0.05)
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))

    # panel letter
    ax.text(0.02, 0.97, panel_letters[row][0],
            transform=ax.transAxes,
            fontsize=22, fontweight='bold',
            va='top', ha='left', zorder=10)

# ---- Column 2: PC time series ----
for row in range(2):
    ax = fig.add_subplot(gs[row, 1])

    ax.plot(pc_data[row].time, pc_data[row],
            color='black', linewidth=2)

    # title above the panel
    ax.set_title(pc_titles[row], fontsize=16)

    ax.set_ylabel(f'Principal Component {row + 1}', fontsize=15)
    ax.set_xlabel('Time', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(True, alpha=0.3)

    # trend / significance label
    #label = f"Trend = {pc_slopes[row]:.3f} yr$^{{-1}}$\np = {pc_pvals[row]:.3f}"
    #if pc_pvals[row] < 0.05:
    #    label += "  (significant)"
    ax.text(0.02, 0.95, panel_letters[row][1],
            transform=ax.transAxes,
            fontsize=22, fontweight='bold',
            va='top', ha='left', zorder=10)
    #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # panel letter

plt.savefig('Supplemental_Figure3_eof_pc_combined.pdf',
            dpi=300, bbox_inches='tight')
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
plt.savefig('Supplemental_Figure4_variance_NPP.pdf',dpi=300, bbox_inches='tight')




from eofs.xarray import Eof


correlation_map = xr.corr(time_mean_NO3_Fe_ratio_anom2, time_mean_NPP_anom2, dim='time')
nlat, nlon = time_mean_NPP_anom2.shape[1], time_mean_NPP_anom2.shape[2]
corr_map = np.full((nlat, nlon), np.nan)
sig_level_map = np.full((nlat, nlon), np.nan)

for j in tqdm(range(nlat), desc="Latitude"):
    for i in range(nlon):
        x = np.array(time_mean_NPP_anom2[:, j, i])
        y = np.array(time_mean_THETA_anom2[:, j, i])
        if np.isfinite(x).sum() > 20 and np.isfinite(y).sum() > 20:
            r, (r90, r95, r99) = corr_rednoise_levels(x, y, nsim=500)
            corr_map[j, i] = r
            if abs(r) >= r99:
                sig_level_map[j, i] = 99
            elif abs(r) >= r95:
                sig_level_map[j, i] = 95
            #elif abs(r) >= r90:
            #    sig_level_map[j, i] = 90



fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)
gl.top_labels = False
gl.right_labels = False

levels = np.linspace(-0.8, 0.8, 40)
fill = ax.contourf(
    np.array(correlation_map.XC),
    np.array(correlation_map.YC),
    corr_map,
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
)

# --- significance hatching ---
ax.contourf(
    np.array(correlation_map.XC),
    np.array(correlation_map.YC),
    sig_level_map,
    levels=[89, 94, 99, 101],
    colors='none',
    hatches=['..', '//', '\\\\'],
    transform=ccrs.PlateCarree()
)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=16)
cb.set_label('Correlation (NPP vs THETA)', fontsize=18)
ax.set_aspect('equal', adjustable='box')
#ax.set_title('Correlation with Monte Carlo AR(1) Red-Noise Significance', fontsize=16)
plt.tight_layout()
font_size = 20  # Adjust as appropriate.
cb.ax.tick_params(labelsize=font_size)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("Supplemental_correlation_NPP_ratio_rednoise.png", dpi=300)
plt.show()



import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature

from scipy import stats




import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from tqdm import tqdm

# ---------------------------------------------------------------
# 1. Define the six (predictor, label) pairs in panel order
# ---------------------------------------------------------------
predictors = [
    (time_mean_MLD_anom2,    'MLD'),
    (time_mean_TRAC04_anom2, 'Fe'),
    (time_mean_THETA_anom2,  r'$\theta$'),
    (time_mean_TRAC06_anom2, r'NO$_3^-$'),
    (time_mean_SIArea_anom2, 'Sea Ice Area'),
    (time_mean_irris_anom2,  'Irradiance'),
]

panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']

# ---------------------------------------------------------------
# 2. Compute correlation + significance for each predictor
#    (skip this block if you've already cached the maps)
# ---------------------------------------------------------------
nlat, nlon = time_mean_NPP_anom2.shape[1], time_mean_NPP_anom2.shape[2]

corr_maps = []
sig_maps  = []

for predictor, label in predictors:
    corr_map = np.full((nlat, nlon), np.nan)
    sig_map  = np.full((nlat, nlon), np.nan)

    for j in tqdm(range(nlat), desc=f'{label}'):
        for i in range(nlon):
            x = np.array(time_mean_NPP_anom2[:, j, i])
            y = np.array(predictor[:, j, i])
            if np.isfinite(x).sum() > 20 and np.isfinite(y).sum() > 20:
                r, (r90, r95, r99) = corr_rednoise_levels(x, y, nsim=500)
                corr_map[j, i] = r
                if   abs(r) >= r99: sig_map[j, i] = 99
                elif abs(r) >= r95: sig_map[j, i] = 95

    corr_maps.append(corr_map)
    sig_maps.append(sig_map)

# ---------------------------------------------------------------
# 3. Build the 2x3 figure
# ---------------------------------------------------------------
proj = ccrs.SouthPolarStereo(central_longitude=0)

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10),
    dpi=300,
    subplot_kw={'projection': proj},
)

levels = np.linspace(-0.8, 0.8, 40)
lon = np.array(time_mean_NPP_anom2.XC)
lat = np.array(time_mean_NPP_anom2.YC)

fill = None  # will hold the last QuadContourSet for the shared colorbar

for ax, corr_map, sig_map, (_, label), tag in zip(
    axes.flat, corr_maps, sig_maps, predictors, panel_labels
):
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.6)

    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.6)

    fill = ax.contourf(
        lon, lat, corr_map,
        levels=levels,
        cmap=plt.cm.RdBu_r,
        extend='both',
        transform=ccrs.PlateCarree(),
    )

    # significance hatching: 95% with '..', 99% with '//'
    ax.contourf(
        lon, lat, sig_map,
        levels=[94, 98, 101],
        colors='none',
        hatches=['..','//'],
        transform=ccrs.PlateCarree(),
    )

    # bold panel letter in top-left corner, inside the axes
    ax.text(
        0.02, 0.97, tag,
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        va='top',
        ha='left',
        zorder=10,
    )

    # variable name as a regular title above the panel
    ax.set_title(label, fontsize=16)
    ax.set_aspect('equal', adjustable='box')
# ---------------------------------------------------------------
# 4. Shared colorbar
# ---------------------------------------------------------------
fig.subplots_adjust(left=0.04, right=0.90, top=0.95, bottom=0.05,
                    wspace=0.08, hspace=0.12)

cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
cb = fig.colorbar(fill, cax=cbar_ax, orientation='vertical')
cb.set_label('Correlation', fontsize=18)
cb.ax.tick_params(labelsize=16)
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

plt.savefig('Figure3.pdf', dpi=300, bbox_inches='tight')
plt.show()



#coarsen_factor=8
data_TAUX = xr.open_dataset("/projects/CDEUTSCH/DATA/oceTAUX_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_TAUX = data_TAUX.coarsen(XG=coarsen_factor,YC=coarsen_factor, boundary='trim').mean()

data_TAUY = xr.open_dataset("/projects/CDEUTSCH/DATA/oceTAUY_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_TAUY = data_TAUY.coarsen(XC=coarsen_factor,YG=coarsen_factor, boundary='trim').mean()



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


zcurl = compute_zcurl_with_time(
    TAUX_on_tracer,
    TAUY_on_tracer,
    dx, dy,
    TAUY_on_tracer.YC.values
)

time_mean_wsc_clim = zcurl.groupby('time.month').mean(dim='time',skipna=True)
time_mean_wsc_anom = zcurl.groupby('time.month')-time_mean_wsc_clim
time_mean_wsc_anom=detrend_dim(time_mean_wsc_anom,dim='time')
time_mean_wsc_anom=time_mean_wsc_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_wsc_anom=time_mean_wsc_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_wsc_anom2=time_mean_wsc_anom/time_mean_wsc_anom.std()

correlation_map = xr.corr(time_mean_wsc_anom2, time_mean_NPP_anom2, dim='time')

nlat, nlon =time_mean_wsc_anom2.shape[1], time_mean_wsc_anom2.shape[2]
corr_map = np.full((nlat, nlon), np.nan)
sig_level_map = np.full((nlat, nlon), np.nan)

for j in tqdm(range(nlat), desc="Latitude"):
    for i in range(nlon):
        x = np.array(time_mean_NPP_anom2[:, j, i])
        y = np.array(time_mean_wsc_anom2[:, j, i])
        if np.isfinite(x).sum() > 20 and np.isfinite(y).sum() > 20:
            r, (r90, r95, r99) = corr_rednoise_levels(x, y, nsim=500)
            corr_map[j, i] = r
            if abs(r) >= r99:
                sig_level_map[j, i] = 99
            elif abs(r) >= r95:
                sig_level_map[j, i] = 95
            #elif abs(r) >= r90:
            #    sig_level_map[j, i] = 90


fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)
gl.top_labels = False
gl.right_labels = False

levels = np.linspace(-0.8, 0.8, 40)
fill = ax.contourf(
    np.array(correlation_map.XC),
    np.array(correlation_map.YC),
    corr_map,
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
)

# --- significance hatching ---
ax.contourf(
    np.array(correlation_map.XC),
    np.array(correlation_map.YC),
    sig_level_map,
    levels=[89, 94, 99, 101],
    colors='none',
    hatches=['..', '//', '\\\\'],
    transform=ccrs.PlateCarree()
)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
font_size = 22  # Adjust as appropriate.
cb.ax.tick_params(labelsize=font_size)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
#cb.set_label('Correlation (NPP vs wsc)', fontsize=18)
ax.set_aspect('equal', adjustable='box')
#ax.set_title('Correlation with Monte Carlo AR(1) Red-Noise Significance', fontsize=16)
plt.tight_layout()
plt.savefig("correlation_NPP_wsc_rednoise.png", dpi=300)
plt.show()



data_NO3 = xr.open_dataset("/projects/CDEUTSCH/DATA/NO3_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
#data_NO3 =data_NO3.coarsen(YC=8, boundary="pad").mean()
#data_NO3 =data_NO3.coarsen(XC=8, boundary="pad").mean()
# Apply the conversion to your dataset
#data_Fe = convert_0_360_to_neg180_180(data_Fe)
#data_Fe= data_Fe.where(data_irris.maskC == 1)
data_NO3 = data_NO3.TRAC04
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





data_Fe = xr.open_dataset("/projects/CDEUTSCH/DATA/Fe_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_Fe=data_Fe.coarsen(YC=8, boundary="pad").mean()
#data_Fe=data_Fe.coarsen(XC=8, boundary="pad").mean()
data_Fe = data_Fe.TRAC06
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


Ratio=data_Fe/data_NO3
print('Ratio',Ratio)

time_mean_ratio_clim = Ratio.groupby('time.month').mean(dim='time',skipna=True)
time_mean_ratio_anom = Ratio.groupby('time.month')-time_mean_ratio_clim
time_mean_ratio_anom=detrend_dim(time_mean_ratio_anom,dim='time')
time_mean_ratio_anom=time_mean_ratio_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_ratio_anom=time_mean_ratio_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_ratio_anom2=time_mean_ratio_anom/time_mean_ratio_anom.std()

correlation_map = xr.corr(time_mean_ratio_anom2, time_mean_NPP_anom2, dim='time')

nlat, nlon = time_mean_ratio_anom2.shape[1], time_mean_ratio_anom2.shape[2]
corr_map = np.full((nlat, nlon), np.nan)
sig_level_map = np.full((nlat, nlon), np.nan)

for j in tqdm(range(nlat), desc="Latitude"):
    for i in range(nlon):
        x = np.array(time_mean_NPP_anom2[:, j, i])
        y = np.array(time_mean_ratio_anom2[:, j, i])
        if np.isfinite(x).sum() > 20 and np.isfinite(y).sum() > 20:
            r, (r90, r95, r99) = corr_rednoise_levels(x, y, nsim=500)
            corr_map[j, i] = r
            if abs(r) >= r99:
                sig_level_map[j, i] = 99
            elif abs(r) >= r95:
                sig_level_map[j, i] = 95
            #elif abs(r) >= r90:
            #    sig_level_map[j, i] = 90


fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)

levels = np.linspace(-0.8, 0.8, 40)
fill = ax.contourf(
    np.array(correlation_map.XC),
    np.array(correlation_map.YC),
    corr_map,
    levels=levels,
    cmap=plt.cm.RdBu_r,
    transform=ccrs.PlateCarree()
)

# --- significance hatching ---
ax.contourf(
    np.array(correlation_map.XC),
    np.array(correlation_map.YC),
    sig_level_map,
    levels=[89, 94, 99, 101],
    colors='none',
    hatches=['..', '//', '\\\\'],
    transform=ccrs.PlateCarree()
)
cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
font_size = 22  # Adjust as appropriate.
cb.ax.tick_params(labelsize=font_size)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
#cb.set_label('Correlation (NPP vs ratio)', fontsize=18)
ax.set_aspect('equal', adjustable='box')
#ax.set_title('Correlation with Monte Carlo AR(1) Red-Noise Significance', fontsize=16)
plt.tight_layout()
plt.savefig("correlation_NPP_ratio_rednoise.png", dpi=300)
plt.show()







### comparison with WODB

infile = '/projects/CDEUTSCH/DATA/WOD23/wod_1955-2023.nc'
data=xr.open_dataset(infile)
#print('data_ATLAS',data)

#data['time']=pd.date_range("1955-01-01", periods=828, freq="M")
new_time = pd.date_range("1955-01-01", periods=data.sizes['time'], freq="MS")
data = data.assign_coords(time=new_time)
data = data.sel(time=slice("2013-01-01", "2023-12-31"))
print('data_ATLAS',data)

#data=data.isel(time=slice(100,828))
#data=data.mean('nbounds')
min_lon = -180
min_lat = -90
min_depth = 0

max_lon = 180
max_lat = -30
max_depth = 100

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat & mask_depth, drop=True)
data=data.mean('depth')

lon=data.lon.values
lat=data.lat.values
print('NO3 WODB',data.no3)
print('NO3 BSOSE',data_NO3)

for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 32, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, data.no3.mean('time').squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    ax.contour(lon,lat, data.no3.mean('time').squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_WODB.png")



data_NO3 = xr.open_dataset("/projects/CDEUTSCH/DATA/NO3_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
#data_NO3 =data_NO3.coarsen(YC=8, boundary="pad").mean()
#data_NO3 =data_NO3.coarsen(XC=8, boundary="pad").mean()
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


#drF = data_NO3.drF  # (Z), vertical cell thickness in m
#hFacC = data_NO3.hFacC  # (Z, YC, XC), vertical fraction of wet cell
#rA = data_NO3.rA

# Step 3: Align shapes for multiplication
#drF_top = drF.sel(Z=Z_top)
#hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
#data_NO3 = (data_NO3 * drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]
data_NO3=data_NO3.mean('Z')


grid_model = {
    'lon': data_NO3['XC'].values,
    'lat': data_NO3['YC'].values
}
grid_WODB = {
    'lon': data['lon'].values,
    'lat': data['lat'].values
}

# === 2️⃣ Create both regridders ONCE ===
# Use reuse_weights=True when rerunning interactively to avoid recomputing weights unnecessarily.
regridder_model_to_satellite = xe.Regridder(
    grid_model,
    grid_WODB,
    method='bilinear',
    periodic=True,
)

data_NO3 = data_NO3.transpose('time', 'YC', 'XC')

# Ensure correct dimension order
data_NO3 = data_NO3.transpose("time", "YC", "XC")

# Initialize an empty list to collect regridded time slices
regridded_list = []

for t in range(132):
    print(f"Regridding timestep {t+1}/{data_NO3.sizes['time']}", flush=True)

    # Select single time slice (drop extraneous dimensions)
    no3_at_t = data_NO3.isel(time=t)

    # Regrid this slice
    no3_regridded_t = regridder_model_to_satellite(no3_at_t)

    # Add back the time coordinate
    no3_regridded_t = no3_regridded_t.expand_dims(time=[data_NO3.time.values[t]])

    # Collect
    regridded_list.append(no3_regridded_t)

# Concatenate along time
no3_on_satellite = xr.concat(regridded_list, dim="time")

# Fix metadata if needed

#print('erorr',error)
print('no3_on_satellite',no3_on_satellite)



lon=no3_on_satellite.lon.values
lat=no3_on_satellite.lat.values


#no3_on_satellite, data_no3 = xr.align(no3_on_satellite, data.no3, join='inner')
error = no3_on_satellite.mean('time')*1000 - data.no3.mean('time')

print('erorr',error)
print('no3_on_satellite',no3_on_satellite)


for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-8, 8, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, error.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    ax.contour(lon,lat, data.no3.mean('time').squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

plt.savefig("Supplemental_no3_diff_WODB.png")





####### EOF with no detrending



time_mean_NPP_clim = data_NPP.groupby('time.month').mean(dim='time',skipna=True)
time_mean_NPP_anom = data_NPP.groupby('time.month')-time_mean_NPP_clim
#time_mean_NPP_anom=detrend_dim(time_mean_NPP_anom,dim='time')
time_mean_NPP_anom=time_mean_NPP_anom.coarsen(YC=16, boundary="pad").mean()
time_mean_NPP_anom=time_mean_NPP_anom.coarsen(XC=16, boundary="pad").mean()
time_mean_NPP_anom2=time_mean_NPP_anom/time_mean_NPP_anom.std()
#time_mean_NPP_anom2=time_mean_NPP_anom2.coarsen(XC=16, boundary="pad").mean()

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
plt.savefig("Supplemental_eof_NPP_nodetrending.pdf", bbox_inches='tight')
plt.close()


import matplotlib.pyplot as plt

pc1 = pcs_normalized.isel(mode=0)

slope, pval = pc_trend_rednoise(pc1.time.values, pc1.values)

print("PC1 trend (per year):", slope)
print("p-value:", pval)

# Create a separate figure for PC1
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(12,4), dpi=300)

plt.plot(pc1.time, pc1, color='black', linewidth=2, label="PC1")

# build trend line
#t = (pc1.time.values.astype('datetime64[M]') - pc1.time.values[0].astype('datetime64[M]')).astype(int)/12
#trend_line = slope*t + (pc1.mean() - slope*t.mean())

#plt.plot(pc1.time, trend_line, color='red', linewidth=2, label="Trend")

plt.ylabel("Principal Component 1", fontsize=18)
plt.xlabel("Time", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.grid(True)

# significance label
label = f"Trend = {slope:.3f} yr$^{{-1}}$\np = {pval:.3f}"
if pval < 0.05:
    label += "  (significant)"

plt.text(0.02,0.95,label, transform=plt.gca().transAxes,
         fontsize=14, verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.legend()
plt.tight_layout()
plt.savefig("Supplemental_pc1_NPP_timeseries.pdf")
plt.show()

