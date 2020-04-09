#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:42:40 2018

functions that can be used for analysis for SatSim stuff

@author: claraburgard
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from cartopy.util import add_cyclic_point
import datetime
from http.cookiejar import CookieJar
import urllib 
import cmocean

def retrieve_SIC(TBtot,TB_i,TB_w):
    retr_SIC = (TBtot - TB_w)/(TB_i - TB_w)
    retr_SIC[TB_i<=TB_w] = 0.
    return retr_SIC

def is_summer(month):
    return (month >= 4) & (month <= 9)

def is_winter(month):
    logic = []
    for j,mm in enumerate(month):
        if mm in [1,2,3,10,11,12]: #[1,2,3,4,10,11,12]
            logic.append(True)
        else:
            logic.append(False)
    return logic

def is_latewinter(month):
    return (month >= 1) & (month <= 3)

def is_ONDJFMAM(month):
    logic = []
    for j,mm in enumerate(month):
        if mm in [1,2,3,4,5,10,11,12]: #[1,2,3,4,10,11,12]
            logic.append(True)
        else:
            logic.append(False)
    return logic

def is_JFM(month):
    logic = []
    for j,mm in enumerate(month):
        if mm in [1,2,3]: 
            logic.append(True)
        else:
            logic.append(False)
    return logic

def is_AMJ(month):
    logic = []
    for j,mm in enumerate(month):
        if mm in [4,5,6]: 
            logic.append(True)
        else:
            logic.append(False)
    return logic

def is_JAS(month):
    logic = []
    for j,mm in enumerate(month):
        if mm in [7,8,9]: 
            logic.append(True)
        else:
            logic.append(False)
    return logic

def is_OND(month):
    logic = []
    for j,mm in enumerate(month):
        if mm in [10,11,12]: 
            logic.append(True)
        else:
            logic.append(False)
    return logic

def map_npstereo_model(lon,lat,var,time_in,name='TBV',lat_lim=66,vmin=160,vmax=270,cmap=cmocean.cm.thermal):
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    cs = ax.coastlines(resolution='110m', linewidth=0.5)
    plt.pcolormesh(lon,lat,var,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmap)
    ax.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax.set_title(str(time_in.values)[0:16])
    cbar = plt.colorbar()
    cbar.set_label(name, rotation=90)
    return fig

def map_npstereo_compare3(lon,lat,var1,var2,time_in,name='TBV',lat_lim=66,vmin=160,vmax=270,vmindiff=-5,vmaxdiff=5):

    f = plt.figure(figsize=(20, 9))
    f.suptitle(str(time_in.values)[0:16],fontsize=22)

    ax1 = plt.subplot(1, 3, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax1.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax1.pcolormesh(lon,lat,var1,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal)
    ax1.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax1.set_title('SatSim',fontsize=20)
    cbar = f.colorbar(abso0, ax=ax1, shrink=1.0,orientation='horizontal')
    cbar.set_label('TB [K]',fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    
    ax2 = plt.subplot(1, 3, 2, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax2.coastlines(resolution='110m', linewidth=0.5)
    abso = ax2.pcolormesh(lon,lat,var2,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal)
    ax2.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax2.set_title('AMSR-E',fontsize=20)
    cbar = f.colorbar(abso, ax=ax2, shrink=1.0,orientation='horizontal')
    cbar.set_label('TB [K]',fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    ax3 = plt.subplot(1, 3, 3, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax3.coastlines(resolution='110m', linewidth=0.5)
    diff = ax3.pcolormesh(lon,lat,var1-var2,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm)
    ax3.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax3.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax3.set_title('SatSim - AMSR-E',fontsize=20)
    cbar = f.colorbar(diff, ax=ax3, shrink=1.0,orientation='horizontal')
    cbar.set_label('$\Delta$TB [K]',fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    f.tight_layout()
    
    return f

def map_npstereo_compare3_withpoint(lon,lat,var1,var2,time_in,lon0,lat0,name='TBV',lat_lim=66,vmin=160,vmax=270,vmindiff=-5,vmaxdiff=5):

    f = plt.figure(figsize=(20, 9))
    f.suptitle(str(time_in.values)[0:16],fontsize=22)

    ax1 = plt.subplot(1, 3, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax1.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax1.pcolormesh(lon,lat,var1,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal)
    ax1.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax1.set_title('SatSim',fontsize=20)
    cbar = f.colorbar(abso0, ax=ax1, shrink=1.0,orientation='horizontal')
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('TB [K]',fontsize=20)
    ax1.scatter(lon0, lat0, s=50,color='None',edgecolors='k',transform=ccrs.PlateCarree())
    
    ax2 = plt.subplot(1, 3, 2, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax2.coastlines(resolution='110m', linewidth=0.5)
    abso = ax2.pcolormesh(lon,lat,var2,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal)
    ax2.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax2.set_title('AMSR-E',fontsize=20)
    cbar = f.colorbar(abso, ax=ax2, shrink=1.0,orientation='horizontal')
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('TB [K]',fontsize=20)
    ax2.scatter(lon0, lat0, s=50,color='None',edgecolors='k',transform=ccrs.PlateCarree())

    ax3 = plt.subplot(1, 3, 3, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax3.coastlines(resolution='110m', linewidth=0.5)
    diff = ax3.pcolormesh(lon,lat,var1-var2,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm)
    ax3.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax3.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax3.set_title('SatSim - AMSR-E',fontsize=20)
    cbar = f.colorbar(diff, ax=ax3, shrink=1.0,orientation='horizontal')
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$\Delta$TB [K]',fontsize=20)
    ax3.scatter(lon0, lat0, s=50,color='None',edgecolors='k',transform=ccrs.PlateCarree())

    
    f.tight_layout()
    
    return f

def download_AMSR_data(start_year,end_year,outputpath,username,passwd,freq='06V',morning='no'):

    username = username
    password = passwd
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(
    urllib.request.HTTPBasicAuthHandler(password_manager),
    #urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
    #urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
    urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)
        

    year_range = range(start_year,end_year+1)
    
    leap_years = [2004,2008,2012,2016]

    for year in year_range:
        if year in leap_years:
            rangei = range(1,367)
        else:
            rangei = range(1,366)
        
        for day in rangei: 

            real_date = datetime.datetime.strptime(str(year)+' '+str(day), '%Y %j')

            #print(real_date)
            #print(day)

            yy = real_date.year
            mm = real_date.month
            dd = real_date.day

            if morning == 'yes':
                #morning - not sure if we need morning!
                url_morning = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0630.001/"+str(yy)+"."+str(mm).zfill(2)+"."+str(dd).zfill(2)+"/NSIDC-0630-EASE2_N25km-AQUA_AMSRE-"+str(yy)+str(day).zfill(3)+"-"+freq+"-M-GRD-RSS-v1.3.nc"
                try:
                    urllib.request.urlretrieve (url_morning, outputpath+'AMSRE_N25km_'+str(freq)+'_M_'+str(yy)+str(mm).zfill(2)+str(dd).zfill(2)+'.nc')
                except urllib.error.URLError as e:
                    print(url_morning)
                    print(e.reason)

            #evening

            if day > 1:
                new_day = day-1
                new_year = year
            else:
                if year-1 in leap_years:
                    new_day = 366
                else:
                    new_day = 365
                new_year = year-1

            previous_date = datetime.datetime.strptime(str(new_year)+' '+str(new_day), '%Y %j')

            print('previous = '+str(previous_date))
            #print(new_day)

            url_evening = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0630.001/"+str(yy)+"."+str(mm).zfill(2)+"."+str(dd).zfill(2)+"/NSIDC-0630-EASE2_N25km-AQUA_AMSRE-"+str(new_year)+str(new_day).zfill(3)+"-"+str(freq)+"-E-GRD-RSS-v1.3.nc"

            try:
                urllib.request.urlretrieve (url_evening, outputpath+'AMSRE_N25km_'+str(freq)+'_E_'+str(previous_date.year)+str(previous_date.month).zfill(2)+str(previous_date.day).zfill(2)+'.nc')#
            except urllib.error.URLError as e:
                print(url_evening)
                print(e.reason)
    return

def def_grid_AMSR(input_AMSR):
    lats = np.fromfile('/work/mh0033/m300411/SatSim/AMSRE_nc/EASE_grid/EASE2_N25km.lats.720x720x1.double', 
                      dtype=np.float64).reshape((720,720))
    lons = np.fromfile('/work/mh0033/m300411/SatSim/AMSRE_nc/EASE_grid/EASE2_N25km.lons.720x720x1.double', 
                         dtype=np.float64).reshape((720,720))
    input_AMSR['lat'] = (('x','y'),lats)
    input_AMSR['lon'] = (('x','y'),lons)
    return input_AMSR

def monthly_scatter_subplots(obs,mod,minlim,maxlim):
    xx = range(minlim,maxlim)

    fig, ax = plt.subplots(4, 3, sharex='col', sharey='row',figsize=(9,9))

    n = 1
    for i in range(4):
        for j in range(3):
            x = obs['TB'].sel(time=obs['time.month']==n).values.flatten()
            y = mod['TBV'].sel(time=mod['time.month']==n).values.flatten()
            ax[i, j].scatter(x, y, alpha=0.05,s=1,edgecolors='None',c='grey')
            ax[i, j].set_title(str(n))
            ax[i, j].plot(xx,xx,'k.',linewidth=0.5)
            mean_diff = np.nanmean(abs(y-x))
            ax[i, j].text(minlim+5,minlim+5,'Mean diff SatSim-Obs = '+str(np.round(mean_diff,2))+'$\pm$'+str(np.round(np.nanstd(y-x),2))+' K')
            ax[i, j].set_xlim(minlim,maxlim)
            ax[i, j].set_ylim(minlim,maxlim)
            n=n+1
    ax[3,1].set_xlabel('AMSR-E')
    ax[1,0].set_ylabel('SatSim')
    return fig

def monthly_scatter_subplots_TB(obs,mod,minlim,maxlim):
    xx = range(minlim,maxlim)

    fig, ax = plt.subplots(4, 3, sharex='col', sharey='row',figsize=(9,9))

    n = 1
    for i in range(4):
        for j in range(3):
            x = obs.sel(time=obs['time.month']==n).values.flatten()
            y = mod.sel(time=mod['time.month']==n).values.flatten()
            ax[i, j].scatter(x, y, alpha=0.05,s=1,edgecolors='None',c='grey')
            ax[i, j].set_title(str(n))
            ax[i, j].plot(xx,xx,'k.',linewidth=0.5)
            mean_diff = np.nanmean(abs(y-x))
            ax[i, j].text(minlim+5,minlim+5,'Mean diff SatSim-Obs = '+str(np.round(mean_diff,2))+'$\pm$'+str(np.round(np.nanstd(y-x),2))+' K')
            ax[i, j].set_xlim(minlim,maxlim)
            ax[i, j].set_ylim(minlim,maxlim)
            n=n+1
    ax[3,1].set_xlabel('AMSR-E')
    ax[1,0].set_ylabel('SatSim')
    return fig

def map_npstereo_compare3_12rows(lon,lat,var1,var2,name='TBV',lat_lim=66,vmin=160,vmax=270,vmindiff=-5,vmaxdiff=5):

    f = plt.figure(figsize=(15, 75))
    #f = plt.figure()
    #f.suptitle(str(time_in.values)[0:16],fontsize=22)
    
    n = 0
    for mm in range(1,13):
        
        wrap_TBV1, wrap_lon = add_cyclic_point(var1.sel(time=var1['time.month']==mm).mean('time').values,coord=lon,axis=1)
        wrap_TBV2, wrap_lon = add_cyclic_point(var2.sel(time=var2['time.month']==mm).mean('time').values,coord=lon,axis=1)
    
        n = n+1
        ax1 = plt.subplot(12, 3, n, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax1.coastlines(resolution='110m', linewidth=0.5)
        abso0 = ax1.pcolormesh(wrap_lon,lat,wrap_TBV1,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal)
        ax1.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
        ax1.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
        if mm == 1:
            ax1.set_title('Experiment',fontsize=18)
        if mm ==12:
            cbar = f.colorbar(abso0, ax=ax1, shrink=1.0,orientation='horizontal')
            cbar.set_label('TBV [K]',fontsize=18)
            cbar.ax.tick_params(labelsize=18)

        n = n+1
        ax2 = plt.subplot(12, 3, n, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax2.coastlines(resolution='110m', linewidth=0.5)
        abso = ax2.pcolormesh(wrap_lon,lat,var2.sel(time=var1['time.month']==mm).mean('time'),transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal)
        ax2.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
        if mm == 1:
            ax2.set_title('Reference',fontsize=18)
        if mm == 12:
            cbar = f.colorbar(abso, ax=ax2, shrink=1.0,orientation='horizontal')
            cbar.set_label('TBV [K]',fontsize=18)
            cbar.ax.tick_params(labelsize=18)

        n = n+1
        ax3 = plt.subplot(12, 3, n, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax3.coastlines(resolution='110m', linewidth=0.5)
        diff = ax3.pcolormesh(wrap_lon,lat,wrap_TBV1-wrap_TBV2,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm)
        ax3.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
        ax3.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
        if mm == 1:
            ax3.set_title('Experiment - Reference',fontsize=18)
        if mm == 12:
            cbar = f.colorbar(diff, ax=ax3, shrink=1.0,orientation='horizontal')
            cbar.set_label('$\Delta$TBV [K]',fontsize=18)
            cbar.ax.tick_params(labelsize=18)
    
    f.tight_layout()
    
    return f

def monthly_scatter_subplots_exp(obs,mod,minlim,maxlim):
    xx = range(minlim,maxlim)

    fig, ax = plt.subplots(4, 3, sharex=True, sharey=True,figsize=(15,15))

    n = 1
    for i in range(4):
        for j in range(3):
            x = obs.sel(time=obs['time.month']==n).values.flatten()
            y = mod.sel(time=mod['time.month']==n).values.flatten()
            ax[i, j].plot(xx,xx,'k--',linewidth=1)
            ax[i, j].scatter(x, y, alpha=0.2,s=10,edgecolors='None',c='grey')
            ax[i, j].set_title(str(n))
            mean_diff = np.nanmean(abs(y-x))
            ax[i, j].text(minlim+20,minlim+5,'Mean diff Exp-Ref = '+str(np.round(mean_diff,2))+'$\pm$'+str(np.round(np.nanstd(y-x),2))+' K')
            n=n+1
    ax[3,1].set_xlabel('Reference')
    ax[1,0].set_ylabel('Experiment')
    ax[0,0].set_xlim(minlim,maxlim)
    ax[0,0].set_ylim(minlim,maxlim)
    return fig

#### get overview over variable over the 4 seasons for one assim run
def map_npstereo_compare3_4seasons(lon,lat,var,label,legend='yes',lat_lim=45,vmin=160,vmax=270,cmap=cmocean.cm.thermal):

    f = plt.figure(figsize=(10, 2.5))
    #f = plt.figure()
    #f.suptitle(str(time_in.values)[0:16],fontsize=22)
    
    wrap_var_JFM, wrap_lon = add_cyclic_point(var.sel(time=is_JFM(var['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_var_AMJ, wrap_lon = add_cyclic_point(var.sel(time=is_AMJ(var['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_var_JAS, wrap_lon = add_cyclic_point(var.sel(time=is_JAS(var['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_var_OND, wrap_lon = add_cyclic_point(var.sel(time=is_OND(var['time.month'])).mean('time').values,coord=lon,axis=1)


    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpl.path.Path(verts * radius + center)        
            
    
    ax1 = plt.subplot(1, 4, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax1.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax1.pcolormesh(wrap_lon,lat,wrap_var_JFM,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmap,rasterized=True)
    ax1.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax1.set_boundary(circle, transform=ax1.transAxes)

    ax2 = plt.subplot(1, 4, 2, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax2.coastlines(resolution='110m', linewidth=0.5)
    abso = ax2.pcolormesh(wrap_lon,lat,wrap_var_AMJ,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmap,rasterized=True)
    ax2.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax2.set_boundary(circle, transform=ax2.transAxes)

    ax3 = plt.subplot(1, 4, 3, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax3.coastlines(resolution='110m', linewidth=0.5)
    abso = ax3.pcolormesh(wrap_lon,lat,wrap_var_JAS,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmap,rasterized=True)
    ax3.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax3.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax3.set_boundary(circle, transform=ax3.transAxes)
        
    ax4 = plt.subplot(1, 4, 4, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax4.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax4.pcolormesh(wrap_lon,lat,wrap_var_OND,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmap,rasterized=True)
    ax4.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax4.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax4.set_boundary(circle, transform=ax4.transAxes)
    if legend=='yes':
        cbar = f.colorbar(abso0, ax=ax4, shrink=1.0,orientation='vertical')
        cbar.set_label(label,rotation=90)   
    f.tight_layout()
    
    return f

#### get overview over variable over the 4 seasons and the 3 assim runs
def map_npstereo_compare3_seasons(lon,lat,obs,sicci,bt,nt,legend='yes',name='TBV',lat_lim=66,vmin=160,vmax=270,vmindiff=-10,vmaxdiff=10):

    f = plt.figure(figsize=(8, 14))
    #f = plt.figure()
    #f.suptitle(str(time_in.values)[0:16],fontsize=22)
    
    wrap_obs_JFM, wrap_lon = add_cyclic_point(obs.sel(time=is_JFM(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_sicci_JFM, wrap_lon = add_cyclic_point(sicci.sel(time=is_JFM(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff1_JFM, wrap_lon = add_cyclic_point((sicci-obs).sel(time=is_JFM(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_bt_JFM, wrap_lon = add_cyclic_point(bt.sel(time=is_JFM(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff2_JFM, wrap_lon = add_cyclic_point((bt-obs).sel(time=is_JFM(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_nt_JFM, wrap_lon = add_cyclic_point(nt.sel(time=is_JFM(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff3_JFM, wrap_lon = add_cyclic_point((nt-obs).sel(time=is_JFM(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    
    wrap_obs_AMJ, wrap_lon = add_cyclic_point(obs.sel(time=is_AMJ(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_sicci_AMJ, wrap_lon = add_cyclic_point(sicci.sel(time=is_AMJ(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff1_AMJ, wrap_lon = add_cyclic_point((sicci-obs).sel(time=is_AMJ(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_bt_AMJ, wrap_lon = add_cyclic_point(bt.sel(time=is_AMJ(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff2_AMJ, wrap_lon = add_cyclic_point((bt-obs).sel(time=is_AMJ(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_nt_AMJ, wrap_lon = add_cyclic_point(nt.sel(time=is_AMJ(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff3_AMJ, wrap_lon = add_cyclic_point((nt-obs).sel(time=is_AMJ(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    
    wrap_obs_JAS, wrap_lon = add_cyclic_point(obs.sel(time=is_JAS(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_sicci_JAS, wrap_lon = add_cyclic_point(sicci.sel(time=is_JAS(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff1_JAS, wrap_lon = add_cyclic_point((sicci-obs).sel(time=is_JAS(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_bt_JAS, wrap_lon = add_cyclic_point(bt.sel(time=is_JAS(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff2_JAS, wrap_lon = add_cyclic_point((bt-obs).sel(time=is_JAS(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_nt_JAS, wrap_lon = add_cyclic_point(nt.sel(time=is_JAS(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff3_JAS, wrap_lon = add_cyclic_point((nt-obs).sel(time=is_JAS(obs['time.month'])).mean('time').values,coord=lon,axis=1)

    wrap_obs_OND, wrap_lon = add_cyclic_point(obs.sel(time=is_OND(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_sicci_OND, wrap_lon = add_cyclic_point(sicci.sel(time=is_OND(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff1_OND, wrap_lon = add_cyclic_point((sicci-obs).sel(time=is_OND(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_bt_OND, wrap_lon = add_cyclic_point(bt.sel(time=is_OND(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff2_OND, wrap_lon = add_cyclic_point((bt-obs).sel(time=is_OND(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_nt_OND, wrap_lon = add_cyclic_point(nt.sel(time=is_OND(obs['time.month'])).mean('time').values,coord=lon,axis=1)
    wrap_diff3_OND, wrap_lon = add_cyclic_point((nt-obs).sel(time=is_OND(obs['time.month'])).mean('time').values,coord=lon,axis=1)


    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpl.path.Path(verts * radius + center)        
            
    
    ax1 = plt.subplot(7, 4, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax1.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax1.pcolormesh(wrap_lon,lat,wrap_obs_JFM,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax1.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax1.set_boundary(circle, transform=ax1.transAxes)

    ax2 = plt.subplot(7, 4, 2, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax2.coastlines(resolution='110m', linewidth=0.5)
    abso = ax2.pcolormesh(wrap_lon,lat,wrap_obs_AMJ,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax2.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax2.set_boundary(circle, transform=ax2.transAxes)

    ax3 = plt.subplot(7, 4, 3, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax3.coastlines(resolution='110m', linewidth=0.5)
    diff1 = ax3.pcolormesh(wrap_lon,lat,wrap_obs_JAS,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax3.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax3.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax3.set_boundary(circle, transform=ax3.transAxes)

        
    ax4 = plt.subplot(7, 4, 4, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax4.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax4.pcolormesh(wrap_lon,lat,wrap_obs_OND,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax4.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax4.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax4.set_boundary(circle, transform=ax4.transAxes)
    if legend=='yes':
        cbar = f.colorbar(abso0, ax=ax4, shrink=1.0,orientation='vertical')
        cbar.set_label('TBV [K]',rotation=90)   


    ax5 = plt.subplot(7, 4, 5, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax5.coastlines(resolution='110m', linewidth=0.5)
    abso = ax5.pcolormesh(wrap_lon,lat,wrap_sicci_JFM,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax5.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax5.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax5.set_boundary(circle, transform=ax5.transAxes)

    ax6 = plt.subplot(7, 4, 6, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax6.coastlines(resolution='110m', linewidth=0.5)
    diff = ax6.pcolormesh(wrap_lon,lat,wrap_sicci_AMJ,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax6.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax6.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax6.set_boundary(circle, transform=ax6.transAxes)

    ax7 = plt.subplot(7, 4, 7, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax7.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax7.pcolormesh(wrap_lon,lat,wrap_sicci_JAS,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax7.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax7.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax7.set_boundary(circle, transform=ax7.transAxes)

    ax8 = plt.subplot(7, 4, 8, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax8.coastlines(resolution='110m', linewidth=0.5)
    abso = ax8.pcolormesh(wrap_lon,lat,wrap_sicci_OND,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax8.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax8.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax8.set_boundary(circle, transform=ax8.transAxes)
    if legend=='yes':
        cbar = f.colorbar(abso, ax=ax8, shrink=1.0,orientation='vertical')
        cbar.set_label('TBV [K]',rotation=90)   

    ax9 = plt.subplot(7, 4, 9, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax9.coastlines(resolution='110m', linewidth=0.5)
    diff = ax9.pcolormesh(wrap_lon,lat,wrap_diff1_JFM,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax9.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax9.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax9.set_boundary(circle, transform=ax9.transAxes)

    ax10 = plt.subplot(7, 4, 10, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax10.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax10.pcolormesh(wrap_lon,lat,wrap_diff1_AMJ,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax10.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax10.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax10.set_boundary(circle, transform=ax10.transAxes)

    ax11 = plt.subplot(7, 4, 11, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax11.coastlines(resolution='110m', linewidth=0.5)
    abso = ax11.pcolormesh(wrap_lon,lat,wrap_diff1_JAS,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax11.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax11.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax11.set_boundary(circle, transform=ax11.transAxes)
 
    ax12 = plt.subplot(7, 4, 12, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax12.coastlines(resolution='110m', linewidth=0.5)
    diff = ax12.pcolormesh(wrap_lon,lat,wrap_diff1_OND,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax12.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax12.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax12.set_boundary(circle, transform=ax12.transAxes)
    if legend=='yes':
        cbar = f.colorbar(diff, ax=ax12, shrink=1.0,orientation='vertical',extend='both')
        cbar.set_label('$\Delta$TBV [K]',rotation=90)    

    ax13 = plt.subplot(7, 4, 13, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax13.coastlines(resolution='110m', linewidth=0.5)
    abso = ax13.pcolormesh(wrap_lon,lat,wrap_bt_JFM,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax13.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax13.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax13.set_boundary(circle, transform=ax13.transAxes)

    ax14 = plt.subplot(7, 4, 14, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax14.coastlines(resolution='110m', linewidth=0.5)
    diff = ax14.pcolormesh(wrap_lon,lat,wrap_bt_AMJ,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax14.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax14.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax14.set_boundary(circle, transform=ax14.transAxes)

    ax15 = plt.subplot(7, 4, 15, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax15.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax15.pcolormesh(wrap_lon,lat,wrap_bt_JAS,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax15.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax15.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax15.set_boundary(circle, transform=ax15.transAxes)

    ax16 = plt.subplot(7, 4, 16, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax16.coastlines(resolution='110m', linewidth=0.5)
    abso = ax16.pcolormesh(wrap_lon,lat,wrap_bt_OND,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax16.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax16.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax16.set_boundary(circle, transform=ax16.transAxes)
    if legend=='yes':
        cbar = f.colorbar(abso, ax=ax16, shrink=1.0,orientation='vertical')
        cbar.set_label('TBV [K]',rotation=90)   

    ax17= plt.subplot(7, 4, 17, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax17.coastlines(resolution='110m', linewidth=0.5)
    diff = ax17.pcolormesh(wrap_lon,lat,wrap_diff2_JFM,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax17.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax17.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax17.set_boundary(circle, transform=ax17.transAxes)

    ax18 = plt.subplot(7, 4, 18, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax18.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax18.pcolormesh(wrap_lon,lat,wrap_diff2_AMJ,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax18.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax18.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax18.set_boundary(circle, transform=ax18.transAxes)

    ax19 = plt.subplot(7, 4, 19, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax19.coastlines(resolution='110m', linewidth=0.5)
    abso = ax19.pcolormesh(wrap_lon,lat,wrap_diff2_JAS,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax19.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax19.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax19.set_boundary(circle, transform=ax19.transAxes)
 
    ax20 = plt.subplot(7, 4, 20, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax20.coastlines(resolution='110m', linewidth=0.5)
    diff = ax20.pcolormesh(wrap_lon,lat,wrap_diff2_OND,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax20.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax20.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax20.set_boundary(circle, transform=ax20.transAxes)
    if legend=='yes':
        cbar = f.colorbar(diff, ax=ax20, shrink=1.0,orientation='vertical',extend='both')
        cbar.set_label('$\Delta$TBV [K]',rotation=90)    

    ax21 = plt.subplot(7, 4, 21, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax21.coastlines(resolution='110m', linewidth=0.5)
    abso = ax21.pcolormesh(wrap_lon,lat,wrap_nt_JFM,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax21.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax21.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax21.set_boundary(circle, transform=ax21.transAxes)

    ax22 = plt.subplot(7, 4, 22, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax22.coastlines(resolution='110m', linewidth=0.5)
    diff = ax22.pcolormesh(wrap_lon,lat,wrap_nt_AMJ,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax22.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax22.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax22.set_boundary(circle, transform=ax22.transAxes)

    ax23 = plt.subplot(7, 4, 23, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax23.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax23.pcolormesh(wrap_lon,lat,wrap_nt_JAS,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax23.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax23.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax23.set_boundary(circle, transform=ax23.transAxes)

    ax24 = plt.subplot(7, 4, 24, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax24.coastlines(resolution='110m', linewidth=0.5)
    abso = ax24.pcolormesh(wrap_lon,lat,wrap_nt_OND,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=cmocean.cm.thermal,rasterized=True)
    ax24.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax24.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax24.set_boundary(circle, transform=ax24.transAxes)
    if legend=='yes':
        cbar = f.colorbar(abso, ax=ax24, shrink=1.0,orientation='vertical')
        cbar.set_label('TBV [K]',rotation=90)   

    ax25= plt.subplot(7, 4, 25, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax25.coastlines(resolution='110m', linewidth=0.5)
    diff = ax25.pcolormesh(wrap_lon,lat,wrap_diff3_JFM,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax25.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax25.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax25.set_boundary(circle, transform=ax25.transAxes)

    ax26 = plt.subplot(7, 4, 26, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax26.coastlines(resolution='110m', linewidth=0.5)
    abso0 = ax26.pcolormesh(wrap_lon,lat,wrap_diff3_AMJ,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax26.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax26.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax26.set_boundary(circle, transform=ax26.transAxes)

    ax27 = plt.subplot(7, 4, 27, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax27.coastlines(resolution='110m', linewidth=0.5)
    abso = ax27.pcolormesh(wrap_lon,lat,wrap_diff3_JAS,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax27.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax27.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax27.set_boundary(circle, transform=ax27.transAxes)
 
    ax28 = plt.subplot(7, 4, 28, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax28.coastlines(resolution='110m', linewidth=0.5)
    diff = ax28.pcolormesh(wrap_lon,lat,wrap_diff3_OND,transform=ccrs.PlateCarree(),vmax=vmaxdiff,vmin=vmindiff,cmap=mpl.cm.coolwarm,rasterized=True)
    ax28.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
    ax28.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='black')
    ax28.set_boundary(circle, transform=ax28.transAxes)
    if legend=='yes':
        cbar = f.colorbar(diff, ax=ax28, shrink=1.0,orientation='vertical',extend='both')
        cbar.set_label('$\Delta$TBV [K]',rotation=90)    


    
    f.tight_layout()
    
    return f