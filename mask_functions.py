#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 14 11:14:31 2018

This script goes through the whole timeseries of data
and defines two masks:
1. for the different ice types: open water (OW,1), first-year ice (FYI,2), multiyear ice (MYI,3)
2. for the different seasons: open water (0), winter (1), melting snow (2), bare summer ice (3)

@author: Clara Burgard
"""
##################################################

import numpy as np
#import matplotlib.pyplot as plt
import xarray as xr
import datetime 
import itertools
import time
import clalib.satsim_analysis_functions as ana

##################################################

def ice_type_wholeArctic(sit,timestep):
    """
    This function defines the mask for ice types
    
	INPUT
	sit : sea-ice thickness in m
	timestep : timestep of your data in h

	OUTPUT
	ice_type : mask defining the different ice types, where:
	1 = open water OW
	2 = first-year ice FYI
	3 = multiyear ice MYI	
	"""
    
    print('FUNCTION TO DEFINE ICE TYPES')
    #initialize ice types
    ice_type = sit * 0 + 2
    #mask out what's not sea ice
    sit_mask = sit>0. 
    #rolling window over a year to check if there was sea ice during the last year
    #if the rolling_mask is less than the amount of days, there was at least one day without ice => FYI, set everything else to MYI
    for lala, lat0 in enumerate(sit_mask['lat']):
        print('loop over latitudes to avoid MemoryError')
        print(lala,'/',len(sit_mask['lat']))
        rolling_obj = sit_mask.sel(lat=lat0).rolling(time=int(365*(24/timestep)))
        print('Step 1 of 3 - might take long')
        rolling_mask = rolling_obj.sum()
        print('Step 2 of 3')
        rolling_mask0 = rolling_mask<int(365*(24/timestep)) 
        print('Step 3 of 3')
        ice_type.loc[dict(lat=lat0)] = ice_type.sel(lat=lat0).where(rolling_mask0, 3) 
    #everything without sea ice is set to OW
    ice_type = ice_type.where(sit_mask, 1) 
    #mask the land
    ice_type = ice_type.where(np.isnan(sit)==False)
    return ice_type

##################################################



def snow_period_masks(tsi,snow):
    
    """
	This function defines when there is snow growth 
	and snow melt 
	
	INPUT
	tsi : sea-ice (or snow) surface temperature in K
	snow : snow thickness in m
	
	OUTPUT
	snowdown_mask : mask defining if the snow depth decreased
				   since the last timestep
	melt_mask : mask defining grid cells where snow decreased and it was 
			   warm enough to be melt
	"""
    
    print('FUNCTION TO DEFINE SNOW GROWTH AND SNOW MELT')
    #initialize snow array
    snow_mask = snow>0.001 #think about this threshold
    #make difference between timesteps
    diff_snow = snow.diff(dim='time')
    #define snowgrowth and snowmelt
    snowgrowth_mask0 = diff_snow>0.001
    snowmelt_mask0 = diff_snow<-0.001
    melt_temp = tsi>=273.
	#mask to define when snow depth increases
    snowup_mask = xr.DataArray(np.logical_and(snowgrowth_mask0.values,snow_mask[:-1:].values),dims=snow.dims,coords=[snow['time'][:-1:],snow.lat,snow.lon])
    #mask to define when snow depth decreases
    snowdown_mask = xr.DataArray(np.logical_and(snowmelt_mask0.values,snow_mask[:-1:].values),dims=snow.dims,coords=[snow['time'][:-1:],snow.lat,snow.lon])
    #mask to define snow melting: snow decreases and the temperature is high enough for melt to happen (otherwise it could be ice to snow formation or sublimation)
    melt_mask = melt_temp & snowdown_mask
    return snowdown_mask, melt_mask


def summer_bareice_mask(sit,snow,timestep):
    
    """
	This function defines areas of bare ice in summer
	
	INPUT
	sit : sea-ice thickness in m
	snow : snow thickness in m
	timestep : timestep of your data in h
	
	OUTPUT
	bareice_summer_mask : mask defining if the grid cells 
	 						 consist of bare ice in summer
	"""
    
    print('FUNCTION TO DEFINE WHERE THERE IS BARE ICE IN SUMMER')
    #prepare mask for ice in summer
    summer_mask = ana.is_summer(sit['time.month'])
    #select ice in summer
    summer = sit.where(summer_mask) 
    #prepare masks identifying where there is ice and snow
    nosnow_mask = snow<0.02
    sit_mask = sit>0.
    #find where there is ice but no snow
    bareice_mask = xr.DataArray(np.logical_and(nosnow_mask.values,sit_mask.values),dims=snow.dims,coords=[snow.time,snow.lat,snow.lon])
    #apply the previous mask to summer only
    bareice_summer = summer.where(bareice_mask) 
    #set nans to False in this mask
    bareice_summer_mask = np.isnan(bareice_summer)==False   
    return bareice_summer_mask



def define_periods(sit,snow,tsi,timestep):
    
    """
	This function combines all season masking functions 
	to have an overall season overview
	
	INPUT
	sit : sea-ice thickness in m
	snow : snow thickness in m
	tsi : sea-ice (or snow) surface temperature in K
	timestep : timestep of your data in h
	
	OUTPUT
	period_masks : mask for all seasons, where
	0 = open water 
	1 = winter
	2 = snow melt
	3 = bare ice in summer	
	"""
    
    print("DEFINING THE MASK FOR THE DIFFERENT 'SEASONS' ")
    #compute the mask for bare ice in summer
    summer_bareice = summer_bareice_mask(sit,snow,timestep)
    #compute mask for snowmelt
    snowmelt = snow_period_masks(tsi,snow)[1]
    #summarizing everything
    sit_mask = sit>0.
    landsea_mask = sit>=0.
    #set periods to 1 as a default
    period_masks = sit * 0 + int(1)
    #set open water to 0
    period_masks = period_masks.where(sit_mask,int(0))
    #set summer bare ice to 3
    period_masks = period_masks.where(~summer_bareice,int(3))
    #set snow melt to 2
    period_masks = period_masks[:-1:].where(~snowmelt,int(2))
    #mask out the land
    period_masks = period_masks.where(landsea_mask)
    return period_masks
   
#################################################################################

