#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Created on Tue Aug 14 11:14:31 2018
#
# Created for the arc3o package
# These functions go through the whole timeseries of data
# and define two masks:
# 1. for the different ice types: open water (OW,1), first-year ice (FYI,2), maultiyear ice (MYI,3)
# 2. for the different seasons: open water (0), winter (1), melting snow (2), bare summer ice (3)
#
# @author: Clara Burgard, github.com/ClimateClara
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


##################################################

import numpy as np
import xarray as xr


##################################################

def is_summer(month):
    """Filters warm conditions months

    Parameters
    ----------
    month: int or np.array or xarray.DataArray
        month of the year, January is 1, December is 12

    Returns
    -------
    same type as 'month': int or np.array or xarray.DataArray
        ``True`` if month is between April (4) and September (9), ``False`` if month is between October (10) and March (3)"""

    return (month >= 4) & (month <= 9)


def ice_type_wholeArctic(sit, timestep):
    """Prepare mask for ice types
    
    This function defines the mask for ice types

    Parameters
    ----------
    sit: xarray.DataArray
        sea-ice thickness in m
    timestep: integer
        timestep of your data in h

    Returns
    -------
    ice_type: xarray.DataArray
        mask defining the different ice types, where: 1 = open water OW, 2 = first-year ice FYI, 3 = multiyear ice MYI
    """

    print('FUNCTION TO DEFINE ICE TYPES')
    # initialize ice types
    ice_type = sit * 0 + 2
    # mask out what's not sea ice
    sit_mask = sit > 0.
    # rolling window over a year to check if there was sea ice during the last year
    # if the rolling_mask is less than the amount of days, there was at least one day without ice => FYI, set everything else to MYI
    for lala, lat0 in enumerate(sit_mask['lat']):
        print('loop over latitudes to avoid MemoryError')
        print(lala, '/', len(sit_mask['lat']))
        rolling_obj = sit_mask.sel(lat=lat0).rolling(time=int(365 * (24 / timestep)))
        print('Step 1 of 3 - might take long')
        rolling_mask = rolling_obj.sum()
        print('Step 2 of 3')
        rolling_mask0 = rolling_mask < int(365 * (24 / timestep))
        print('Step 3 of 3')
        ice_type.loc[dict(lat=lat0)] = ice_type.sel(lat=lat0).where(rolling_mask0, 3)
        # everything without sea ice is set to OW
    ice_type = ice_type.where(sit_mask, 1)
    # mask the land
    ice_type = ice_type.where(np.isnan(sit) == False)
    return ice_type


##################################################


def snow_period_masks(tsi, snow):
    """Identify snow growth and snow melt.

    This function defines when there is snow growth and snow melt

    Parameters
    ----------
    tsi: xarray.DataArray
        sea-ice (or snow) surface temperature in K
    snow: xarray.DataArray
        snow thickness in m

    Returns
    -------
    snowdown_mask: xarray.DataArray
        mask defining if the snow depth decreased since the last timestep
    melt_mask: xarray.DataArray
        mask defining grid cells where snow decreased and it was warm enough to be melt
    """

    print('FUNCTION TO DEFINE SNOW GROWTH AND SNOW MELT')
    # initialize snow array
    snow_mask = snow > 0.001  # think about this threshold
    # make difference between timesteps
    diff_snow = snow.diff(dim='time')
    # define snowgrowth and snowmelt
    snowgrowth_mask0 = diff_snow > 0.001
    snowmelt_mask0 = diff_snow < -0.001
    melt_temp = tsi >= 273.
    # mask to define when snow depth increases
    snowup_mask = xr.DataArray(np.logical_and(snowgrowth_mask0.values, snow_mask[:-1:].values), dims=snow.dims,
                               coords=[snow['time'][:-1:], snow.lat, snow.lon])
    # mask to define when snow depth decreases
    snowdown_mask = xr.DataArray(np.logical_and(snowmelt_mask0.values, snow_mask[:-1:].values), dims=snow.dims,
                                 coords=[snow['time'][:-1:], snow.lat, snow.lon])
    # mask to define snow melting: snow decreases and the temperature is high enough for melt to happen (otherwise it could be ice to snow formation or sublimation)
    melt_mask = melt_temp & snowdown_mask
    return snowdown_mask, melt_mask


def summer_bareice_mask(sit, snow, timestep):
    """Identify areas of bare ice in summer.

    This function defines areas of bare ice in summer.

    Parameters
    ----------
    sit: xarray.DataArray
        sea-ice thickness in m
    snow: xarray.DataArray
        snow thickness in m
    timestep: int
        timestep of your data in h

    Returns
    -------
    bareice_summer_mask: xarray.DataArray
        mask defining if the grid cells consist of bare ice in summer
    """

    print('FUNCTION TO DEFINE WHERE THERE IS BARE ICE IN SUMMER')
    # prepare mask for ice in summer
    summer_mask = is_summer(sit['time.month'])
    # select ice in summer
    summer = sit.where(summer_mask)
    # prepare masks identifying where there is ice and snow
    nosnow_mask = snow < 0.02
    sit_mask = sit > 0.
    # find where there is ice but no snow
    bareice_mask = xr.DataArray(np.logical_and(nosnow_mask.values, sit_mask.values), dims=snow.dims,
                                coords=[snow.time, snow.lat, snow.lon])
    # apply the previous mask to summer only
    bareice_summer = summer.where(bareice_mask)
    # set nans to False in this mask
    bareice_summer_mask = np.isnan(bareice_summer) == False
    return bareice_summer_mask


def define_periods(sit, snow, tsi, timestep):
    """Build masks for different seasons.

    This function combines all season masking functions to have an overall season overview.

    Parameters
    ----------
    sit: xarray.DataArray
        sea-ice thickness in m
    snow: xarray.DataArray
        snow thickness in m
    tsi: xarray.DataArray
        sea-ice (or snow) surface temperature in K
    timestep: int
        timestep of your data in h

    Returns
    -------
    period_masks: xarray.DataArray
        mask for all seasons, where 0 = open water, 1 = cold conditions, 2 = snow melt, 3 = bare ice in summer
    """

    print("DEFINING THE MASK FOR THE DIFFERENT 'SEASONS' ")
    # compute the mask for bare ice in summer
    summer_bareice = summer_bareice_mask(sit, snow, timestep)
    # compute mask for snowmelt
    snowmelt = snow_period_masks(tsi, snow)[1]
    # summarizing everything
    sit_mask = sit > 0.
    landsea_mask = sit >= 0.
    # set periods to 1 as a default
    period_masks = sit * 0 + int(1)
    # set open water to 0
    period_masks = period_masks.where(sit_mask, int(0))
    # set summer bare ice to 3
    period_masks = period_masks.where(~summer_bareice, int(3))
    # set snow melt to 2
    period_masks = period_masks[:-1:].where(~snowmelt, int(2))
    # mask out the land
    period_masks = period_masks.where(landsea_mask)
    return period_masks

#################################################################################
