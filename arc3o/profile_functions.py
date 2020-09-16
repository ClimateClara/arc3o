#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Created on Tue Aug 14 11:21:31 2018
#
# Created for the arc3o package
# These functions create property profiles from surface data given along 
# time, latitude and longitude by MPI-ESM 
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

################################################################

import numpy as np
import xarray as xr
import time
from tqdm import tqdm

################################################################

def compute_ice_snow_int_temp(sit,snd,tsi): 
    """
    Compute the snow/ice interface temperature.
    
    This function computes the temperature at the snow-ice interface, inspired from :cite:`semtner76`. 
    
    Parameters
    ----------
    sit : xarray.DataArray
        sea-ice thickness in m
    snd : xarray.DataArray
        snow thickness in m
    tsi : xarray.DataArray
        sea-ice (or snow) surface temperature in K

    Returns
    -------
    T_i : xarray.DataArray
        temperature at snow-ice interface in K
    """
    
    k_s  = 0.31      # thermal conductivity of snow in W/K/m #from MPIOM paper
    k_i  = 2.17      # thermal conductivity of ice in W/K/m #from MPIOM paper
    bottom_temp = tsi*0 + 273.15-1.8 
    T_i = ((tsi*(k_s/snd)) + (bottom_temp*(k_i/sit))) / ((k_s/snd)+(k_i/sit))
    return T_i

def build_temp_profile(surf_temp_ice,empty_temp_prof,winter_mask,fyi_mask,myi_mask,layer_amount,snow_opt,snd,sit,e_bias_fyi,e_bias_myi): #gives the temperature in middle of layer
    """Build the temperature profile.

    This function builds linear temperature profiles over depth between the snow surface temperature and ice surface temperature
    and between  ice surface temperature and bottom freezing temperature (-1.8°C).

    Parameters
    ----------
    surf_temp_ice: xarray.DataArray
        sea-ice (or snow) surface temperature in K
    empty_temp_prof: xarray.DataArray
        empty xarray with the expected dimensions of the output
    winter_mask: xarray.DataArray
        mask telling us which points are in "winter"
    layer_amount: int
        number of layers the profile should have
    snow_opt: str 
        ``'yes'`` if there is snow on top, ``'no'`` if there is no snow on top
    snd: xarray.DataArray
        snow thickness in m
    sit: xarray.DataArray
        sea-ice thickness in m
    e_bias_fyi: xarray.DataArray
        tuning parameter to correct for MEMLS bias in emissivity for first-year ice
    e_bias_myi: xarray.DataArray
        tuning parameter to correct for MEMLS bias in emissivity for multiyear ice

    Returns
    -------
    shorter_prof: xarray.DataArray
        tuned temperature profile over layer_amount in K

    Notes
    -----
    All input data arrays must be broadcastable towards each other!
    """

    print('------------------------------------')
    print('BUILDING TEMPERATURE PROFILES')
    print('------------------------------------')
    
    ## just take winter points
    winter_tsi = surf_temp_ice.where(winter_mask)
    ## define bottom water temperature in K
    bottom_temp = winter_tsi * 0 + 273.15-1.8  
    ## prepare a longer profile to be able to compute the temperature in the middle of the layers instead
    ## of having the bottom temperature exactly at freezing point and the top layer exactly at the
    ## surface temperature
    extended_prof = surf_temp_ice*xr.DataArray(np.ones(layer_amount+1),coords=[('layer_nb', range(1,layer_amount+2))])

#    extended_prof = xr.DataArray(np.zeros((len(empty_temp_prof.time),len(empty_temp_prof.lat),len(empty_temp_prof.lon),layer_amount+1)),
#                                    coords=([('time', empty_temp_prof.time), 
#                                            ('lat', empty_temp_prof.lat), 
#                                            ('lon', empty_temp_prof.lon), 
#                                            ('layer_nb', range(1,layer_amount+2))]))

    #temperature of bottom layer of longer profile is bottom temperature
    extended_prof.loc[dict(layer_nb=1)] = bottom_temp
    
    shorter_prof = empty_temp_prof.copy()
    
    if snow_opt == 'yes':
        print('profiles with snow')
        ## properties in winter only
        winter_sn = snd.where(winter_mask)
        winter_sit = sit.where(winter_mask)
        ## compute the snow-ice interface
        top_ice = compute_ice_snow_int_temp(winter_sit,winter_sn,winter_tsi)
        ## temperature at top layer is the snow surface temperature
        extended_prof.loc[dict(layer_nb=layer_amount)] = top_ice
        ## compute the linear steps
        temp_diff = (top_ice - bottom_temp)/layer_amount
        ## snow layer temperature is mean between snow and ice surface temperature
        shorter_prof.loc[dict(layer_nb=layer_amount)] = (winter_tsi + top_ice)/2 
        ##############################################################
#         #experiment without gradient in the snow!!
#         top_ice = winter_tsi
#         extended_prof.loc[dict(layer_nb=layer_amount)] = top_ice
#         temp_diff = (top_ice - bottom_temp)/layer_amount
#         shorter_prof.loc[dict(layer_nb=layer_amount)] = winter_tsi
        #############################################################
        
    elif snow_opt == 'no':
        print('profiles ignoring snow')
        ## top ice temperature is the surface temperature given by MPI-ESM
        extended_prof.loc[dict(layer_nb=layer_amount)] = winter_tsi
        ## compute linear steps
        temp_diff = (winter_tsi - bottom_temp)/layer_amount
        ## top temperature (= snow) is set to nan
        shorter_prof.loc[dict(layer_nb=layer_amount)] =  np.nan
    
    ## loop over layers and compute temperature at the middle of each layer, using the linear steps
    ## depending on snow_opt being 'yes' or 'no' 
    for ll in tqdm(range(1,layer_amount)):
        time.sleep(3)
        shorter_prof.loc[dict(layer_nb=ll)] = bottom_temp + temp_diff/2 + (ll-1) * temp_diff
    
    shorter_prof_fyi = e_bias_fyi*shorter_prof.where(fyi_mask,0)
    shorter_prof_myi = e_bias_myi*shorter_prof.where(myi_mask,0)
    shorter_prof = shorter_prof_fyi + shorter_prof_myi
    
    return shorter_prof


def build_thickness_profile(sit,snd,empty_thick_prof,winter_mask,layer_amount):
    
    """Build the thickness profile.

    This function builds thickness profiles with equidistant layers over total ice thickness.

    Parameters
    ----------
    sit: xarray.DataArray
        sea-ice thickness in m
    snd: xarray.DataArray
        snow thickness in m
    empty_thick_prof: xarray.DataArray
        empty xarray with the expected dimensions of the output
    winter_mask: xarray.DataArray
        mask telling us which points are in "winter"
    layer_amount: int
        number of layers the profile should have

    Returns
    -------
    empty_thick_prof: xarray.DataArray
        thickness profile over layer_amount in m
    """

    print('------------------------------------')
    print('BUILDING THICKNESS PROFILES')
    print('------------------------------------')
    
    ## only choose winter points
    winter_sit = sit.where(winter_mask)
    ## divide total ice thickness in equidistant layers, we don't take into account the snow
    ## layer here
    thick = winter_sit/(layer_amount-1) 
    ## loop over layers and assign layer thickness
    for ll in tqdm(range(1,layer_amount)):
        time.sleep(3)
        empty_thick_prof.loc[dict(layer_nb=ll)] = thick
    #assign snow thickness to upper layer
    empty_thick_prof.loc[dict(layer_nb=layer_amount)] = snd 
    
    return empty_thick_prof

 
def sal_approx_fy(norm_z):
    
    """Build salinity profile f(depth) for first-year ice.

    This function builds a salinity profile as a function of depth for first-year ice. It is based on Eq. 17 in :cite:`griewank15`.

    Parameters
    ----------
    norm_z: xarray.DataArray
        normalized depth, 1 is bottom and 0 is top

    Returns
    -------
    sal_fy: xarray.DataArray
        salinity profile for first-year ice in g/kg
    """
    
    a=1.0964
    b=-1.0552
    c=4.41272
    sal_fy = norm_z/(a+b*norm_z)+c
    sal_fy = sal_fy.where(norm_z>0,0)
    return sal_fy

def sal_approx_my(norm_z):
    
    """Build salinity profile f(depth) for multiyear ice.

    This function builds a salinity profile as a function of depth for multiyear ice. It is based on Eq. 18 in :cite:`griewank15`.

    Parameters
    ----------
    norm_z: xarray.DataArray
        normalized depth, 1 is bottom and 0 is top

    Returns
    -------
    sal_my: xarray.DataArray
        salinity profile for multiyear ice in g/kg
    """

    a=0.17083
    b=0.92762
    c=0.024516
    norm_z_ok = norm_z.where(norm_z > 0.0001)
    sal_my = norm_z_ok/a + (norm_z_ok/b)**(1/c)
    sal_my = sal_my.where(norm_z>0,0)
    return sal_my


def build_salinity_profile(empty_sal_prof,fyi_mask,myi_mask,winter_mask,layer_amount):
    
    """Combine FYI and MYI salinity profiles

    This function builds salinity profiles according to the ice type, using :func:`sal_approx_fy` and :func:`sal_approx_my`.

    Parameters
    ----------
    empty_sal_prof: xarray.DataArray
        empty xarray with the expected dimensions of the output
    fyi_mask: xarray.DataArray
        mask telling where there is first-year ice
    myi_mask: xarray.DataArray
        mask telling where there is multiyear ice
    winter_mask: xarray.DataArray
        mask telling us which points are in "winter"
    layer_amount: int
        number of layers the profile should have

    Returns
    -------
    tot_sal: xarray.DataArray
        salinity profiles over layer_amount and for different ice types in g/kg
    """

    print('------------------------------------')
    print('BUILDING SALINITY PROFILES')
    print('------------------------------------')
    
    norm_d = empty_sal_prof.where(winter_mask)
    ## compute the normalized depth over layer_amount-1 (ignoring the snow again)
    for ll in tqdm(range(0,layer_amount-1)):
        time.sleep(3)
        norm_d.loc[dict(layer_nb=ll+1)] = 1 - (1/(layer_amount-1)*ll + 1/(2*(layer_amount-1))) #points to the middle of each layer

    ## compute salinity FYI and set everything else to 0
    sal_fyi = sal_approx_fy(norm_d.where(fyi_mask,0))
    ## compute salinity MYI and set everything else to 0
    sal_myi = sal_approx_my(norm_d.where(myi_mask,0))
    ## merge both fields
    tot_sal = sal_fyi+sal_myi
    ## set salinity of snow to 0
    tot_sal.loc[dict(layer_nb=layer_amount)] = 0.
    ## apply winter mask
    tot_sal = tot_sal.where(winter_mask) #currently open water is nan
    return tot_sal 


def build_wetness_profile(empty_wet_prof,winter_mask):
    
    """Build wetness profiles

    This function builds wetness profiles currently everything is set to 0 as the ice brine volume fraction is computed in
    :func:`Vb`.

    Parameters
    ----------
    empty_wet_prof: xarray.DataArray
        empty xarray with the expected dimensions of the output
    winter_mask: xarray.DataArray
        mask telling us which points are in "winter"

    Returns
    -------
    wet: xarray.DataArray
        wetness profiles (everything set to zero)
    """

    print('------------------------------------')
    print('BUILDING WETNESS PROFILES')
    print('------------------------------------')
    wet = empty_wet_prof*0
    wet = wet.where(winter_mask)
    return wet



def Sb(T):
    
    """Compute brine salinity.

    This function computes the brine salinity. It is based on Eq. 39 in :cite:`vant78` and Eq. 3.4 and 3.5 in :cite:`notz05`.

    Parameters
    ----------
    T: xarray.DataArray
        temperature in K or °C

    Returns
    -------
    tot_Sb: xarray.DataArray
        Brine Salinity in g/kg
    """

    #T must be in degrees C!
    print('Sb: checking if it is degrees C')
    if T.max() > 100.:
        T = T-273.15

    ## compute brine salinity depending on the temperature range

    #print('Sb : step 1')
    T1 = T.where((T>=-43.2) & (T<=-36.8))
    Sb1 = 508.18 + 14.535*T1 + 0.2018*T1**2
    Sb1 = Sb1.where((T>=-43.2) & (T<=-36.8),0)

    #print('Sb : step 2')  
    T2 = T.where((T>-36.8) & (T<=-22.9))
    Sb2 = 242.94 + 1.5299*T2 + 0.0429*T2**2
    Sb2 = Sb2.where((T>-36.8) & (T<=-22.9),0)

    #Eq. 3.4 in Notz 2005
    T3 = T.where((T>-22.9) & (T<-8))
    Sb3 = -1.20 - 21.8*T3 - 0.919*T3**2 - 0.0178*T3**3
    Sb3 = Sb3.where((T>-22.9) & (T<-8),0)
	
    #Eq. 3.6 in Notz 2005
    T4 = T.where((T>=-8) & (T<0))
    Sb4 = 1./(0.001-(0.05411/T4)) 
    Sb4 = Sb4.where((T>=-8) & (T<0),0)
    
    #if temperature too high, set to 0
    T5 = T.where((T>=0))
    Sb5 = 0.*T5
    Sb5 = Sb5.where((T>=0),0)
    
    tot_Sb = Sb1 + Sb2 + Sb3 + Sb4 + Sb5
    return tot_Sb

def Vb(S,Sbr):
    
    """Compute brine volume fraction

    This function computes the brine volume fraction. It is based on Eq. 1.5 in :cite:`notz05`.

    Parameters
    ----------
    S: xarray.DataArray
        bulk salinity in g/kg
    Sbr: xarray.DataArray
        brine salinity in g/kg

    Returns
    -------
    bvf: xarray.DataArray
        brine volume fraction (between 0 and 1)
    """
    
    ## Eq. 1.5 from Notz 2005
    bvf = S.where(Sbr>0,1)/Sbr.where(Sbr>0,1)
    bvf = bvf.where((bvf<=1),1)
    return bvf

def icerho(T,S):
    
    """Compute sea-ice density.

    This function computes the sea-ice density. It is based on :cite:`pounder65` and Eq. 3.8 in :cite:`notz05`.

    Parameters
    ----------
    T: xarray.DataArray
        temperature in K
    S: xarray.DataArray
        bulk salinity in g/kg

    Returns
    -------
    rho_tot: xarray.DataArray
        sea-ice density in kg/m3
    """

    #print('calculating density')  
    T = T-273.15
    ## compute density of pure ice (Eq. ??, Pounder 1965, cited in Notz 2005)
    rho_0 = 916.18 - 0.1403*T
    ## compute brine salinity
    Sbr = Sb(T)
    ## compute density of brine (Eq. 3.8, Notz 2005)
    rho_w = 1000.3 + 0.78237*Sbr + 2.8008*(10**-4)*Sbr**2
    ## compute liquid water fraction
    lwf = Vb(S,Sbr)
    ## compute sea-ice density
    rho_tot = lwf*rho_w + (1-lwf)*rho_0
    return rho_tot 
  	

def build_density_profile(empty_dens_prof,temperature,salinity,winter_mask,layer_amount,snow_dens):
    
    """Build density profiles.

    This function builds sea-ice and snow density profiles

    Parameters
    ----------
    empty_dens_prof: xarray.DataArray
        empty xarray with the expected dimensions of the output
    temperature: xarray.DataArray
        temperature profile in K
    salinity: xarray.DaraArray
        salinity profile in g/kg
    winter_mask: xarray.DataArray
        mask telling us which points are in "winter"
    layer_amount: int
        number of layers the profile should have
    snow_dens: float
        constant snow density we want to assign to the snow (usually 300 or 330 kg/m3)

    Returns
    -------
    dens: xarray.DataArray
        density profiles over `layer_amount` taking into account ice and snow
    """

    print('------------------------------------')
    print('BUILDING DENSITY PROFILES')
    print('------------------------------------')
    
    ## compute sea-ice density
    dens = icerho(temperature,salinity)    
    ## set snow density to 330 kg/m3, now to 300 following ECHAM6 on 09.03.19
    dens.loc[dict(layer_nb=layer_amount)] = snow_dens 
    ## choose winter points
    dens = dens.where(winter_mask)
    return dens


def build_corrlen_profile(empty_corrlen_prof,prof_thick,fyi_mask,winter_mask,layer_amount):
    
    """Build correlation length profiles.

    This function builds correlation length profiles. Based on experiments by R.T. Tonboe, the correlation lengths are
    defined as follows:

    * First-year ice upper 20 cm: 0.25 mm
    * First-year ice lower 20 cm: 0.35 mm
    * Multiyear ice: 1.5 mm

    Parameters
    ----------
    empty_corrlen_prof: xarray.DataArray
        empty xarray with the expected dimensions of the output
    prof_thick: xarray.DataArray
        thickness profile in m
    fyi_mask: xarray.DataArray
        mask telling where there is first-year ice
    winter_mask: xarray.DataArray
        mask telling us which points are in "winter"
    layer_amount: int
        number of layers the profile should have

    Returns
    -------
    corrlen: xarray.DataArray
        correlation length over layer_amount in mm
    """

    print('------------------------------------')
    print('BUILDING CORRELATION LENGTH PROFILES')
    print('------------------------------------')
    
    ## intialize profiles
    corrlen = empty_corrlen_prof.copy()
    corrlen1 = empty_corrlen_prof.copy()
    thick = prof_thick.copy()
    ## build masks for different depths for first-year ice
    cum_sit = thick.isel(layer_nb=range(layer_amount-2,-1,-1)).cumsum(dim='layer_nb')
    upper_ice_mask = cum_sit.isel(layer_nb=range(layer_amount-2,-1,-1))<0.20
    lower_ice_mask = cum_sit.isel(layer_nb=range(layer_amount-2,-1,-1))>=0.20
    ## apply correlation lengths to different depth masks
    corrlen1 = corrlen1.isel(layer_nb=range(0,layer_amount-1)).where(~lower_ice_mask,0.25) 
    corrlen1 = corrlen1.isel(layer_nb=range(0,layer_amount-1)).where(~upper_ice_mask,0.35) 
    corrlen1 = corrlen1.isel(layer_nb=range(0,layer_amount-1)).where(fyi_mask,1.5) 
    ## combine ice and snow layers
    corrlen.loc[dict(layer_nb=range(1,layer_amount))] = corrlen1
    corrlen.loc[dict(layer_nb=layer_amount)]  = 0.15
    ## pick only winter sea ice                          
    corrlen = corrlen.where(winter_mask)
    return corrlen


def build_sisn_prof(empty_def_prof,myi_mask,fyi_mask,winter_mask,layer_amount):
    
    """Build snow/FYI/MYI information profiles

    This function builds layer type profiles (1 = snow, 3 = first-year ice, 4 = multiyear ice)

    Parameters
    ----------
    empty_def_prof: xarray.DataArray
        empty xarray with the expected dimensions of the output
    myi_mask: xarray.DataArray
        mask telling where there is multiyear ice
    fyi_mask: xarray.DataArray
        mask telling where there is first-year ice
    winter_mask: xarray.DataArray
        mask telling us which points are in "winter"
    layer_amount: int
        number of layers the profile should have

    Returns
    -------
    sisn: xarray.DataArray
        profiles defining layer type, where 1 = snow, 3 = first-year ice, 4 = multiyear ice
    """

    print('------------------------------------')
    print('BUILDING PROFILES DEFINING ICE AND SNOW IN THE COLUMN')
    print('------------------------------------')
    
    ## set default to 4
    sisn = empty_def_prof*0+4
    ## where not myi, set to 3
    sisn = sisn.where(myi_mask,3)
    ## top layer is snow
    sisn.loc[dict(layer_nb=layer_amount)] = 1
    ## pick only winter points
    sisn = sisn.where(winter_mask)
    return sisn


def create_profiles(surf_temp_ice,sit,snow,layer_amount,info_ds,e_bias_fyi,e_bias_myi,snow_dens):
    
    """Combine all profile informations.

    This is the main profile function. It summarizes all profiles to write them out, for both snow-covered and snow-free profiles.

    Parameters
    ----------
    surf_temp_ice: xarray.DataArray
        sea ice (or snow) surface temperature
    sit: xarray.DataArray
        sea-ice thickness in m
    snow: xarray.DataArray
        snow thickness in m
    layer_amount: int
        number of layers the profile should have
    info_ds: xarray.Dataset
        dataset containing the mask information
    e_bias_fyi: float
        tuning coefficient for first-year ice
    e_bias_myi: float
        tuning coefficient for multiyear ice
    snow_dens: float
        snow density (constant)

    Returns
    -------
    profiles1: xarray.Dataset
        profiles of all properties if covered by snow
    profiles2: xarray.Dataset
        profiles of all properties if bare ice
    """

    print('------------------------------------')
    print('BUILDING INPUT PROFILES FOR MEMLS')
    print('------------------------------------')
    
    ## set the dimensions, taken from the mask file
#    timelength = info_ds['time']
#    lat = info_ds['lat']
#    lon = info_ds['lon']
    
    ## prepare the two xarrays to receive their data
#    profiles1 = xr.Dataset({
#                     'layer_temperature': (['time','lat','lon','layer_nb'], np.zeros((len(timelength),len(lat),len(lon),layer_amount))),
#                     'layer_salinity': (['time','lat','lon','layer_nb'], np.zeros((len(timelength),len(lat),len(lon),layer_amount))),
#                     'layer_thickness': (['time','lat','lon','layer_nb'], np.zeros((len(timelength),len(lat),len(lon),layer_amount))),
#                     'layer_wetness': (['time','lat','lon','layer_nb'], np.zeros((len(timelength),len(lat),len(lon),layer_amount))),
#                     'layer_density': (['time','lat','lon','layer_nb'], np.zeros((len(timelength),len(lat),len(lon),layer_amount))),
#                     'layer_correlation_length': (['time','lat','lon','layer_nb'], np.zeros((len(timelength),len(lat),len(lon),layer_amount))),
#                     'layer_fyi_myi_sno_col': (['time','lat','lon','layer_nb'], np.zeros((len(timelength),len(lat),len(lon),layer_amount))),
#                     },
#                     coords={'time': timelength, 'lon': lon, 'lat': lat, 'layer_nb': range(1,layer_amount+1)})

    res = sit*xr.DataArray(np.ones(layer_amount),coords=[('layer_nb', range(1,layer_amount+1))])

    profiles1 = xr.Dataset({
                 'layer_temperature': (res.dims, np.zeros(res.shape)),
                 'layer_salinity': (res.dims, np.zeros(res.shape)),
                 'layer_thickness': (res.dims, np.zeros(res.shape)),
                 'layer_wetness': (res.dims, np.zeros(res.shape)),
                 'layer_density': (res.dims, np.zeros(res.shape)),
                 'layer_correlation_length': (res.dims, np.zeros(res.shape)),
                 'layer_fyi_myi_sno_col': (res.dims, np.zeros(res.shape)),
                 },
                coords = res.coords) 
    
    
    profiles2 = xr.Dataset({
                     'layer_temperature': (res.dims, np.zeros(res.shape)),
                     'layer_salinity': (res.dims, np.zeros(res.shape)),
                     'layer_thickness': (res.dims, np.zeros(res.shape)),
                     'layer_wetness': (res.dims, np.zeros(res.shape)),
                     'layer_density': (res.dims, np.zeros(res.shape)),
                     'layer_correlation_length': (res.dims, np.zeros(res.shape)),
                     'layer_fyi_myi_sno_col': (res.dims, np.zeros(res.shape)),
                     },
                    coords = res.coords)    
	
	## dummy empty xarray to be used in the profile building functions
    empty_array_prof = xr.DataArray(np.zeros(res.shape),coords = res.coords)
	
	## read out the different masks used here 
    winter_mask = info_ds['season'] == 1
    fyi_mask = info_ds['ice_type'] == 2
    myi_mask = info_ds['ice_type'] == 3

	## fill the empty array with the different property profiles
    profiles1['layer_thickness'] = build_thickness_profile(sit,snow,empty_array_prof.copy(),winter_mask,layer_amount)
    profiles2['layer_thickness'] = profiles1['layer_thickness'].copy()
    profiles1['layer_salinity'] = build_salinity_profile(empty_array_prof.copy(),fyi_mask,myi_mask,winter_mask,layer_amount)
    profiles2['layer_salinity'] = profiles1['layer_salinity'].copy()
    profiles1['layer_wetness'] = build_wetness_profile(empty_array_prof.copy(),winter_mask)
    profiles2['layer_wetness'] = profiles1['layer_wetness'].copy()
    profiles1['layer_correlation_length'] = build_corrlen_profile(empty_array_prof.copy(),profiles1['layer_thickness'],fyi_mask,winter_mask,layer_amount)
    profiles2['layer_correlation_length'] = profiles1['layer_correlation_length'].copy()
    profiles1['layer_fyi_myi_sno_col'] = build_sisn_prof(empty_array_prof.copy(),myi_mask,fyi_mask,winter_mask,layer_amount)
    profiles2['layer_fyi_myi_sno_col'] = profiles1['layer_fyi_myi_sno_col'].copy()    
    
    ## temperature and density depend on if there is a snow cover
    snow_opt = 'yes'
    profiles1['layer_temperature'] = build_temp_profile(surf_temp_ice,empty_array_prof.copy(),winter_mask,fyi_mask,myi_mask,layer_amount,snow_opt,snow,sit,e_bias_fyi,e_bias_myi)
    profiles1['layer_density'] = build_density_profile(empty_array_prof.copy(),profiles1['layer_temperature'],profiles1['layer_salinity'],winter_mask,layer_amount,snow_dens)
    snow_opt = 'no'
    profiles2['layer_temperature'] = build_temp_profile(surf_temp_ice,empty_array_prof.copy(),winter_mask,fyi_mask,myi_mask,layer_amount,snow_opt,snow,sit,e_bias_fyi,e_bias_myi)
    profiles2['layer_density'] = build_density_profile(empty_array_prof.copy(),profiles2['layer_temperature'],profiles2['layer_salinity'],winter_mask,layer_amount,snow_dens)

    return profiles1,profiles2
