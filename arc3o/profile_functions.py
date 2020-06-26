#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 14 11:21:31 2018

create property profiles from surface data given along 
time, latitude and longitude by MPI-ESM 

@author: Clara Burgard
"""

################################################################

import numpy as np
#import matplotlib.pyplot as plt
import xarray as xr
import datetime 
from multiprocessing import Pool, TimeoutError
import itertools
from dask.diagnostics import ProgressBar
import time
from tqdm import tqdm
import os

################################################################

def compute_ice_snow_int_temp(sit,snd,tsi): 
    
    """
    This function computes the temperature at the snow-ice interface
	inspired from Semtner, 1976 
	
	INPUT
	sit : sea-ice thickness in m
	snd : snow thickness in m
	tsi : sea-ice (or snow) surface temperature in K
	
	OUTPUT
	T_i : temperature at snow-ice interface in K	
    """	
    
    k_s  = 0.31      # thermal conductivity of snow in W/K/m #from MPIOM paper
    k_i  = 2.17      # thermal conductivity of ice in W/K/m #from MPIOM paper
    bottom_temp = tsi*0 + 273.15-1.8 
    T_i = ((tsi*(k_s/snd)) + (bottom_temp*(k_i/sit))) / ((k_s/snd)+(k_i/sit))
    return T_i

def build_temp_profile(surf_temp_ice,empty_temp_prof,winter_mask,fyi_mask,myi_mask,layer_amount,snow_opt,snd,sit,e_bias_fyi,e_bias_myi): #gives the temperature in middle of layer
    
    """
	This function builds linear temperature profiles over depth
	between the snow surface temperature and ice surface temperature
	and between  ice surface temperature and freezing temperature (-1.8°C)
	
	INPUT
	surf_temp_ice : sea-ice (or snow) surface temperature in K
	empty_temp_prof : empty xarray with the expected dimensions of the output
	winter_mask : mask telling us which points are in "winter"
	layer_amount : number of layers the profile should have
	snow_opt : 
		- 'yes' if there is snow on top
        - 'no' if there is no snow on top
	snd : snow thickness in m
	sit : sea-ice thickness in m
	e_bias_fyi and e_bias_myi : tuning parameter to correct for MEMLS bias in emissivity
	
	OUTPUT
	e_bias*shorter_prof : tuned temperature profile over layer_amount in K	
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
    
    """
	This function builds thickness profiles with equidistant layers over total ice thickness. 
	
	INPUT
	sit : sea-ice thickness in m
	snd : snow thickness in m
	empty_thick_prof : empty xarray with the expected dimensions of the output
	winter_mask : mask telling us which points are in "winter"
	layer_amount : number of layers the profile should have
	
	OUTPUT
	empty_thick_prof : thickness profile over layer_amount in m	
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
    
    """
	This function builds a salinity profile as a function of depth
	for first-year ice 
	formula from Griewank and Notz, 2015
	
	INPUT
	norm_z : normalized depth, 1 is bottom and 0 is top
	 
	OUTPUT
	sal_fy : salinity profile for first-year ice in g/kg	
    """
    
    a=1.0964
    b=-1.0552
    c=4.41272
    sal_fy = norm_z/(a+b*norm_z)+c
    sal_fy = sal_fy.where(norm_z>0,0)
    return sal_fy

def sal_approx_my(norm_z):
    
    """
	This function builds a salinity profile as a function of depth
	for multiyear ice 
	formula from Griewank and Notz, 2015
	
	INPUT
	norm_z : normalized depth, 1 is bottom and 0 is top
	 
	OUTPUT
	sal_fy : salinity profile for multiyear ice in g/kg	
	"""
	
    a=0.17083
    b=0.92762
    c=0.024516
    norm_z_ok = norm_z.where(norm_z > 0.0001)
    sal_my = norm_z_ok/a + (norm_z_ok/b)**(1/c)
    sal_my = sal_my.where(norm_z>0,0)
    return sal_my


def build_salinity_profile(empty_sal_prof,fyi_mask,myi_mask,winter_mask,layer_amount):
    
    """
	This function builds salinity profiles according to the ice type and 
	formulas from Griewank and Notz, 2015
	
	INPUT
	empty_sal_prof : empty xarray with the expected dimensions of the output
	fyi_mask : mask telling where there is first-year ice
	myi_mask : mask telling where there is multiyear ice
	winter_mask : mask telling us which points are in "winter"
	layer_amount : number of layers the profile should have
	
	OUTPUT
	tot_sal : salinity profiles over layer_amount and for different ice types in g/kg	
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
    
    """
	This function builds wetness profiles
	currently everything is set to 0
	
	INPUT
	empty_wet_prof : empty xarray with the expected dimensions of the output
	winter_mask : mask telling us which points are in "winter"
	
	OUTPUT
	wet : wetness profiles	
	"""
	
    print('------------------------------------')
    print('BUILDING WETNESS PROFILES')
    print('------------------------------------')
    wet = empty_wet_prof*0
    wet = wet.where(winter_mask)
    return wet



def Sb(T):
    
    """
	This function computes the brine salinity
	formulas from Notz, 2005 
	
	INPUT
	T : temperature in K or °C
	
	OUTPUT
	tot_Sb : Brine Salinity in g/kg	
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
    
    """
	This function computes the liquid water fraction (=brine volume fraction)
	equation from Notz, 2005 
	
	INPUT
	S : bulk salinity in g/kg
	Sbr : Brine salinity in g/kg
	
	OUTPUT
	lwf : liquid water fraction (between 0 and 1)	
	"""
    
    ## Eq. 1.5 from Notz 2005
    lwf = S.where(Sbr>0,1)/Sbr.where(Sbr>0,1)
    lwf = lwf.where((lwf<=1),1)
    return lwf

def icerho(T,S):
    
    """
	This function computes the sea-ice density
	equations from Notz, 2005
	
	INPUT
	T : temperature in K
	S : bulk salinity in g/kg
	
	OUTPUT
	rho_tot : sea-ice density in kg/m3	
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
    
    """
	This function builds sea-ice and snow density profiles 
	
	INPUT
	empty_dens_prof : empty xarray with the expected dimensions of the output
	temperature : temperature profile in K
	salinity : salinity profile in g/kg
	winter_mask : mask telling us which points are in "winter"
	layer_amount : number of layers the profile should have
	snow_dens : constant snow density we want to assign to the snow (usually 300 or 330 kg/m3)
	
	OUTPUT
	dens : density profiles over layer_amount taking into account ice and snow
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
    
    """
	This function builds correlation length profiles 
	
	INPUT
	empty_corrlen_prof : empty xarray with the expected dimensions of the output
	prof_thick : thickness profile in m
	fyi_mask : mask telling where there is first-year ice
	winter_mask : mask telling us which points are in "winter"
	layer_amount : number of layers the profile should have
	
	OUTPUT
	corrlen : correlation length over layer_amount in mm
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
    
    """
	This function builds layer type profiles 
	
	INPUT
	empty_def_prof : empty xarray with the expected dimensions of the output
	myi_mask : mask telling where there is multiyear ice
	fyi_mask : mask telling where there is first-year ice
	winter_mask : mask telling us which points are in "winter"
	layer_amount : number of layers the profile should have
	
	OUTPUT
	sisn : profiles defining layer type, where
	        1 = snow, 3 = first-year ice, 4 = multiyear ice 		  
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
    
    """
	This function summarizes all profiles to write them out
	
	INPUT
	surf_temp_ice : sea ice (or snow) surface temperature
	sit : sea-ice thickness in m
	snow : snow thickness in m
	layer_amount : number of layers the profile should have
	info_ds : file storing the masks
	e_bias_fyi : tuning coefficient for first-year ice
	e_bias_myi : tuning coefficient for multiyear ice
	snow_dens : snow density (constant)

	
	OUTPUT
	profiles1 : profiles of all properties if covered by snow
	profiles2 : profiles of all properties if bare ice		  
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