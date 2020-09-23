# -*- coding: utf-8 -*-

# Created on Tue Aug 14 11:14:31 2018
#
# Created for the arc3o package
# Main functions for the "operational" ARC3O
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

import numpy as np
import xarray as xr
import datetime 
import arc3o.profile_functions as pf
import arc3o.mask_functions as mf
import arc3o.memls_functions_2D as mf2
import subprocess

import os
if not os.getenv('READTHEDOCS'):
    from pathos.multiprocessing import ProcessingPool as Pool

#########################################

def prep_time(input_data):
    
    """
    This function transforms the date format given by MPI-ESM into a proper
    date format for xarray
    
    Parameters
    ----------
    input_data: xarray.Dataset
        the MPI-ESM xarray.Dataset with the original date format
    
    Returns
    -------
    input_data: xarray.Dataset
        the MPI-ESM xarray.Dataset with the new date format
    """
    
    timelength = input_data['time']

    yy = np.array([u[:4] for u in timelength.values.astype('str')]).astype('int')
    mm = np.array([u[4:6] for u in timelength.values.astype('str')]).astype('int')
    dd = np.array([u[6:8] for u in timelength.values.astype('str')]).astype('int')
    hh0 = np.array([u[9:11] for u in timelength.values.astype('str')]).astype('int')
    hh0[hh0==5] = 50
    hh = (hh0*0.01*24).astype('int')

    time0=[]
    for tt in range(len(yy)):
        time0.append(datetime.datetime(yy[tt],mm[tt],dd[tt],hh[tt],0,0))
    input_data['time'] = np.array(time0,dtype='datetime64[ns]')
    return input_data




def new_outputpath(log,outputpath,existing):
    
    """Create new folders to organize experiments
    
    This function helps to organize different experiments
    
    Parameters
    ----------
    log: str 
        ``'yes'`` if you are starting a new experiment and want a new folder for it, ``'no'`` if you want to continue an
        experiment and want to work in an existing folder
    outputpath: str
        path where you want the folder to be/the folder is
    existing: str
        if your folder exists already, just write the name here, if not you can write ``'default'``
    
    Returns
    -------
    new_dir: str
        prints the directory you are working in now, if new it will have created the directory
    """
    
    if log=='yes':
        now=datetime.datetime.now()
        new_dir = outputpath+str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+'-'+str(now.hour).zfill(2)+str(now.minute).zfill(2)+'/'
        subprocess.call(["mkdir", new_dir])
        print('output directory was created: '+new_dir)
        return new_dir
    elif log=='no':
        new_dir = outputpath+existing+'/'
        print('output directory stays '+new_dir)
        return new_dir



def prep_mask(input_data,write_mask,outputpath,timestep):
    
    """
    Prepare or read out the mask for the seasons and ice types
    This function prepares or reads out the mask for the seasons and ice types
    
    Parameters
    ----------
    input_data: xarray.Dataset
        the MPI-ESM data over the whole time period of interest
    write_mask: str
        ``'yes'`` if you want to write to a file, ``'no'`` if you already have written it to a file before
    outputpath: str
        path where you want the file to be written
    timestep: int
        timestep of data in hours
    
    Returns
    -------
    info_ds: xarray.Dataset
        dataset with masks for seasons and ice types
    'period_masks_assim.nc': netcdf file
        File containing ``info_ds`` in ``outputpath``. Written if *write_mask* is ``'yes'``. Can be found in ``outputpath``.
    
    """
        
    if write_mask == 'yes':
        
        print('------- YOU CHOSE TO WRITE THE MASKS TO NETCDF ---------------')

        sit = input_data['siced']#/input_data['seaice'] #ice depth, echam, 211 #this is the real thickness now
        #sit = sit.where(~np.isnan(sit),0)
        lat = input_data['lat']
        lon = input_data['lon']
        #snifrac = input_data['snifrac'].where(input_data['snifrac']>0,0)
        snow = (input_data['sni']*1000./300.)#not really sure/snifrac#(input_data['snifrac']*input_data['seaice']) #water equivalent of snow on ice, echam, 214, density of 330 in MPIOM, this gets nan if the denominator is 0, density changed to 300 following ECHAM paper 
        snow = snow.where(~np.isnan(snow),0)
        tsi = input_data['tsi'].where(sit>0.)
        timelength = input_data['time']
        print('variables read')

        info_ds = xr.Dataset({'ice_type': (['time','lat','lon'], np.zeros((len(timelength),len(lat),len(lon)))),
                         'season': (['time','lat','lon'], np.zeros((len(timelength),len(lat),len(lon))))
                         },
                         coords={'time': timelength, 'lon': lon, 'lat': lat})

        info_ds['ice_type'] = mf.ice_type_wholeArctic(sit,timestep)
        info_ds['season'] = mf.define_periods(sit,snow,tsi,timestep) 

        info_ds[['ice_type', 'season']].to_netcdf(os.path.abspath(outputpath+'period_masks_assim.nc'),'w') #looks weird in ncview but seems to be ok
        
    elif write_mask == 'no':
        print('------- YOU CHOSE TO READ THE MASKS FROM NETCDF ---------------')

        info_ds = xr.open_dataset(os.path.abspath(outputpath+'period_masks_assim.nc'))    

    return info_ds



def prep_prof(input_data,write_profiles,info_ds,outputpath,yy,e_bias_fyi,e_bias_myi,snow_dens):

    """Prepare or read out the property profiles for use in MEMLS
    
    This function prepares or reads out the property profiles for use in MEMLS.
    
    Parameters
    ----------
    input_data: xarray.Dataset
        the MPI-ESM data over the whole time period of interest
    write_profiles: str
        ``'yes'`` if you want to write to a file, ``'no'`` if you already have written it to a file in ``outputpaht`` before
    info_ds: xarray.Dataset
        dataset with masks for seasons and ice types over whole time period of interest
    outputpath: str
        path where you want the file to be written
    yy: int
        actually stands for yyyymm, defines the year and month we are looking at
    e_bias_fyi: float
        tuning parameters for the first-year ice temperature profile to influence the MEMLS result
    e_bias_myi: float
        tuning parameters for the multiyear ice temperature profile to influence the MEMLS result
    snow_dens: float
        snow density (constant)
    
    Returns
    -------
    profiles1: xarray.Dataset
        profiles assuming snow-covered ice
    profiles2: xarray.Dataset
        profiles assuming bare ice
    'profiles_for_memls_snowno_yyyymm.nc': netcdf file
        Snow-free profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'profiles_for_memls_snowyes_yyyymm.nc': netcdf file
        Snow-covered profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    """

    if write_profiles == 'yes':

        print('------- YOU CHOSE TO WRITE THE PROFILES TO NETCDF ---------------')
        
        sit = input_data['siced']#/input_data['seaice'] 
        #sit = sit.where(~np.isnan(sit),0)
        snifrac = input_data['snifrac'].where(input_data['snifrac']>0,0)
        snow = (input_data['sni']*1000./300.)#not really sure/snifrac#(input_data['snifrac']*input_data['seaice']) #water equivalent of snow on ice, echam, 214 => FIND OUT , density of snow echam
        snow = snow.where(snifrac,0).where(sit)
        tsi = input_data['tsi'].where(sit>0.)   #surface temperature of ice, echam, 102
        print('variables read')
        layer_amount = 11 #1 is bottom, 10 is top, 11 is snow
        
        profiles1, profiles2 = pf.create_profiles(tsi,sit,snow,layer_amount,info_ds,e_bias_fyi,e_bias_myi,snow_dens)
        profiles1[['layer_temperature', 'layer_salinity', 'layer_thickness', 'layer_wetness','layer_density', 'layer_correlation_length', 'layer_fyi_myi_sno_col']].to_netcdf(os.path.abspath(outputpath+'profiles_for_memls_snowyes_'+str(yy)+'.nc'),'w')
        profiles2[['layer_temperature', 'layer_salinity', 'layer_thickness', 'layer_wetness','layer_density', 'layer_correlation_length', 'layer_fyi_myi_sno_col']].to_netcdf(os.path.abspath(outputpath+'profiles_for_memls_snowno_'+str(yy)+'.nc'),'w')
        
    elif write_profiles == 'no':

        print('------- YOU CHOSE TO READ THE PROFILES FROM NETCDF ---------------')
        profiles1 = xr.open_dataset(os.path.abspath(outputpath+'profiles_for_memls_snowyes_'+str(yy)+'.nc'))
        profiles2 = xr.open_dataset(os.path.abspath(outputpath+'profiles_for_memls_snowno_'+str(yy)+'.nc'))
    
    return profiles1,profiles2



def run_memls_2D(profiles,freq):
    
    """Run MEMLS
    
    This function calls MEMLS for multi-dimensional files.

    Parameters
    ----------
    profiles: xarray.Dataset
        either the snow covered or bare ice property profiles
    freq: float
        frequency in GHz

    Returns
    -------
    ds: xarray.Dataset
        Dataset containing TBH, TBV, emissivity H, emissivity V
    """
    
    temp = profiles['layer_temperature']
    dens = profiles['layer_density']
    wet = profiles['layer_wetness']
    si = profiles['layer_fyi_myi_sno_col']
    sal = profiles['layer_salinity']
    thick = profiles['layer_thickness']
    corrlen = profiles['layer_correlation_length']
    
    freq = freq
    
    TBH,TBV,eh,ev = mf2.memls_2D_1freq(freq,thick,temp,wet,dens,corrlen,sal,si)

    print('-------------- PREPARE DATASET -----------------')
    ds = xr.Dataset({'TBH': TBH,
                 'TBV': TBV,
                 'eh':  eh,
                 'ev':  ev    
                 })
    
    print('MEMLS and Dataset finished')
    return ds

#####################################################################################


#def memls_module_yearloop(start_year,end_year,profiles_yes,profiles_no,snifrac,barefrac,outputpath,compute_memls,many_years):
#    for yy in range(start_year,end_year+1):
#        print('YEAR '+str(yy))
#        profiles0_yes = profiles_yes.sel(time=profiles_yes['time.year']==yy)
#        profiles0_no = profiles_no.sel(time=profiles_no['time.year']==yy)
#        if compute_memls == 'yes':
#
#            snowyes = profiles0_yes
#            snowno = profiles0_no.sel(layer_nb=range(1,11))
#
#            print('----- PROFILES WITH SNOW -------')
#            ds1_yes = run_memls_2D(snowyes,6.9)
#            print('----- PROFILES WITHOUT SNOW -------')
#            ds1_no = run_memls_2D(snowno,6.9)
#            print('----- COMBINING BRIGHTNESS TEMPERATURES -------')
#            ds1 = ds1_yes*snifrac + ds1_no*barefrac
#
#            ds1.to_netcdf(os.path.abspath(outputpath+'TB_test_assim_'+str(yy)+'.nc'),'w')
#
#        elif compute_memls == 'no':
#            print('-------------- YOU DO NOT WANT TO COMPUTE MEMLS? ----------------')
#            #ds1 = xr.open_dataset(os.path.abspath(outputpath+'TB_test_assim_'+str(yy)+'.nc'))  
#    
#    if many_years == 'no':
#        cdo.mergetime(input=outputpath+'TB_test_assim_*.nc',output=os.path.abspath(outputpath+'TB_test_assim_'+str(start_year)+'-'+str(end_year)+'.nc'))
#        
#    return 

def memls_module_general(yy,freq_of_int,profiles_yes,profiles_no,snifrac,barefrac,outputpath,compute_memls):
        
    """Combine the brightness temperatures of bare and snow-covered ice
    
    This function wraps around MEMLS to combine the brightness temperatures of bare and snow-covered ice.

    Parameters
    ----------
    yy: int
        actually stands for yyyymm, defines the year and month we are looking at
    freq_of_int: float
        frequency in GHz
    profiles_yes: xarray.Dataset
        Dataset containing property profiles for snow-covered ice
    profiles_no: xarray.Dataset
        Dataset containing property profiles for bare ice
    snifrac: xarray.DataArray
        snow fraction given by MPI-ESM data, already month of interest selected
    barefrac: xarray.DataArray
        bare ice fraction given by MPI-ESM data, already month of interest selected
    outputpath: str
        path where you want the file to be written
    compute_memls: str
        ``'yes'`` if you want to feed profiles to MEMLS and write the result out, ``'no'`` if you do not want to run MEMLS again and
        use previous output

    Returns
    -------
    ds1 : xarray.Dataset
        Dataset containing the weighted ice brightness temperatures and emissivities (``TBH``, ``TBV``, ``eh``, ``ev``).
    'TB_assim_yyyymm_f.nc': netcdf files
        Chunked by months. Ice surface brightness temperatures for cold conditions. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.


    Notes
    -----
    ``f`` in the filename is the rounded value of ``freq_of_int``.

    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!

    """ 
    
    profiles0_yes = profiles_yes
    profiles0_no = profiles_no
    
    if compute_memls == 'yes':

        snowyes = profiles0_yes
        snowno = profiles0_no.sel(layer_nb=range(1,11))

        print('----- PROFILES WITH SNOW -------')
        ds1_yes = run_memls_2D(snowyes,freq_of_int)
        print('----- PROFILES WITHOUT SNOW -------')
        ds1_no = run_memls_2D(snowno,freq_of_int)
        print('----- COMBINING BRIGHTNESS TEMPERATURES -------')
#             print('ds1_yes = ',ds1_yes)
#             print('ds1_no = ',ds1_no)
#             print('barefrac = ',barefrac)
#             print('snifrac = ',snifrac)
        ds1 = ds1_yes*snifrac + ds1_no*barefrac
 
        ds1[['TBH', 'TBV', 'eh', 'ev']].to_netcdf(os.path.abspath(outputpath+'TB_assim_'+str(yy)+'_'+str(round(freq_of_int))+'.nc'),'w')

    elif compute_memls == 'no':
        print('-------------- YOU DO NOT WANT TO COMPUTE MEMLS? THEN WE SKIP THIS SECTION ----------------')
        ds1 = xr.open_dataset(os.path.abspath(outputpath+'TB_assim_'+str(yy)+'_'+str(round(freq_of_int))+'.nc'))  
    
    return 


#########################################
# COMPUTES ICE BRIGHTNESS TEMPERATURE FROM SEASON INFORMATION AND MEMLS COMPUTED TBV
#########################################

def compute_TBVice(info_ds,TB,pol,snow_emis,surf_temp):
    
    """Produce ice surface brightness temperature for all seasons.

    This function combines ice surface brightness temperatures computed through MEMLS (i.e. for cold conditions)
    with ice surface brightness temperatures from other seasons.

    Parameters
    ----------
    info_ds: xarray.Dataset
        dataset containing masks about seasons and ice types
    TB: xarray.DataArray
        array of brightness temperatures in one polarization computed by MEMLS
    pol: str
        ``'V'`` for vertical polarization, ``'H'`` for horizontal polarization
    snow_emis: float
        assign the snow emissivity to ``1`` or ``np.nan`` for melting snow periods
    surf_temp: xarray.DataArray
        snow surface temperature to use for the brightness temperature if snow emissivity is 1

    Returns
    -------
    TBice: xarray.DataArray
        array of ice surface brightness temperature in the given polarization for all seasons

    Notes
    -----
    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!


    """ 

    #computes ice TB from season info and MEMLS computed TBV
    open_water = info_ds['season']==0
    winter = info_ds['season']==1
    snow_melt = info_ds['season']==2
    bare_ice_summer = info_ds['season']==3

    TB_ice = TB.copy()
    TB_ice = TB_ice.where(~open_water,0) #think about if setting this to 0 is wise
    TB_ice = TB_ice.where(~snow_melt,np.nan)
    TB_ice = TB_ice.where(snow_melt,0).where(~snow_melt,snow_emis)*surf_temp+TB_ice.where(~snow_melt,0)   
    if pol=='V':
        TB_ice = TB_ice.where(~bare_ice_summer,262.29+4.49) #we add what the atmospheric effect reduces
    elif pol=='H':
        TB_ice = TB_ice.where(~bare_ice_summer,240.2+4.71) #not totally good value => std ~ 9K
    return TB_ice

############################################

def compute_emisV(info_ds,eice,pol,snow_emis):
    """Produce emissivities for all seasons.
    
    This function combines ice surface emissivities computed through MEMLS (i.e. for cold conditions)
    with ice surface emissivities from other seasons.

    Parameters
    ----------
    info_ds: xarray.Dataset
        dataset containing masks about seasons and ice types
    eice: xarray.DataArray
        array of emissivity in one polarization computed by MEMLS
    pol: str
        ``'V'`` for vertical polarization, ``'H'`` for horizontal polarization
    snow_emis: float
        assign the snow emissivity to ``1`` or ``np.nan`` for melting snow periods

    Returns
    -------
    e_ice: xarray.DataArray
        array of ice emissivities in the given polarization

    Notes
    -----
    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!
    """ 
    
    #computes ice emissivity from season info and MEMLS computed TBV
    open_water = info_ds['season']==0
    winter = info_ds['season']==1
    snow_melt = info_ds['season']==2
    bare_ice_summer = info_ds['season']==3

    e_ice = eice.copy()
    e_ice = e_ice.where(~open_water,0) #think about if setting this to 0 is wise
    e_ice = e_ice.where(~snow_melt,snow_emis)
    if pol=='V':
        #we do not have data about it. We use TB = e*Teff => e = TB/Teff. As Teff we assume that surface is at 0°C
        e_ice = e_ice.where(~bare_ice_summer,(262.+4.49)/273.15) #we add what the atmospheric effect reduces, 
    elif pol=='H':
        #we do not have data about it. We use TB = e*Teff => e = TB/Teff. As Teff we assume that surface is at 0°C
        e_ice = e_ice.where(~bare_ice_summer,(240.2+4.71)/273.15) #same here, not totally happy with value
    return e_ice

########################################

def comp_F(m1,m2,W,pol):

    """Compute the empirical term for any residual non-linear wind variations
    
    This function is necessary for the atmospheric and ocean contribution in :func:`amsr`. It computes
    the empirical term for any residual non-linear wind variations. It is based on Eq. 60 of :cite:`wentz00`. Data for
    6.9 GHz was not available so we use the same values as for 10.7 GHz

    Parameters
    ----------
    m1: xarray.DataArray
        coefficients given by [Wentz & Meissner, 2000]_
    m2: xarray.DataArray
        coefficients given by [Wentz & Meissner, 2000]_
    W: xarray.DataArray
        wind speed in m/s
    pol: str
        ``'V'`` for vertical polarization, ``'H'`` for horizontal polarization

    Returns
    -------
    F: xarray.DataArray
        empirical term for any residual non-linear wind variations
    """ 
    
    if pol == 'H':
        W1 = 7 #m/s
        W2 = 12 #m/s
    elif pol == 'V':
        W1 = 3 #m/s
        W2 = 12 #m/s
    F1 = m1*W.where(W<W1,0)
    #print(F1)
    F2 = m1*W.where((W>=W1) & (W<=W2),0) + (0.5*(m2 - m1) * (W.where((W>=W1) & (W<=W2),W1) - W1)**2 / (W2 - W1))
    #print(F2)
    fill1 = (0.5*(m2 - m1)  *(W2 + W1))*(1/m2)
    F3 = m2*W.where(W>W2,fill1) -  (0.5*(m2 - m1)  *(W2 + W1))
    #print(F3)
    F = F1 + F2 + F3
    return F

##########################################

def amsr(V,W,L,Ta,Ts,TBV_ice,TBH_ice,e_icev,e_iceh,c_ice,freq,slm,mpf): # get rid of H

    """Add the atmospheric and oceanic contribution to the ice surface brightness temperature.

    This function adds the atmospheric and oceanic contribution to the brightness temperature to the ice surface
    brightness temperature, resulting in a brightness temperature for the top of the atmosphere. This function is based
    on equations given in :cite:`wentz00` and is tailored to AMSR2 frequencies. It was extended by
    `C. Burgard <http://www.github.com/ClimateClara>`_ to include melt ponds for the use in ARC3O.

    Parameters
    ----------
    V: xarray.DataArray
        columnar water vapor in mm
    W: xarray.DataArray
        windspeed over water in m/s
    L: xarray.DataArray
        columnar cloud liquid water in mm
    Ta: xarray.DataArray
        ice surface temperature in K
    Ts: xarray.DataArray
        sea surface temperature in K
    TBV_ice: xarray.DataArray
        brightness temperature ice surface vertical polarization in K
    TBH_ice: xarray.DataArray
        brightness temperature ice surface horizontal polarization in K
    e_icev: xarray.DataArray
        ice emissivity vertical polarization
    e_iceh: xarray.DataArray
        ice emissivity horizontal polarization
    c_ice: xarray.DataArray
        ice concentration between 0 and 1
    freq: float
        AMSR frequency of interest in GHz, one of the following: 6.9, 10.7, 18.7, 23.8, 36.5, 50.3, 52.8, 89.0
    slm: xarray.DataArray
        sea-land mask (0 for ocean, 1 for land)
    mpf: xarray.DataArray
        melt pond fraction between 0 and 1

    Returns
    -------
    TBH: xarray.DataArray
        brightness temperature, horizontal polarization, at top of the atmosphere in K
    TBV: xarray.DataArray
        brightness temperature, vertical polarization, at top of the atmosphere in K
    """     
 

    
    theta = 55.0
    
    print('------------- SETTING UP ATMOSPHERIC COEFFICIENTS -------------------------')

    ds0 = xr.Dataset({'frequencies': (['frequency'], np.array([6.93, 10.65, 18.70, 23.80, 36.50, 50.30, 52.80, 89.00])),
                             'b0': (['frequency'], np.array([239.50E+0,  239.51E+0,  240.24E+0,  241.69E+0,  239.45E+0,  242.10E+0,  245.87E+0,  242.58E+0])),
                             'b1': (['frequency'], np.array([213.92E-2,  225.19E-2,  298.88E-2,  310.32E-2,  254.41E-2,  229.17E-2,  250.61E-2,  302.33E-2])),
                             'b2': (['frequency'], np.array([-460.60E-4, -446.86E-4, -725.93E-4, -814.29E-4, -512.84E-4, -508.05E-4, -627.89E-4, -749.76E-4])),
                             'b3': (['frequency'], np.array([457.11E-6,  391.82E-6,  814.50E-6,  998.93E-6,  452.02E-6,  536.90E-6,  759.62E-6,  880.66E-6])),
                             'b4': (['frequency'], np.array([-16.84E-7,  -12.20E-7,  -36.07E-7,  -48.37E-7,  -14.36E-7,  -22.07E-7,  -36.06E-7,  -40.88E-7])),
                             'b5': (['frequency'], np.array([0.50E+0,     0.54E+0,    0.61E+0,    0.20E+0,    0.58E+0,    0.52E+0,    0.53E+0,    0.62E+0])),
                             'b6': (['frequency'], np.array([-0.11E+0,   -0.12E+0,   -0.16E+0,   -0.20E+0,   -0.57E+0,   -4.59E+0,  -12.52E+0,   -0.57E+0])),
                             'b7': (['frequency'], np.array([-0.21E-2,   -0.34E-2,   -1.69E-2,   -5.21E-2,   -2.38E-2,   -8.78E-2,  -23.26E-2,   -8.07E-2])),
                             'ao1': (['frequency'], np.array([8.34E-3,    9.08E-3,   12.15E-3,   15.75E-3,   40.06E-3,  353.72E-3, 1131.76E-3,   53.35E-3])),
                             'ao2': (['frequency'], np.array([-0.48E-4,  -0.47E-4,   -0.61E-4,   -0.87E-4,   -2.00E-4,  -13.79E-4,   -2.26E-4,   -1.18E-4])),
                             'av1': (['frequency'], np.array([0.07E-3,    0.18E-3,    1.73E-3,    5.14E-3,    1.88E-3,    2.91E-3,    3.17E-3,    8.78E-3])),
                             'av2': (['frequency'], np.array([0.00E-5,    0.00E-5,   -0.05E-5,    0.19E-5,    0.09E-5,    0.24E-5,    0.27E-5,    0.80E-5])),
                             'aL1': (['frequency'], np.array([0.0078, 0.0183, 0.0556, 0.0891,  0.2027,  0.3682,  0.4021,  0.9693])),
                             'aL2': (['frequency'], np.array([0.0303, 0.0298, 0.0288, 0.0281,  0.0261,  0.0236,  0.0231,  0.0146])),
                             'aL3': (['frequency'], np.array([0.0007, 0.0027, 0.0113, 0.0188,  0.0425,  0.0731,  0.0786,  0.1506])),
                             'aL4': (['frequency'], np.array([0.0000, 0.0060, 0.0040, 0.0020, -0.0020, -0.0020, -0.0020, -0.0020])),
                             'aL5': (['frequency'], np.array([1.2216, 1.1795, 1.0636, 1.0220,  0.9546,  0.8983,  0.8943,  0.7961])),
                             'r0v': (['frequency'], np.array([-0.27E-3,  -0.32E-3,  -0.49E-3,  -0.63E-3,  -1.01E-3, -1.20E-3, -1.23E-03, -1.53E-3])),
                             'r0h': (['frequency'], np.array([0.54E-3,   0.72E-3,   1.13E-3,   1.39E-3,   1.91E-3,  1.97E-3,  1.97E-03,  2.02E-3])),
                             'r1v': (['frequency'], np.array([-0.21E-4,  -0.29E-4,  -0.53E-4,  -0.70E-4,  -1.05E-4, -1.12E-4, -1.13E-04, -1.16E-4])),
                             'r1h': (['frequency'], np.array([0.32E-4,   0.44E-4,   0.70E-4,   0.85E-4,   1.12E-4,  1.18E-4,  1.19E-04,  1.30E-4])),
                             'r2v': (['frequency'], np.array([-2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5, -2.10E-5, -2.10E-05, -2.10E-5])),
                             'r2h': (['frequency'], np.array([-25.26E-6, -28.94E-6, -36.90E-6, -41.95E-6, -54.51E-6, -5.50E-5, -5.50E-5,  -5.50E-5])),
                             'r3v': (['frequency'], np.array([0.00E-6,   0.08E-6,   0.31E-6,   0.41E-6,   0.45E-6,  0.35E-6,  0.32E-06, -0.09E-6])),
                             'r3h': (['frequency'], np.array([0.00E-6,  -0.02E-6,  -0.12E-6,  -0.20E-6,  -0.36E-6, -0.43E-6, -0.44E-06, -0.46E-6])),
                             'm1v': (['frequency'], np.array([0.00020, 0.00020, 0.00140, 0.00178, 0.00257, 0.00260, 0.00260, 0.00260])),
                             'm1h': (['frequency'], np.array([0.00200, 0.00200, 0.00293, 0.00308, 0.00329, 0.00330, 0.00330, 0.00330])),
                             'm2v': (['frequency'], np.array([0.00690, 0.00690, 0.00736, 0.00730, 0.00701, 0.00700, 0.00700, 0.00700])),
                             'm2h': (['frequency'], np.array([0.00600, 0.00600, 0.00656, 0.00660, 0.00660, 0.00660, 0.00660, 0.00660]))
                            },
                            coords={'frequency': np.array([6.9, 10.7, 18.7, 23.8, 36.5, 50.3, 52.8, 89.0])})

    ds1 = ds0.sel(frequency=freq)

    print('------------- PREPARING THE SPECIALTIES OF MIXED SURFACE -------------------------')
    #sea-ice concentration
    c_ice = c_ice.where(c_ice>0,0)
    c_ice = c_ice.where(c_ice<1,1)

    # actual surface temperature weighting between surface temperature of ice and SST
    Ts_mix = c_ice*Ta + (1.0-c_ice)*Ts


    print('------------- BASIC MODEL FOR ATMOSPHERE -------------------------')

    #eq 9: cosmic background radiation in K
    T_C=2.7 

    #eq 27a: SST that is typical for water vapor V
    Tv = 273.16 + 0.8337*V - 3.029E-5*(V**3.33)
    #eq 27b
    Tv = Tv.where(V<=48,301.16) 

    #G accounts for fact that effective air temperature is typically higher (lower) for the case of unusually warm (cold) water
    a = Ts_mix-Tv
    #eq 27c
    G1 = 1.05 * a.where(np.abs(a)<=20,0) * (1-(a.where(np.abs(a)<=20,0)**2)/1200.0)
    #eq 27d
    G2 = np.sign(a).where(np.abs(a)>20,0)*14
    G = G1+G2

    #eq 26a: effective air temperature for downwelling radiation in K
    TD = ds1['b0'] + ds1['b1']*V + ds1['b2']*V**2 + ds1['b3']*V**3 + ds1['b4']*V**4 + ds1['b5']*G
    #eq 26b: effective air temperature for upwelling radiation in K
    TU = TD + ds1['b6'] + ds1['b7']*V

    #eq 28: vertically integrated oxygen absorption
    AO = ds1['ao1'] + ds1['ao2']*(TD - 270.0)
    #eq 29: vapor absorption
    AV = ds1['av1']*V + ds1['av2']*V**2
    #cloud temperature mean between surface temp and 273, not sure how this is correct but I trust it
    Tl=(Ts_mix+273.0)/2.0
    #eq 33: vertically integrated liquid water absorption
    AL = ds1['aL1']*(1.0 - ds1['aL2']*(Tl-283.0))*L


    #eq 22: total transmittance
    tau = np.exp((-1.0/np.cos(np.radians(theta))) * (AO + AV + AL)) 

    #eq 24: upwelling and downwelling brightness temperatures 
    TBU = TU * (1.0-tau)
    TBD = TD * (1.0-tau)

    print('------------- DIELECTRIC CONSTANT OF SEA WATER AND SPECULAR SEA SURFACE ---------------')

    # salinity
    s = 35.0
    #eq 41: chlorinity in ppm
    C = 0.5536*s
    #eq 42
    delta_T = 25.0 - (Ts-273.15)
    #eq 40: conductivity of sea water
    zeta = 2.03E-2 + 1.27E-4*delta_T + 2.46E-6*delta_T**2 - C*(3.34E-5 - 4.60E-7*delta_T + 4.60E-8*delta_T**2)
    #eq 39: ionic conductivity of sea water in 1/s
    sigma = 3.39E9*(C**0.892)*np.exp(-delta_T*zeta)


    # spread factor
    eta = 0.012 # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    # light speed in cm/s
    light_speed = 3.00E10 
    epsilon_R = 4.44 # this value is from wentz and meisner, 2000, p. 22, generally assumed to be temperature independent
    #eq 36: dielectric constant for distilled water
    eps_S0 = 87.90*np.exp(-0.004585*(Ts-273.15))
    #eq 43: static dielectric constant for sea water and saline solutions
    epsilon_S = eps_S0*(np.exp(-3.45E-3*s + 4.69E-6*s**2 + 1.36E-5*s*(Ts-273.15)))
    #radiation wavelength in cm
    llambda = light_speed/(ds1['frequencies']*1E9)
    #eq 38: relaxation length of (distilled?) water in m
    lamb_R0 =  3.30*np.exp(-0.0346*(Ts-273.15) + 0.00017*(Ts-273.15)**2)
    #eq 44: relaxation length for water in m
    lambda_R = lamb_R0 - (6.54E-3*(1 - 3.06E-2*(Ts-273.15) + 2.0E-4*(Ts-273.15)**2)*s)
    #eq 35: dielectric constant
    epsilon = epsilon_R + ((epsilon_S - epsilon_R) / (1.0 + ((1j*lambda_R) / llambda)**(1.0-eta))) - ((2.0*1j*sigma*llambda) / light_speed)

    #eq.45: v- and h-pol reflectivity coefficients calculated with Fresnel equations
    rho_H = (np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / (np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))
    rho_V = (epsilon * np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / (epsilon * np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))

    #eq46: power reflectivity
    R_0H = np.abs(rho_H)**2
    R_0V = np.abs(rho_V)**2 + (4.887E-8 - 6.108E-8*(Ts-273.0)**3)

    print('------------- WIND-ROUGHENED SEA SURFACE ---------------')

    #eq 57: sea-surface reflectivity = specular power reflectivity + wind-induced component of sea-surface reflectivity
    R_geoH = R_0H - (ds1['r0h'] + ds1['r1h']*(theta-53.0) + ds1['r2h']*(Ts-288.0) + ds1['r3h']*(theta-53.0)*(Ts-288.0))*W
    R_geoV = R_0V - (ds1['r0v'] + ds1['r1v']*(theta-53.0) + ds1['r2v']*(Ts-288.0) + ds1['r3v']*(theta-53.0)*(Ts-288.0))*W

    #empirical term for any residual non-linear wind variations
    F_H = comp_F(ds1['m1h'],ds1['m2h'],W,'H')
    F_V = comp_F(ds1['m1v'],ds1['m2v'],W,'V')

    #eq 49
    R_H = (1 - F_H) * R_geoH
    R_V = (1 - F_V) * R_geoV

    #there are extra formulas for r2, not totally sure what they mean
    #r2v = -2.1E-5
    #r2h = -5.5E-5 + 0.989E-6 * (37-freq) #if freq <= 37
    #r2h = -5.5*10E-5

    #eq.8: surface emissivity given by Kirchhoff's law 
    emis_h = 1-R_H
    emis_v = 1-R_V

    #eq.56: effective slope variance
    if freq >= 37:
        Delta_S2 = 5.22E-3*W
    else:
        Delta_S2 = 5.22E-3 * (1-0.00748*(37.0-ds1['frequencies'])**1.3)*W
    Delta_S2 = Delta_S2.where(Delta_S2 <= 0.069, 0.069)

    print('------------- ATMOSPHERIC RADIATION SCATTERED BY SEA SURFACE ---------------')

    #eq.62: fit parameter
    term = Delta_S2 - 70.0*Delta_S2**3
    OmegaH = (6.2 - 0.001*(37.0-ds1['frequencies'])**2) * term * tau**2.0
    OmegaV = (2.5 + 0.018*(37.0-ds1['frequencies'])) * term * tau**3.4

    #eq.61: scattered sky radiation
    T_BOmegaH = ((1+OmegaH) * (1-tau) * (TD - T_C) + T_C) * R_H 
    T_BOmegaV = ((1+OmegaV) * (1-tau) * (TD - T_C) + T_C) * R_V
    
    print('------------- REFLECTIVITY OF MELT PONDS ---------------')

    #eq 42
    delta_T = 25.0 - (Ts-273.15)
    #eq 40: conductivity of sea water
    zeta = 2.03E-2 + 1.27E-4*delta_T + 2.46E-6*delta_T**2

    # spread factor
    eta = 0.012 # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    # light speed in cm/s
    light_speed = 3.00E10 
    epsilon_R = 4.44 # this value is from wentz and meisner, 2000, p. 22, generally assumed to be temperature independent
    #eq 36: dielectric constant for distilled water
    eps_S0 = 87.90*np.exp(-0.004585*(Ts-273.15))
    #eq 43: static dielectric constant for sea water and saline solutions
    epsilon_S = eps_S0
    #radiation wavelength in cm
    llambda = light_speed/(ds1['frequencies']*1E9)
    #eq 38: relaxation length of (distilled?) water in m
    lamb_R0 =  3.30*np.exp(-0.0346*(Ts-273.15) + 0.00017*(Ts-273.15)**2)
    #eq 44: relaxation length for water in m
    lambda_R = lamb_R0
    #eq 35: dielectric constant
    epsilon = epsilon_R + ((epsilon_S - epsilon_R) / (1.0 + ((1j*lambda_R) / llambda)**(1.0-eta)))

    #eq.45: v- and h-pol reflectivity coefficients calculated with Fresnel equations
    rho_H = (np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / (np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))
    rho_V = (epsilon * np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / (epsilon * np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))

    #eq46: power reflectivity
    R_0H = np.abs(rho_H)**2
    R_0V = np.abs(rho_V)**2 + (4.887E-8 - 6.108E-8*(Ts-273.0)**3)    

    #eq.8: surface emissivity given by Kirchhoff's law 
    emis0h = 1-R_0H
    emis0v = 1-R_0V
    
## not clear how to use
#     print('------------- WIND DIRECTION EFFECTS ---------------')

#     gamma1_v = 7.83*10**-4*W - 2.18*10**-5*W**2
#     gamma1_h = 1.20*10**-3*W - 8.57*10**-5*W**2
#     gamma2_v = -4.46*10**-4*W + 3.0*10**-5*W**2
#     gamma2_h = -8.93*10**-4*W + 3.76*10**-5*W**2

#     if freq > 37:
#         delta_emis_h = 0
#         delta_emis_v = 0

#     elif freq == 36.5 or freq == 18.7:
#         delta_emis_h = gamma1_h*cos(phi) + gamma2_h*cos(2*phi)
#         delta_emis_v = gamma1_v*cos(phi) + gamma2_v*cos(2*phi)

#     elif freq == 10.7:
#         delta_emis_h = 0.82 * (gamma1_h*cos(phi) + gamma2_h*cos(2*phi))
#         delta_emis_v = 0.82 * (gamma1_v*cos(phi) + gamma2_v*cos(2*phi))    

#     elif freq == 6.9:
#         delta_emis_h = 0.62 * (gamma1_h*cos(phi) + gamma2_h*cos(2*phi))
#         delta_emis_v = 0.62 * (gamma1_v*cos(phi) + gamma2_v*cos(2*phi))    

#     emis_h = emis_h + delta_emis_h
#     emis_v = emis_v + delta_emis_v

    print('------------- COMPUTING FINAL BRIGHTNESS TEMPERATURE ---------------')

#     TBH = TBU + tau * ((1.0 - c_ice)*emis_h*Ts + c_ice*TBH_ice + (1.0 - c_ice) * (1.0 - emis_h) * (T_BOmegaH + tau*T_C) + c_ice*(1.0 - e_iceh) * (TBD + tau*T_C))    
#     TBV = TBU + tau * ((1.0 - c_ice)*emis_v*Ts + c_ice*TBV_ice + (1.0 - c_ice) * (1.0 - emis_v) * (T_BOmegaV + tau*T_C) + c_ice*(1.0 - e_icev) * (TBD + tau*T_C))

    #including melt pond fraction: THINK ABOUT WHAT TO USE AS SURF TEMP FOR MELT PONDS    
    TBH = TBU + tau * ((1.0 - c_ice)*emis_h*Ts + (c_ice - mpf*c_ice)*TBH_ice + (mpf*c_ice)*emis0h*273.15 + (1.0 - c_ice) * (1.0 - emis_h) * (T_BOmegaH + tau*T_C) + (c_ice - mpf*c_ice)*(1.0 - e_iceh) * (TBD + tau*T_C) +  (mpf*c_ice) * (1.0 - emis0h) * (TBD + tau*T_C))
    TBV = TBU + tau * ((1.0 - c_ice)*emis_v*Ts + (c_ice - mpf*c_ice)*TBV_ice + (mpf*c_ice)*emis0v*273.15 + (1.0 - c_ice) * (1.0 - emis_v) * (T_BOmegaV + tau*T_C) + (c_ice - mpf*c_ice)*(1.0 - e_icev) * (TBD + tau*T_C) +  (mpf*c_ice) * (1.0 - emis0v) * (TBD + tau*T_C))

    
    TBH = TBH.where(slm==0)
    TBV = TBV.where(slm==0)
    
    return TBH.drop('frequency'),TBV.drop('frequency')

##############################################################################################################

def TB_tot(input_data,info_ds,memls_output,freq,snow_emis,surf_temp):
    
    """Compute the brightness temperature at top of the atmosphere
    
    This function computes the brightness temperature at top of the atmosphere.

    Parameters
    ----------
    input_data: xarray.Dataset
        MPI-ESM data for period of interest (yyyymm)
    info_ds: xarray.Dataset
        dataset of mask for seasons and ice types for period of interest (yyyymm)
    memls_output: xarray.Dataset
        dataset with MEMLS output
    freq: float
        freguency in GHz
    snow_emis: float
        assign the snow emissivity to ``1`` or ``np.nan`` for melting snow periods
    surf_temp: xarray.DataArray
        snow surface temperature to use for the brightness temperature if snow emissivity is 1

    Returns
    -------
    ds: xarray.Dataset
        dataset containing brightness temperatures at both polarizations at top of the atmosphere in K

    Notes
    -----
    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!
    """     
    
    #combines TB_ice with TB_ocean and atmosphere
    V = input_data['qvi'] #columnar water vapor in mm (*0.001 kg/m**3 *1000 to convert into mm)
    W = input_data['wind10'] #wind velocity 
    L = input_data['xlvi'] #columnar liquid water
    SST = input_data['tsw']
    IST = input_data['tsi']
    sic = input_data['seaice']
    #ocean_frac = 1.-sic-data0['slm']
    slm = input_data['slm']
    mpf = input_data['ameltfrac']
    print('variables read')
    
    print('-------------- ADAPT MEMLS OUTPUT TO ALL SEASONS -----------------')
    TBV_ice = compute_TBVice(info_ds,memls_output['TBV'],'V',snow_emis,surf_temp)
    TBH_ice = compute_TBVice(info_ds,memls_output['TBH'],'H',snow_emis,surf_temp) # until no evaluation for H-pol, V-pol is used
    eice_v = compute_emisV(info_ds,memls_output['ev'],'V',snow_emis)
    eice_h = compute_emisV(info_ds,memls_output['eh'],'H',snow_emis) # until no evaluation for H-pol, V-pol is used
    
    TBH,TBV = amsr(V,W,L,IST,SST,TBV_ice,TBH_ice,eice_v.where(np.isnan(eice_v)==False,0),eice_h.where(np.isnan(eice_h)==False,0),sic,freq,slm,mpf)
    
    print('-------------- PREPARE DATASET -----------------')
    ds = xr.Dataset({'TBH': TBH,
                 'TBV': TBV,
                 })
    
    print('TBtot finished')
    return ds

#def satsim_complete(orig_data,start_year,end_year,outputpath,write_mask='no',write_profiles='no',many_years='no',compute_memls='yes'):
#    #take care of the stuff that goes over the whole time period
#    if many_years == 'yes':
#        info_ds = prep_mask(orig_data,write_mask,outputpath)
#        orig_data.close()
#        for yy in range(start_year,end_year+1):
#            year_data = orig_data.sel(time=orig_data['time.year']==yy)
#            snifrac = year_data['snifrac'].where(data['snifrac']>0,0)
#            profiles_yes, profiles_no = prep_prof(year_data,write_profiles,info_ds.sel(time=info_ds['time.year']==yy),outputpath,yy)
#            memls_module(yy,yy,profiles_yes,profiles_no,year_data['snifrac'],year_data['barefrac'],outputpath,compute_memls,many_years) #output is written to file      
#            TB = xr.open_dataset(os.path.abspath(outputpath+'TB_test_assim_'+str(yy)+'.nc'))
#            ds_TB = TB_tot(year_data,info_ds.sel(time=info_ds['time.year']==yy),TB,6.9)
#            ds_TB[['TBH' , 'TBV']].to_netcdf(os.path.abspath(outputpath+'TBtot_test_assim_'+str(yy)+'.nc'),'w')
#    else:
#        info_ds,profiles_yes, profiles_no = prep_fct(orig_data,write_mask,write_profiles,outputpath,[start_year,end_year])
#        memls_module(start_year,end_year,profiles_yes,profiles_no,orig_data['snifrac'],orig_data['barefrac'],outputpath,compute_memls,many_years) #output is written to file
#        TB = xr.open_dataset(os.path.abspath(outputpath+'TB_test_assim_'+str(start_year)+'-'+str(end_year)+'.nc'))
#        ds_TB = TB_tot(orig_data,info_ds,TB,6.9)
#        ds_TB[['TBH' , 'TBV']].to_netcdf(os.path.abspath(outputpath+'TBtot_test_assim_'+str(start_year)+'-'+str(end_year)+'.nc'),'w')
#    return ds_TB

###########################################################################################################################

##################### NEW VERSION TO PARALLELIZE THE COMPUTATION

def satsim_loop(file,yy,mm,info_ds,freq_of_int,e_bias_fyi,e_bias_myi,outputpath,write_profiles,compute_memls,snow_emis,snow_dens):
    
    """Simulate the brightness temperature of an Arctic Ocean surface.

    This function combines all necessary functions for the brightness temperature simulation.

    Parameters
    ----------
    file: str
        path to file containing MPI-ESM data
    yy: int
        year (format yyyy)
    mm: int
        month (format mm)
    info_ds: xarray.Dataset
        dataset of mask for seasons and ice types
    freq_of_int: float
        freguency in GHz
    e_bias_fyi: float
        tuning parameter for the first-year ice temperature profile to influence the MEMLS result
    e_bias_myi: float
        tuning parameter for the multiyear ice temperature profile to influence the MEMLS result
    outputpath: str
        path where files should be written
    write_profiles: str
        ``'yes'`` if you want to write to a file, ``'no'`` if you already have written it to a file in ``outputpath`` before
    compute_memls: str
        ``'yes'`` if you want to feed profiles to MEMLS and write the result out, ``'no'`` if you do not want to run MEMLS again and
        use previous output
    snow_emis: float
        assign the snow emissivity to ``1`` or ``np.nan`` for melting snow periods
    snow_dens: float
        constant snow density to use

    Returns
    -------
    ds_TB: xarray.Dataset
        brightness temperatures at both polarizations at top of the atmosphere in K
    'profiles_for_memls_snowno_yyyymm.nc': netcdf files
        Chunked by months. Snow-free profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'profiles_for_memls_snowyes_yyyymm.nc': netcdf files
        Chunked by months. Snow-covered profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'TB_assim_yyyymm_f.nc': netcdf files
        Chunked by months. Ice surface brightness temperatures for cold conditions. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'TBtot_assim_yyyymm_f.nc': netcdf files
        Chunked by months. Brightness temperatures at the top of atmosphere. Can be found in ``outputpath``.


    Notes
    -----
    ``f`` in the filenames is the rounded ``freq_of_int``.

    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!

    """

    if freq_of_int != 6.9:
        return print('ARC3O has currently only been evaluated for 6.9 GHz, vertical polarization! You need to tweak the code if you want to apply it to other frequencies ;)') 
	
    #function inside the loop for the timesteps
    year_data = xr.open_dataset(file)
    #write time in the right format
    year_data = prep_time(year_data) #commented for sensitivity experiments
    #print('year_data = ',year_data['snifrac'])
    snifrac = year_data['snifrac'].where(year_data['snifrac']>0,0)
    profiles_yes, profiles_no = prep_prof(year_data,write_profiles,info_ds.sel(time=year_data['time']),outputpath,int(str(yy)+str(mm).zfill(2)),e_bias_fyi,e_bias_myi,snow_dens)
    #print('profiles_yes = ',profiles_yes)
    memls_module_general(int(str(yy)+str(mm).zfill(2)),freq_of_int,profiles_yes,profiles_no,snifrac,year_data['barefrac'],outputpath,compute_memls) #output is written to file      
    TB = xr.open_dataset(outputpath+'TB_assim_'+str(yy)+str(mm).zfill(2)+'_'+str(round(freq_of_int))+'.nc')
    ds_TB = TB_tot(year_data,info_ds.sel(time=year_data['time']),TB,freq_of_int,snow_emis,year_data['tsi'])
    ds_TB[['TBH' , 'TBV']].to_netcdf(outputpath+'TBtot_assim_'+str(yy)+str(mm).zfill(2)+'_'+str(round(freq_of_int))+'.nc','w')
    year_data.close()
    return ds_TB

def compute_parallel(start_year,end_year,freq_of_int,e_bias_fyi,e_bias_myi,snow_emis,snow_dens,inputpath,outputpath,file_begin,file_end,info_ds,write_profiles,compute_memls,pool_nb):

    """Parallelize the brightness temperature simulation.

    This function parallelizes the brightness temperature simulation for the different months of a given year.

    Parameters
    ----------
    start_year: int
        first year of interest (format yyyy)
    end_year: int
        last year of interest (format yyyy)
    freq_of_int: float
        freguency in GHz
    e_bias_fyi: float
        tuning parameter for the first-year ice temperature profile to influence the MEMLS result
    e_bias_myi: float
        tuning parameter for the multiyear ice temperature profile to influence the MEMLS result
    snow_emis:
        assign the snow emissivity to ``1`` or ``np.nan`` for melting snow periods
    snow_dens: float
        constant snow density to use
    inputpath: str
        path where to find the MPI-ESM data chunked in month
    outputpath: str
        path where files should be written
    file_begin: str
        how the MPI-ESM data filename starts (before date)
    file_end: str
        how the MPI-ESM data filename ends (after date)
    info_ds: xarray.Dataset
        dataset of mask for seasons and ice types
    write_profiles: str
        ``'yes'`` if you want to compute the property profiles and write them to files,
        ``'no'`` if you if you have already written them out in ``outputpath`` and nothing has changed
    compute_memls: str
        ``'yes'`` if you want to feed profiles to MEMLS and write the result out,
        ``'no'`` if you have already written them out and nothing has changed
    pool_nb: int, optional
        number of parallel pool workers to compute several months parallelly

    Returns
    -------
    'profiles_for_memls_snowno_yyyymm.nc': netcdf files
        Chunked by months. Snow-free profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'profiles_for_memls_snowyes_yyyymm.nc': netcdf files
        Chunked by months. Snow-covered profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'TB_assim_yyyymm_f.nc': netcdf files
        Chunked by months. Ice surface brightness temperatures for cold conditions. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'TBtot_assim_yyyymm_f.nc': netcdf files
        Chunked by months. Brightness temperatures at the top of atmosphere. Can be found in ``outputpath``.

    Notes
    -----
    ``f`` in the filenames is the rounded ``freq_of_int``.

    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!

    """   
    
    for year in range(start_year,end_year+1):
        def f(mm):
            yy = year
            print('-------------'+str(mm).zfill(2)+'/'+str(yy)+'-------------------')
            file = inputpath+file_begin+str(yy)+str(mm).zfill(2)+file_end
            print(file)
            satsim_loop(file,yy,mm,info_ds,freq_of_int,e_bias_fyi,e_bias_myi,outputpath,write_profiles,compute_memls,snow_emis,snow_dens)
            return 

        p = Pool(pool_nb)
        p.map(f, range(1,13))
    return 

def satsim_complete_parallel(orig_data,freq_of_int,start_year,end_year,inputpath,outputpath,file_begin,file_end,timestep=6,write_mask='yes',write_profiles='yes',compute_memls='yes',e_bias_fyi=0.968,e_bias_myi=0.968,snow_emis=1,snow_dens=300.,pool_nb=12):

    """Compute top-of-atmosphere brightness temperature over a long time period.

    This function is the full observation operator and should be used as ARC3O main function.

    Parameters
    ----------
    orig_data: xarray.Dataset
        MPI-ESM data all years merged together
    freq_of_int: float
        freguency in GHz
    start_year: int
        first year of interest (format yyyy)
    end_year: int
        last year of interest (format yyyy)
    inputpath: str
        path where to find the MPI-ESM data chunked in months
    outputpath: str
        path where files should be written
    file_begin: str
        how the MPI-ESM data filename starts (before date)
    file_end: str
        how the MPI-ESM data filename ends (after date)
    timestep: int, optional
        timestep of data in hours; *default is 6*
    write_mask: str, optional
        ``'yes'`` if you want to write to a file, ``'no'`` if you already have written it to a file before; *default is 'yes'*
    write_profiles: str, optional
        ``'yes'`` if you want to compute the property profiles and write them to files,
        ``'no'`` if you if you have already written them out in ``outputpath`` and nothing has changed; *default is 'yes'*
    compute_memls: str, optional
        ``'yes'`` if you want to feed profiles to MEMLS and write the result out,
        ``'no'`` if you have already written them out and nothing has changed; *default is 'yes'*
    e_bias_fyi: float, optional
        tuning parameter for the first-year ice temperature profile to bias-correct the MEMLS result; *default is 0.968*
    e_bias_myi: float, optional
        tuning parameter for the multiyear ice temperature profile to bias-correct the MEMLS result; *default is 0.968*
    snow_emis: float, optional
        assign the snow emissivity to ``1`` or ``np.nan`` for melting snow periods; *default is 1*
    snow_dens: float, optional
        constant snow density to use; *default is 300.*
    pool_nb: int, optional
        number of parallel pool workers to compute several months parallelly, *default is 12*

    Returns
    -------
    'period_masks_assim.nc': netcdf file
        Masks for ice type and seasons. Written if *write_mask* is ``'yes'``. Can be found in ``outputpath``.
    'profiles_for_memls_snowno_yyyymm.nc': netcdf files
        Chunked by months. Snow-free profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'profiles_for_memls_snowyes_yyyymm.nc': netcdf files
        Chunked by months. Snow-covered profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'TB_assim_yyyymm_f.nc': netcdf files
        Chunked by months. Ice surface brightness temperatures for cold conditions. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'TBtot_assim_yyyymm_f.nc': netcdf files
        Chunked by months. Brightness temperatures at the top of atmosphere. Can be found in ``outputpath``.

    Notes
    -----
    ``f`` in the filenames is the rounded ``freq_of_int``.

    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!


    """   
    
    info_ds = prep_mask(orig_data,write_mask,outputpath,timestep)
    orig_data.close()
    compute_parallel(start_year,end_year,freq_of_int,e_bias_fyi,e_bias_myi,snow_emis,snow_dens,inputpath,outputpath,file_begin,file_end,info_ds,write_profiles,compute_memls,pool_nb)
    return

def satsim_complete_1month(orig_data,freq_of_int,yyyy,mm,inputpath,outputpath,file_begin,file_end,timestep=6,write_mask='yes',write_profiles='yes',compute_memls='yes',e_bias_fyi=0.968,e_bias_myi=0.968,snow_emis=1,snow_dens=300.):
    
    """Compute top-of-atmosphere brightness temperature for one single month.
    
    This function is a subset of the full observation operator and should be used as ARC3O main function if you only want to simulate
    brightness temperatures for one given month.

    Parameters
    ----------
    orig_data:
        MPI-ESM data all years merged together
    freq_of_int: float
        freguency in GHz
    yyyy: int
        year of interest (format yyyy)
    mm: int
        month of interest (format mm)
    inputpath: str
        path where to find the MPI-ESM data chunked in months
    outputpath: str
        path where files should be written
    file_begin: str
        how the MPI-ESM data filename starts (before date)
    file_end: str
        how the MPI-ESM data filename ends (after date)
    timestep: int, optional
        timestep of data in hours; *default is 6*
    write_mask: str, optional
        ``'yes'`` if you want to write to a file, ``'no'`` if you already have written it to a file before; *default is 'yes'*
    write_profiles: str, optional
        ``'yes'`` if you want to compute the property profiles and write them to files,
        ``'no'`` if you if you have already written them out in ``outputpath`` and nothing has changed; *default is 'yes'*
    compute_memls: str, optional
        ``'yes'`` if you want to feed profiles to MEMLS and write the result out,
        ``'no'`` if you have already written them out and nothing has changed; *default is 'yes'*
    e_bias_fyi: float, optional
        tuning parameter for the first-year ice temperature profile to bias-correct the MEMLS result; *default is 0.968*
    e_bias_myi: float, optional
        tuning parameter for the multiyear ice temperature profile to bias-correct the MEMLS result; *default is 0.968*
    snow_emis: float, optional
        assign the snow emissivity to ``1`` or ``np.nan`` for melting snow periods; *default is 1*
    snow_dens: float, optional
        constant snow density to use; *default is 300.*
    
    Returns
    -------
    'period_masks_assim.nc': netcdf file
        Masks for ice type and seasons. Written if *write_mask* is ``'yes'``. Can be found in ``outputpath``.
    'profiles_for_memls_snowno_yyyymm.nc': netcdf file
        Snow-free profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'profiles_for_memls_snowyes_yyyymm.nc': netcdf file
        Snow-covered profiles of ice and snow properties. Written if *write_profiles* is ``'yes'``. Can be found in ``outputpath``.
    'TB_assim_yyyymm_f.nc': netcdf file
        Ice surface brightness temperatures for cold conditions. Written if *compute_memls* is ``'yes'``. Can be found in ``outputpath``.
    'TBtot_assim_yyyymm_f.nc': netcdf file
        Brightness temperatures at the top of atmosphere. Can be found in ``outputpath``.

    Notes
    -----
    ``f`` in the filenames is the rounded ``freq_of_int``.

    .. note::
        Please remain aware that the results from ARC3O have only been evaluated for the frequency of **6.9 GHz,
        vertical polarization** at the moment! The use for other frequencies and polarizations is at your own risk!
        Especially, this function will also give out results for H polarization. However, they have not been evaluated yet!

    """   

    info_ds = prep_mask(orig_data,write_mask,outputpath,timestep)
    orig_data.close()
    file = inputpath+file_begin+str(yy)+str(mm).zfill(2)+file_end
    satsim_loop(file,yy,mm,info_ds,freq_of_int,e_bias_fyi,e_bias_myi,outputpath,write_profiles,compute_memls,snow_emis,snow_dens)
    return
    
