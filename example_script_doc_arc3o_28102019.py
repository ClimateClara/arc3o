#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:00:40 2019

Example script to use ARC3O

@author: claraburgard
"""

import xarray as xr
import operation_satsim_functions as satsim

#inputpath for monthly files
inputpath = 'pathtofolder'
#inputpath for the whole time period
inputpath0 = 'pathtofolder'
#outputpath for overarching folder for output files
outputpath0 = 'pathtofolder'

#outputpath for your experiment files
#'yes': create a new path for the output files (will create a new folder in outputpath0, called 
# yyyymmdd-hhmm)
#'no': keeps the path given as third option
outputpath = satsim.new_outputpath('yes',outputpath0,'20190516-1047')


## read in the whole time period
orig_data = xr.open_dataset(inputpath0+'assim_SICCI2_50km_echam6_200211-200812_selcode_Arctic.nc')
#write time in the right format
orig_data = satsim.prep_time(orig_data)

### give the first and last year of your time period
start_year = 2003
end_year = 2008

### explain how monthly file names are built
file_begin = 'assim_SICCI2_50km_echam6_'
file_end = '_selcode_Arctic.nc'

### frequency of interest in GHz (must fit one of the AMSR-E frequencies)
freq_of_int = 6.9

### run the operator
satsim.satsim_complete_parallel(orig_data,
                                freq_of_int,
                                start_year,end_year,
                                inputpath,outputpath,file_begin,file_end,
                                write_mask='yes',write_profiles='yes',
                                compute_memls='yes',
                                e_bias_fyi=0.968,e_bias_myi=0.968)

################################
################################
#### if you only want to run 1 MONTH

# year and month of interest
yy = 2004
mm = 6

### explain how monthly file names are built
file_begin = 'assim_SICCI2_50km_echam6_'
file_end = '_selcode_Arctic.nc'
file_begin = 'assim_SICCI2_50km_echam6_'
file_end = '_selcode_Arctic.nc'

### frequency of interest in GHz (must fit one of the AMSR-E frequencies)
freq_of_int = 6.9

### run the operator
satsim.satsim_complete_1month(orig_data,
                              freq_of_int,
                              yy,mm,
                              inputpath,outputpath,
                              file_begin,file_end,
                              write_mask='no',write_profiles='yes',
                              compute_memls='yes',
                              e_bias_fyi=0.968,e_bias_myi=0.968)



