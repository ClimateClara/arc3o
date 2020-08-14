How to run
==========

Input data
----------

Currently, ARC3O is tailored for model output from ECHAM, the atmospheric module of the Max Planck Institute Earth System Model.

The input data for ARC3O should be divided into monthly files, found in the folder ``inputpath``.
Also, to prepare the seasons and ice types mask, ARC3O needs one file in which all the data is merged, found in the folder ``inputpath0``.

These monthly and overview files should contain fields [lat,lon,time] of the following variables:
	* ``snifrac``: snow fraction on ice [0-1]
	* ``siced``: sea-ice thickness [m]
	* ``sni``: snow water equivalent [m]
	* ``tsi``: surface temperature at surface of snow (if present) or of ice (if no snow) [K]
	* ``qvi``: columnar water vapor [mm]
	* ``wind10``: wind speed [m/s]
	* ``xlvi``: columnar liquid water [mm]
	* ``tsw``: sea-surface temperature [K]
	* ``seaice``: sea-ice concentration [0-1]
	* ``slm``: sea-land mask
	* ``ameltfrac``: melt-pond fraction [0-1]

Running ARC3O
-------------

You can run ARC3O with the function :func:`arc3o.core_functions.satsim_complete_parallel` as follows:

.. code-block:: python

    import xarray as xr
    import arc3o as arc3o

    # inputpath for the monthly chucked files of the climate model output
    inputpath = 'pathtofolder1'
    # inputpath for the merged file of the climate model output for the whole time series
    inputpath0 = 'pathtofolder2'
    # outputpath for output folders of your different experiments
    outputpath0 = 'pathtofolder3'

    # outputpath for your experiment files
    #'yes': create a new path for the output files (will create a new folder in outputpath0, called
    # yyyymmdd-hhmm)
    #'no': keeps the path given in the third option
    outputpath = arc3o.new_outputpath('no',outputpath0,'20190516-1047')


    # read in the whole time period
    orig_data = xr.open_dataset(inputpath0+'assim_SICCI2_50km_echam6_200211-200812_selcode_Arctic.nc')
    # transform the unusual MPI-ESM timestamp into a better readable one
    orig_data = arc3o.prep_time(orig_data)

    # give the first and last year of the time period of your data
    start_year = 2003
    end_year = 2008

    # explain how the monthly chunked file names are built around yyyymm (the ones in inputpath),
    # example files are called 'assim_SICCI2_50km_echam6_yyyymm_selcode_Arctic.nc' where yyyy is the year and mm the month
    file_begin = 'assim_SICCI2_50km_echam6_'
    file_end = '_selcode_Arctic.nc'

    # frequency of interest in GHz (must fit one of the AMSR-E frequencies: 6.9, 10.7, 18.7, 23.8, 36.5, 50.3, 52.8, 89.0)
    freq_of_int = 6.9

    # run ARC3O
    arc3o.satsim_complete_parallel(orig_data,               # climate model output, whole time series
                                freq_of_int,                # frequency of interest
                                start_year,end_year,        # first and last year of the time period of interest
                                inputpath,                  # where to find the monthly chunked climate model output
                                outputpath,                 # where to write out the results
                                file_begin,file_end,        # file name as wrapped around yyyymm for the monthly chunked files
                                timestep=6,                 # timestep of climate model data in hours
                                write_mask='yes',           # 'yes' if you want to compute and write out the ice type and season mask, 'no' if you already have a file 'period_masks_assim.nc' in outputpath
                                write_profiles='yes',       # 'yes' if you want to compute and write out the profiles, 'no' if you already have monthly chunked files 'profiles_for_memls_snowno_yyyymm.nc' and 'profiles_for_memls_snowyes yyyymm.nc' in outputpath
                                compute_memls='yes',        # 'yes' if you want to compute and write out the cold conditions ice surface brightness temperature, 'no' if you already have monthly chunked files 'TB_assim_yyyymm_f.nc' in outputpath
                                e_bias_fyi=0.968,           # factor affecting the temperature profiles to bias-correct the brightness temperature (for first-year ice)
                                e_bias_myi=0.968,           # factor affecting the temperature profiles to bias-correct the brightness temperature (for multiyear ice)
                                snow_emis=1,                # snow emissivity for periods of melting snow
                                snow dens=300.)             # snow density in kg/m3

.. note::

    The process takes up a lot of memory so I recommend using a supercomputer if possible. I am working on a "lighter" version
    but not sure when it will be ready so you'll need to tweak it yourself until then. Sorry :(

If you only want to run one month instead of the whole time series, you can also run ARC3O with :func:`arc3o.core_functions.satsim_complete_1month`:

.. code-block:: python

    import xarray as xr
    import arc3o as arc3o

    # inputpath for the monthly chucked files of the climate model output
    inputpath = 'pathtofolder1'
    # inputpath for the merged file of the climate model output for the whole time series
    inputpath0 = 'pathtofolder2'
    # outputpath for output folders of your different experiments
    outputpath0 = 'pathtofolder3'

    # outputpath for your experiment files
    #'yes': create a new path for the output files (will create a new folder in outputpath0, called
    # yyyymmdd-hhmm)
    #'no': keeps the path given in the third option
    outputpath = arc3o.new_outputpath('no',outputpath0,'20190516-1047')

    # read in the whole time period
    orig_data = xr.open_dataset(inputpath0+'assim_SICCI2_50km_echam6_200211-200812_selcode_Arctic.nc')
    # transform the unusual MPI-ESM timestamp into a better readable one
    orig_data = arc3o.prep_time(orig_data)

    # year and month of interest
    yyyy = 2004
    mm = 6

    # explain how the monthly chunked file names are built around yyyymm (the ones in inputpath),
    # example files are called 'assim_SICCI2_50km_echam6_yyyymm_selcode_Arctic.nc' where yyyy is the year and mm the month
    file_begin = 'assim_SICCI2_50km_echam6_'
    file_end = '_selcode_Arctic.nc'

    ### frequency of interest in GHz (must fit one of the AMSR-E frequencies)
    freq_of_int = 6.9

    ### run the operator
    arc3o.satsim_complete_1month(orig_data,            # climate model output, whole time series
                              freq_of_int,              # frequency of interest
                              yyyy,mm,                  # year and month of interest
                              inputpath,                # where to find the monthly chunked climate model output
                              outputpath,               # where to write out the results
                              file_begin,file_end,      # file name as wrapped around yyyymm for the monthly chunked files
                              timestep=6,               # timestep of climate model data in hours
                              write_mask='yes',         # 'yes' if you want to compute and write out the ice type and season mask, 'no' if you already have a file 'period_masks_assim.nc' in outputpath
                              write_profiles='yes',     # 'yes' if you want to compute and write out the profiles, 'no' if you already have monthly chunked files 'profiles_for_memls_snowno_yyyymm.nc' and 'profiles_for_memls_snowyes yyyymm.nc' in outputpath
                              compute_memls='yes',      # 'yes' if you want to compute and write out the cold conditions ice surface brightness temperature, 'no' if you already have monthly chunked files 'TB_assim_yyyymm_f.nc' in outputpath
                              e_bias_fyi=0.968,         # factor affecting the temperature profiles to bias-correct the brightness temperature (for first-year ice)
                              e_bias_myi=0.968,         # factor affecting the temperature profiles to bias-correct the brightness temperature (for multiyear ice)
                              snow_emis=1,              # snow emissivity for periods of melting snow
                              snow dens=300.)           # snow density in kg/m3



Output
------

The output of ARC3O is written into several netcdf files to ``outputpath``:
    * 'period_masks_assim.nc': Masks for ice type and seasons.
    * 'profiles_for_memls_snowno_yyyymm.nc': Snow-free profiles of ice and snow properties.
    * 'profiles_for_memls_snowyes_yyyymm.nc': Snow-covered profiles of ice and snow properties.
    * 'TB_assim_yyyymm_f.nc': Ice surface brightness temperatures (H and V polarization) for grid cells with ice in cold conditions.
    * 'TBtot_assim_yyyymm_f.nc': Brightness temperatures (H and V polarization) at the top of atmosphere (incl. other seasons than cold conditions and ocean and atmosphere contribution) for all ocean grid cells.

.. note::

	Please remain aware that the assumptions used in ARC3O have only been evaluated for the frequency of 6.9 GHz,
	vertical polarization at the moment! The use for other frequencies and polarizations is at your own risk!

