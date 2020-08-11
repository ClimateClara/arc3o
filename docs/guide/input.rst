Input data
==========

Currently, ARC3O is tailored for model output from ECHAM, the atmospheric module of the Max Planck Institute Earth System Model.

The input data for ARC3O should be divided into monthly files, found in the folder ``inputpath``. 
Also, to prepare the seasons and ice types mask, ARC3O needs one file in which all the data is merged. 

These monthly and overview files should contain fields [``lat``,``lon``,``time``] of the following variables:
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