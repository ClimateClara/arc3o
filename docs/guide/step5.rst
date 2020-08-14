Step 5: Add sea-ice concentration and atmospheric effect
--------------------------------------------------------

In a fifth step, the contribution of the ocean and melt ponds to the surface brightness temperature (through the sea-ice concentration
the melt-pond fraction, the snow/ice surface temperature, the sea surface temperature and wind speed) is included. The resulting
surface brightness temperature (combining ice and open water brightness temperatures) is used as an input as a
surface brightness temperature for a simple atmospheric radiative transfer model. This radiative transfer model additionally
requires the atmospheric columnar liquid water and water vapor.
Both adding the oceanic contribution and adding the atmospheric contribution are done in the main function with :func:`arc3o.core_functions.amsr`,
based on the geophysical model described in :cite:`wentz00`.

The resulting top-of-the-atmosphere brightness temperatures for all grid cells are written to ``TBtot_assim_yyyymm_f.nc`` in ``outputpath``,
where *f* is the rounded frequency of interest.
