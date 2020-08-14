.. _step1:

Step 1: Prepare masks for season and ice types
----------------------------------------------

In :cite:`burgard20a` and :cite:`burgard20b`, we show that different approaches are needed to simulate the sea-ice surface brightness temperatures
depending on the ice type (first-year or multiyea ice) and on the "season" (cold conditions, melting snow, bare ice in summer). This is why,
in a first step, ARC3O creates masks to divide the grid cells into those criteria. This is done in the main function with
:func:`arc3o.core_functions.prep_mask`.

To divide the grid cells into different ice types, the function :func:`arc3o.mask_functions.ice_type_wholeArctic`
uses the time series of the sea-ice thickness to determine if there was ice in the grid cell in the year preceding each timestep,
thus dividing into open water, first-year and multiyear ice.

To divide the grid cells into the different seasons, the function :func:`arc3o.mask_functions.define_periods` uses information about
sea-ice thickness, snow thickness, and surface temperature to identify periods of melting snow and periods of bare ice in summer. All
other grid cells are defined as "cold conditions".