Step 2: Prepare sea-ice profiles for cold conditions
----------------------------------------------------

In a second step, we prepare the climate model output data for the simulation of the ice surface brightness temperature
for cold conditions through the Microwave Emission Model for Layered Snowpacks (MEMLS, :cite:`wiesmann99` and :cite:`matzler06`).
This is done in the main function with :func:`arc3o.core_functions.prep_prof`.

Based on the climate model sea-ice thickness, snow thickness, and the snow/ice surface temperature, combined with the masks
described in :ref:`step1`, ARC3O adds the dimension *layer_number* to the lat-lon-time arrays. This way, it prepares two sets
of profiles (ten ice layers and one snow layer), one set for snow-covered ice and one set for bare ice (see :func:`arc3o.profile_functions.create_profiles`). These profiles
describe the layer:
    * temperature :func:`arc3o.profile_functions.build_temp_profile`
    * salinity :func:`arc3o.profile_functions.build_salinity_profile`
    * thickness :func:`arc3o.profile_functions.build_thickness_profile`
    * wetness :func:`arc3o.profile_functions.build_wetness_profile`
    * density :func:`arc3o.profile_functions.build_density_profile`
    * correlation length :func:`arc3o.profile_functions.build_corrlen_profile`
    * type :func:`arc3o.profile_functions.build_sisn_profile`

The resulting profiles are written to ``profiles_for_memls_snowno_yyyymm.nc`` and ``profiles_for_memls_snowyes_yyyymm.nc``
in ``outputpath``.
