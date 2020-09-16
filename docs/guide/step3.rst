Step 3: Compute sea-ice surface brightness temperature for cold conditions
--------------------------------------------------------------------------

In a third step, the two sets of profiles prepared in :ref:`step2` are used to simulate the brightness temperature at the surface of
the snow and ice column, through the Microwave Emission Model for Layered Snowpacks (MEMLS, :cite:`wiesmann99` and :cite:`matzler06`).
This results in two sets of brightness temperatures (one set for snow-covered ice and one set for bare ice). They are then
combined, weighted by the bare-ice fraction given by the climate model output. This step is done in the main function with
:func:`arc3o.core_functions.memls_module_general`.

The MEMLS simulation is conducted through calling the function :func:`arc3o.core_functions.run_memls_2D`. More details about
the detailed steps of the brightness temperature simulation can be found in :func:`arc3o.memls_functions_2D.memls_2D_1freq`
and its documentation of :ref:`memls_api`.

The resulting ice surface brightness temperatures for cold conditions are written to ``TB_assim_yyyymm_f.nc`` in  ``outputpath``,
where *f* is the rounded frequency of interest.
