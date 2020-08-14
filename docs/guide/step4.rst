Step 4: Compute sea-ice surface brightness temperature for all conditions
-------------------------------------------------------------------------

In a fourth step, the ice surface brightness temperatures are extended to other seasons than "cold conditions", i.e.
to the seasons "melting snow conditions" and "bare ice in summer conditions". This step is done in the main function with
:func:`arc3o.core_functions.compute_TBVice` and :func:`arc3o.core_functions.compute_emisV`.

.. note::

    Be aware that, currently, the brightness temperature of "melting snow conditions" and "bare ice in summer conditions" are set to the
    values for 6.9 GHz, vertical polarization, for all frequencies and polarizations! This is because other frequencies and
    polarizations were not evaluated!