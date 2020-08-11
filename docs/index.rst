.. arc3o documentation master file, created by
   sphinx-quickstart on Mon Aug 10 11:47:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation about the Arctic Observation Operator!
===================================================================

The Arctic Ocean Observation Operator (ARC3O) computes brightness temperatures at 6.9 GHz, 
vertical polarization, based on climate model output. More information about the motivation, 
structure and evaluation can be found in `Burgard et al., 2020a`_ and `Burgard et al., 2020b`_. 

Currently, it is customized for output of the Max Planck Institute Earth System Model but can be 
used for other models if the variable names are changed accordingly in the ARC3O functions.

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Getting started:
   
   start/about
   start/installation

.. toctree::
   :maxdepth: 2
   :caption: User's Guide:
   
   guide/input
   guide/workflow
   guide/step1
   guide/step2
   guide/step3
   guide/step4
   guide/step5
   
.. toctree::
   :maxdepth: 2
   :caption: Help & References:   
   
   api/arc3o
   literature/references
   literature/publications
    

How to cite ARC3O
-----------------

The detailed description and evaluation of ARC3O is found in `Burgard et al., 2020b`_ and should 
therefore, when used, be cited as follows:

Burgard, C., Notz, D., Pedersen, L. T., and Tonboe, R. T. (2020): "The Arctic Ocean Observation Operator for 6.9 GHz (ARC3O) – Part 2: Development and evaluation", *The Cryosphere*, 14, 2387–2407, https://doi.org/10.5194/tc-14-2387-2020.

.. _`Burgard et al., 2020a`: https://tc.copernicus.org/articles/14/2369/2020/
.. _`Burgard et al., 2020b`: https://tc.copernicus.org/articles/14/2387/2020/


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
