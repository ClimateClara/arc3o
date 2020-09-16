.. _installation:

How to install
==============

you can install it via

.. code-block:: bash

  git clone https://github.com/ClimateClara/arc3o.git
  pip install arc3o/

or via


.. code-block:: bash

  pip install git+https://github.com/ClimateClara/arc3o.git

This package is programmed in python 3.6 and should be working with all `python
versions > 3.6`. Additional requirements are numpy, xarray, pandas, tqdm and pathos.

We recommend to install the dependencies via 

.. code-block:: bash
  
  conda install -c conda-forge pandas tqdm pathos 

as they might not work well using ``pip``.
