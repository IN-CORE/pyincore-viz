pyincore-viz
============

**pyincore-viz** is a component of IN-CORE that provides visualization and other utilities for use with **pyincore**.
 

Mac specific notes
------------------

- We use `matplotlib` library to create graphs. There is a Mac specific installation issue addressed at `here (1) <https://stackoverflow.com/questions/4130355/python-matplotlib-framework-under-macosx>`_ and 
  `here (2) <https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>`_. 
  In a nutshell, insert line: `backend : Agg` into `~/.matplotlib/matplotlibrc` file.

Installation with conda
-----------------------

Installing **pyincore-viz** with Conda is officially supported by IN-CORE development team. 

To add `conda-forge <https://conda-forge.org/>`__  channel to your environment, run

.. code-block:: console

   conda config –-add channels conda-forge

To install **pyincore-viz** package, run

.. code-block:: console

   conda install -c in-core pyincore-viz


To update **pyIncore-viz**, run

.. code-block:: console

   conda update -c in-core pyincore-viz


Installation with pip
-----------------------

Installing **pyincore-viz** with pip is **NOT supported** by IN-CORE development team.
Please use pip for installing pyincore-viz at your discretion. 

**Installing pyincore-viz with pip is only tested on linux environment.**

**Prerequisite**

* GDAL C library must be installed to install pyincore-viz. (for Ubuntu, **gdal-bin** and **libgdal-dev**)

To install **pyincore-viz** package, run

.. code-block:: console

   pip install pyincore-viz

Documentation Containter
------------------------

To build the container with the documentation you can use:

.. code-block:: console

   docker build -t doc/pyincore-viz -f Dockerfile.docs .
   docker run -ti -p 8000:80 doc/pyincore-viz


Then check the documentation at `http://localhost:8000/doc/pyincore_viz/ <http://localhost:8000/doc/pyincore_viz/>`_.
