.. DAFoam documentation master file, created by
   sphinx-quickstart on Wed Jun  6 10:06:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DAFoam
======

DAFoam implements Discrete Adjoint for OpenFOAM. 

DAFoam has two major layers: OpenFOAM and Python.
The OpenFOAM layer contains multiple C++ libraries and discrete adjoint solvers.
These adjoint solvers can run as standalone executives for derivative computation.
The Python layer is a high-level Python interface. 
It allows the adjoint solvers to interact with other external modules for optimization.

Contents:

.. toctree::
   :maxdepth: 2

   Installation
   Tutorials
   OpenFOAM_Layer
   Python_Layer


