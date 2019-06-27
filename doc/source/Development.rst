.. _Development:

Development
-----------

DAFoam contains two main layers: OpenFOAM and Python, and they interact through file input and output.

The OpenFOAM layer is written in C++ and contains libraries and solvers for the discrete adjoint, the documentation of the classes and functions in the OpenFOAM layer is as follows:

`OpenFOAM layer <_static/OpenFOAM/index.html>`_

The Python layer contains wrapper class to control the adjoint solvers and also calls other external modules to perform optimization. The input parameters and the APIs of the Python layer are as follows:

`Python layer <_static/Python/index.html>`_
