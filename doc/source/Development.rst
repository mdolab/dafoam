.. _Development:

Development
-----------

DAFoam contains two main layers: OpenFOAM and Python, and they interact through file input and output.

The OpenFOAM layer is written in C++ and contains libraries and solvers for the discrete adjoint, the documentation of the classes and functions in the OpenFOAM layer is as follows:

`OpenFOAM Layer Doxygen <_static/OpenFOAM/index.html>`_

The Python layer contains wrapper class to control the adjoint solvers and also calls other external modules to perform optimization. The input parameters and the APIs of the Python layer are as follows:

`Python Layer Doxygen <_static/Python/index.html>`_

Refer to classes-python-pyDAFoam-PYDAFOAM-aCompleteInputParameterSet() for detailed explanation of the optimization input parameters.
