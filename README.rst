DAFoam
======

DAFoam contains multiple discrete adjoint solvers for OpenFOAM. These adjoint solvers run as standalone executives to compute derivatives. DAFoam also has a python interface that allows the adjoint solvers to interact with external modules for design optimization. DAFoam has the following features:

- It allows rapid discrete adjoint development with adding only O(100) lines of codes.
- It supports design optimization for a wide range of disciplines (aerodynamics, heat transfer, structures, hydrodynamics, radiations, etc.).
- It implements an efficient discrete adjoint approach that has competitive speed, scalability, accuracy, and compatibility.

Installation
------------

See **doc/install.rst**

Documentation
-------------

To build the documentation, go to the **doc** folder and run::

  ./Allwmake

The built documentation is located at doc/DAFoamDoc.html

Citation
--------

License
-------
