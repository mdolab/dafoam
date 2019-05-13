DAFoam
======

DAFoam contains a suite of discrete adjoint solvers for OpenFOAM. These adjoint solvers run as standalone executives to compute derivatives. DAFoam also has a Python interface that allows the adjoint solvers to interact with external modules for high-fidelity design optimization. DAFoam has the following features:

- It implements an efficient discrete adjoint approach with competitive speed, scalability, accuracy, and compatibility.
- It allows rapid discrete adjoint development for any steady-state OpenFOAM solvers with modifying only O(100) lines of source codes.
- It supports design optimizations for a wide range of disciplines such as aerodynamics, heat transfer, structures, hydrodynamics, and radiation.

Documentation
-------------

Refer to `Installation <doc/Installation.rst>`_ and `Tutorials <doc/Tutorials.rst>`_.

To build the full documentation, go to the **doc** folder and run::

  ./Allwmake

The built documentation is located at **doc/DAFoamDoc.html**

License
-------
