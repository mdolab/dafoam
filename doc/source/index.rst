Home
====

DAFoam: Discrete Adjoint with OpenFoam
--------------------------------------

DAFoam contains a suite of discrete adjoint solvers for OpenFOAM. These adjoint solvers run as standalone executives to compute derivatives. DAFoam also has a Python interface that allows the adjoint solvers to interact with external modules for high-fidelity design optimization. DAFoam has the following features:

- It implements an efficient discrete adjoint approach with competitive speed, scalability, accuracy, and compatibility.
- It allows rapid discrete adjoint development for any steady-state OpenFOAM solvers with modifying only O(100) lines of source codes.
- It supports design optimizations for a wide range of disciplines such as aerodynamics, heat transfer, structures, hydrodynamics, and radiation.

.. image:: images/DPW6_Transparent.png

The DAFoam repository comprises of five main directories:

- applications: adjoint solvers and utilities
- doc: documentation
- python: python interface to other optimization packages
- src: the core DAFoam libraries
- tutorials: sample optimization setup for each adjoint solver

Contents:

.. toctree::
   :maxdepth: 2

   self
   Download
   Tutorials
   Development
   Publications


