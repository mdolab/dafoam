DAFoam: Discrete Adjoint with OpenFOAM
======================================

[![tests](https://github.com/mdolab/dafoam/actions/workflows/reg_tests.yml/badge.svg)](https://github.com/mdolab/dafoam/actions/workflows/reg_tests.yml) [![codecov](https://codecov.io/gh/mdolab/dafoam/branch/master/graph/badge.svg?token=8F8E7FAFGA)](https://codecov.io/gh/mdolab/dafoam)

DAFoam implements an efficient discrete adjoint method to perform high-fidelity gradient-based design optimization with the [MACH-Aero](https://github.com/mdolab/MACH-Aero) framework. DAFoam has the following features:

- It uses a popular open-source package [OpenFOAM](https://www.openfoam.com) for multiphysics analysis.
- It implements an efficient discrete adjoint approach with competitive speed, scalability, accuracy, and compatibility.
- It allows rapid discrete adjoint development for any steady and unsteady OpenFOAM primal solvers with modifying only a few hundred lines of source codes.
- It supports design optimizations for a wide range of disciplines such as aerodynamics, heat transfer, solid mechanics, hydrodynamics, and radiation.

![](cover.png)

Documentation
-------------

Refer to https://dafoam.github.io for installation, documentation, and tutorials.

Citation
--------

Please cite the following papers in any publication for which you find DAFoam useful. 

- Ping He, Charles A. Mader, Joaquim R.R.A. Martins, Kevin J. Maki. DAFoam: An open-source adjoint framework for multidisciplinary design optimization with OpenFOAM. AIAA Journal, 58:1304-1319, 2020. https://doi.org/10.2514/1.J058853

- Ping He, Charles A. Mader, Joaquim R.R.A. Martins, Kevin J. Maki. An aerodynamic design optimization framework using a discrete adjoint approach with OpenFOAM. Computer & Fluids, 168:285-303, 2018. https://doi.org/10.1016/j.compfluid.2018.04.012

License
-------

Copyright 2019 MDO Lab

Distributed using the GNU General Public License (GPL), version 3; see the LICENSE file for details.
