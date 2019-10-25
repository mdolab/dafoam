DAFoam: Discrete Adjoint with OpenFOAM
======================================

[![Build Status](https://travis-ci.com/mdolab/dafoam.svg?token=PKPXZYJ4Ny7P59TKMPSX&branch=master)](https://travis-ci.com/mdolab/dafoam)
[![Documentation Status](https://readthedocs.org/projects/dafoam/badge/?version=latest)](https://dafoam.readthedocs.io/en/latest/?badge=latest)

DAFoam contains a suite of discrete adjoint solvers for OpenFOAM. These adjoint solvers run as standalone executives to compute derivatives. DAFoam also has a Python interface that allows the adjoint solvers to interact with external modules for high-fidelity design optimization using the [MACH framework](http://mdolab.engin.umich.edu/docs/machFramework/MACH-Aero.html). DAFoam has the following features:

- It implements an efficient discrete adjoint approach with competitive speed, scalability, accuracy, and compatibility.
- It allows rapid discrete adjoint development for any steady-state OpenFOAM solvers with modifying only a few hundred lines of source codes.
- It supports design optimizations for a wide range of disciplines such as aerodynamics, heat transfer, structures, hydrodynamics, and radiation.

![](doc/source/images/DPW6_Transparent.png)

Documentation
-------------

Refer to https://dafoam.rtfd.io for DAFoam installation and tutorials.

To build the documentation locally, go to the **doc** folder and run:

`./Allwmake`

The built documentation is located at **doc/DAFoamDoc.html**

Citation
--------

Refer to the following two papers for more technical background of DAFoam. If you use DAFoam in publications, please cite these papers.

Ping He, Charles A. Mader, Joaquim R.R.A. Martins, Kevin J. Maki. DAFoam: An open-source adjoint framework for multidisciplinary design optimization with OpenFOAM. AIAA Journal, 2019. https://doi.org/10.2514/1.J058853

```
@article{DAFoamAIAAJ19,
	Author = {Ping He and Charles A. Mader and Joaquim R. R. A. Martins and Kevin J. Maki},
	Doi = {10.2514/1.J058853},
	Journal = {AIAA Journal},
	Title = {{DAFoam}: An open-source adjoint framework for multidisciplinary design optimization with {OpenFOAM}},
	Year = {2019}}
```

Ping He, Charles A. Mader, Joaquim R.R.A. Martins, Kevin J. Maki. An aerodynamic design optimization framework using a discrete adjoint approach with OpenFOAM. Computer & Fluids, 168:285-303, 2018. https://doi.org/10.1016/j.compfluid.2018.04.012

```
@article{DAFoamCAF18,
	Author = {Ping He and Charles A. Mader and Joaquim R. R. A. Martins and Kevin J. Maki},
	Doi = {10.1016/j.compfluid.2018.04.012},
	Journal = {Computers \& Fluids},
	Pages = {285--303},
	Title = {An aerodynamic design optimization framework using a discrete adjoint approach with {OpenFOAM}},
	Volume = {168},
	Year = {2018}}
```

License
-------

Copyright 2019 MDO Lab

Distributed using the GNU General Public License (GPL), version 3; see the LICENSE file for details.
