.. _HeatTransfer_UBend:

U-bend internal cooling channel
-------------------------------

**NOTE**: Before running this case, please read the instructions in :ref:`Aerodynamics_NACA0012_Incomp` to get an overall idea of the DAFoam optimization setup.

This is a heat transfer optimization case for a U-bend internal cooling channel. The summary of the case is as follows:

    | Case: Heat transfer optimization for U bend cooling channels
    | Geometry: von Karman U bend duct
    | Objective function: Nusselt number
    | Design variables: 114 FFD points moving in the x, y, and z directions
    | Constraints: Symmetry constraint (total number: 38)
    | Mach number: 0.02
    | Reynolds number: 4.2e4
    | Mesh cells: 4.8K
    | Adjoint solver: simpleTDAFoam

The configuration files are available at `Github <https://github.com/mdolab/dafoam/tree/master/tutorials/HeatTransfer/UBend>`_. To run this case, first source the DAFoam environment (see :ref:`Tutorials`). Then you can go into the **run** folder and run::

  ./Allrun.sh 1

The optimization progress will then be written in the **log.opt** file.

For this case, the optimization converges in 6 steps, see the following figure. 
The baseline design has Nu=25.14 and the optimized design has Nu=26.89.

.. image:: images/UBend_Heat_Opt.jpg

We use ICEM to generate the FFD points.

.. image:: images/UBend_Heat_Mesh.jpg

We use simpleTDAFoam, which is based on simpleDAFoam with an extra scalar equation for temperature.