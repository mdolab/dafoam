.. _Aerodynamics_NACA0012_Comp:

NACA0012 airfoil compressible
-----------------------------

**NOTE**: Before running this case, please read the instructions in :ref:`Aerodynamics_NACA0012_Incomp` to get an overall idea of the DAFoam optimization setup.

This is an aerodynamic shape optimization case for an airfoil at transonic conditions. The summary of the case is as follows:

    | Case: Airfoil aerodynamic optimization
    | Geometry: NACA0012 
    | Objective function: Drag coefficient
    | Design variables: 40 FFD points moving in the y direction, one angle of attack
    | Constraints: Symmetry, volume, thickness, and lift constraints (total number: 81)
    | Mach number: 0.7
    | Reynolds number: 2.3 million
    | Mesh cells: 8.6K
    | Adjoint solver: rhoSimpleCDAFoam

To run this case, first source the DAFoam environment (see :ref:`Tutorials`). Then you can go into the **run** folder and run::

  ./Allrun.sh 1

The optimization progress will then be written in the **log.opt** file. 

For this case, the optimization converges in 17 steps, see the following figure. 
The baseline design has C_D=0.01777, C_L=0.5000, and the optimized design has C_D=0.01205, C_L=0.5000.

.. image:: images/NACA0012_Comp_Opt.jpg

In this case, we need to use rhoSimpleCDAFoam, a compressible solver that uses the SIMPLEC algorithm. 
The case setup is similar to :ref:`Aerodynamics_NACA0012_Incomp`.
The major difference is in the ``aeroOptions`` dictionary where we need to define different ``divschemes``, ``fvrelaxfactors``, and ``simplecontrol``. 
These parameters are critical to ensure robust flow simulations for transonic conditions.