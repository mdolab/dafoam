.. _Tutorials:

Tutorials
---------

There are multiple optimization cases in the **tutorials** folder. 

.. toctree::
   :maxdepth: 3

   Tutorial_Aerodynamics
   Tutorial_HeatTransfer
   Tutorial_Structure
   Tutorial_Hydrodynamics
   Tutorial_Aerothermal
   Tutorial_Aerostructural

In each optimization case, the **run** folder contains all the optimization setup. 
The **optOutput** folder stores all the optimization results and logs. 
We recommend you first read the instructions in :ref:`Aerodynamics_NACA0012_Incomp` before running other cases.
All these tutorials use very coarse meshes, you need to refine the mesh for more realistic runs.

The optimization configurations are defined in **runScript.py**. There are seven sections:

- **Imports**. Import all external modules. No need to change.
- **Input Parameters**. Define the flow, adjoint, and optimization parameters. The explanation of these input parameters is in `Python layer <_static/Python/index.html>`_. Refer to classes-python-pyDAFoam-PYDAFOAM-aCompleteInputParameterSet()
- **DVGeo**. Import FFD files in plot3d format and define design variables.
- **DAFoam**. Adjoint misc setup. No need to change.
- **DVCon**. Define geometric constraints such as volume, thickness, and curvature constraints.
- **optFuncs**. Link optimization functions. No need to change.
- **Task**. Define optimization tasks (objective function, physical constraints, etc).

|

Before running the tutorials, you need to load the DAFoam environment.

- If you use the pre-compiled package, run this command to start a container::

   docker run -it --rm -u dafoamuser -v $HOME:/UserHome -w /UserHome dafoam/opt-packages:latest bash -rcfile /opt/setupDAFoam.sh

 This will mount your local computer's home directory to the container's /UserHome directory and login there. Then, copy the tutorials from /opt/repos/dafoam to /UserHome::

   cp -r /opt/repos/dafoam/tutorials .

 Finally, you can go into the **run** folder of a tutorial and run the optimization. For example, for the aerodynamic optimization of NACA0012 airfoil, run::

   cd tutorials/Aerodynamics/NACA0012_Airfoil_Incompressible/run && ./Allrun.sh 1

 The last parameter **1** means running the optimization using 1 CPU core. After this, check the log.opt for the optimization progress. All the intermediate shapes and logs (flow, adjoint, mesh quality, design variables, etc.) are stored in the optOutput directory. Once the optimization is finished, you can run ``exit`` to quit the container and use `Paraview <https://www.paraview.org/>`_ to post-process the optimization results on your local computer. Remember to choose **Case Type-Decomposed Case** to view the decomposed (parallel) cases in Paraview. 
 
 A few notes:
   
   - Treat the Docker container as disposable, i.e., start one container for one optimization run. If the optimization is running and you want to kill it, just run ``exit`` to quit the container.

   - Do not store simulation results in the container because they will be deleted after you exit. Run simulations on the mounted space /UserHome instead.
   
   - dafoamuser has the sudo privilege and its password is: dafoamuser.

   - Always run ``./Allclean.sh`` before running ``./Allrun.sh``. 


- If you have compiled DAFoam from the source code following :ref:`Installation`, load the OpenFOAM environment::

   . $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc

 Then, copy the tutorials to your local folder::

   cp -r $HOME/repos/dafoam/tutorials .

 Finally, you can go into the **run** folder of a tutorial and run::

   ./Allrun.sh 1

 A few notes:

   - Before running the optimization, source the OpenFOAM environment: ". $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc"
   
   - Because the OpenFOAM and Python layers interact through IO, job cleaning needs special attention. We assume you compile DAFoam from source and run it on an HPC system. In this case, the running executives will be automatically cleaned when you kill the job. However, if you compile DAFoam and run it on your local computer (not recommended, use the pre-compiled docker version instead!), you need to manually kill the job and clean the running stuff (e.g., the foamRun.sh script and other running executives).

   - Always run Allclean.sh before running Allrun.sh. 

