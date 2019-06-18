.. _Tutorials:

Tutorials
---------

There are multiple optimization cases in the tutorials folder. 

.. toctree::
   :maxdepth: 2

   Tutorial_Aerodynamics
   Tutorial_HeatTransfer
   Tutorial_Structure
   Tutorial_Hydrodynamics
   Tutorial_Aerothermal
   Tutorial_Aerostructural

In each optimization case, the **run** folder contains all the optimization setup. The **optOutput** folder is to store all the optimization results and logs. 

The optimization configurations are defined in **runScript.py**. There are seven sections:

- **Imports**. Import all external modules. No need to change.
- **Input Parameters**. Define the flow, adjoint, and optimization parameters. The explanation of these input parameters is in `Python layer <_static/Python/index.html>`_.
- **DVGeo**. Import FFD files in plot3d format and define design variables.
- **DAFoam**. Adjoint misc setup. No need to change.
- **DVCon**. Define geometric constraints such as volume, thickness, and curvature constraints.
- **optFuncs**. Link optimization functions. No need to change.
- **Task**. Define optimization tasks (objective function, physical constraints, etc).

|

Before running the tutorials, you need to load the DAFoam environment.

- If you use the self-compiled package, first initialize the DAFoam package::

    ./getDAFoam.sh

 This will download the DAFoam docker image (if not exist) and start a docker container. You can check docker images on your computer by::

     docker images
 
 You can also check docker containers running on your computer::

     docker ps -a

 Once the DAFoam docker container is up, you can load the DAFoam environment by running::

    ./startDAFoam.sh

 This will create a new bash session. Then in the bash session, copy the tutorials from /opt to your local folder::

    cp -r /opt/repos/dafoam/tutorials .

 Finally, you can go into the **run** folder of a tutorial and run::

   ./Allrun.sh

 Once finished, you can release the resources occupied by the container by running::

    docker stop dafoam && docker rm dafoam

- If you have compiled DAFoam from the source code following :ref:`Installation`, load the OpenFOAM environment::

   . $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc

 Then, copy the tutorials to your local folder::

    cp -r $HOME/repos/dafoam/tutorials .

 Finally, you can go into the **run** folder of a tutorial and run::

   ./Allrun.sh

