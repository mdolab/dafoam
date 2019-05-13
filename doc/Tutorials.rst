.. _Tutorials:

Tutorials
---------

There are multiple optimization cases in the tutorials folder. In each optimization case, the **run** folder contains all the optimization setup. The **optOutput** folder is to store all the optimization results and logs. 

To run an optimization, first load the OpenFOAM environment::

   . $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc

Then go to the **run** folder, and run::

   ./Allrun.sh

The optimization configurations are defined in **runScript.py**. There are seven sections:

- **Imports**. Import all external modules. No need to change.
- **Input Parameters**. Define the flow, adjoint, and optimization parameters. The explanation of these input parameters is in the **Python Layer** tab in DAFoamDoc.html.
- **DVGeo**. Import FFD files in plot3d format and define design variables.
- **DAFoam**. Adjoint misc setup. No need to change.
- **DVCon**. Define geometric constraints such as volume, thickness, and curvature constraints.
- **optFuncs**. Link optimization functions. No need to change.
- **Task**. Define optimization tasks (objective function, physical constraints, etc).