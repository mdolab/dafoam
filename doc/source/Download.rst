.. _Download:

Download 
--------

There are two options to run DAFoam: **pre-compiled package** and **source code**. If you are running DAFoam for the first time, we recommend using the pre-compiled version. For production runs on a HPC system, you need to compile DAFoam from the source.

**Pre-compiled package**

The pre-compiled package is available on Docker Hub. It supports Linux (Ubuntu, Centos, etc) and MacOS systems.

To download the pre-compiled package, you need to first install **docker**, refer to https://www.docker.com/ for more details. On Ubuntu, you can get docker by::

    sudo apt-get install docker.io

Once you have installed docker, download the following script:

`getDAFoam.sh <https://github.com/mdolab/dafoam_files/raw/master/scripts/getDAFoam.sh>`_

and run::

    chmod +x getDAFoam.sh && ./getDAFoam.sh

This will download the docker image for DAFoam optimization package. Then, download this script:

`startDAFoam.sh <https://github.com/mdolab/dafoam_files/raw/master/scripts/startDAFoam.sh>`_

and run::

    chmod +x startDAFoam.sh && ./startDAFoam.sh

This will start a new bash window where you can run DAFoam tutorials. Refer to :ref:`Tutorials` for more details.

**Source code**

DAFoam depends on multiple prerequisites and packages. Refer to :ref:`Installation` for detailed DAFoam installation instructions.

