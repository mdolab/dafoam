.. _Download:

Download 
--------

There are two options to run DAFoam: **pre-compiled package** and **source code**. If you are running DAFoam for the first time, we recommend using the pre-compiled version, which supports Linux (Ubuntu, Fedora, CentOS, etc), MacOS, and Windows systems. For production runs on an HPC system, you need to compile DAFoam from the source.

- **Pre-compiled package**

 The pre-compiled package is available on Docker Hub. Before downloading the pre-compiled package, you need to install **Docker**. Follow the installation instructions for `Ubuntu <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`_, `Fedora <https://docs.docker.com/install/linux/docker-ce/fedora/>`_, `CentOS <https://docs.docker.com/install/linux/docker-ce/centos/>`_, `MacOS <https://docs.docker.com/docker-for-mac/install/>`_, and  `Windows <https://docs.docker.com/docker-for-windows/install/>`_. Once finished, open a terminal and verify the installation by::

    docker --version

 You should be able to see your installed Docker version. 

 Next, run this command from the terminal::

    docker run -it --rm -u dafoamuser -v $HOME:/UserHome -w /UserHome dafoam/opt-packages:latest bash -rcfile /opt/setupDAFoam.sh

 It will first download the pre-compiled package from the Docker Hub if it has not been downloaded. Then it will start a Docker container (a light-weight virtual machine), mount your local computer's home directory to the container's /UserHome directory, login to /UserHome as dafoamuser, and set the relevant DAFoam environmental variables. Now you are ready to run DAFoam tutorials. Refer to :ref:`Tutorials` for more details.

- **Source code**

 The DAFoam source code is available at https://github.com/mdolab/dafoam. DAFoam depends on multiple prerequisites and packages. Refer to :ref:`Installation` for installation instructions.

