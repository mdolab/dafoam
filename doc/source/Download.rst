.. _Download:

Download 
--------

.. note::
    NOTE: this website is for DAFoam v1.1 and is no longer updated. For DAFoam v2.0+, visit `dafoam.github.io <https://dafoam.github.io>`_

The current stable version of DAFoam is v1.1. See the changes log from `here. <https://github.com/mdolab/dafoam/releases/tag/v1.1.1>`_

There are two options to run DAFoam: **pre-compiled package** and **source code**. If you are running DAFoam for the first time, we recommend using the pre-compiled version, which supports Linux (Ubuntu, Fedora, CentOS, etc), MacOS, and Windows systems. For production runs on an HPC system, you need to compile DAFoam from the source.

- **Pre-compiled package**

 The pre-compiled package is available on Docker Hub. Before downloading the pre-compiled package, you need to install **Docker**. Follow the installation instructions for `Ubuntu <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`_, `Fedora <https://docs.docker.com/install/linux/docker-ce/fedora/>`_, `CentOS <https://docs.docker.com/install/linux/docker-ce/centos/>`_, `MacOS <https://docs.docker.com/docker-for-mac/install/>`_, and  `Windows <https://docs.docker.com/docker-for-windows/install/>`_. 
 
 For example, on Ubuntu, you can install the latest Docker by running this command in the terminal::

    sudo apt-get remove docker docker-engine docker.io containerd runc && sudo apt-get update && sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common -y && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && sudo apt-get update && sudo apt-get install docker-ce -y

 Then you need to add your user name to the docker group by::

    sudo usermod -aG docker $USER

 After this, you need to logout and re-login your account to make the usermod command effective. Once done, verify the docker installation by::

    docker --version

 You should be able to see your installed Docker version. Note that different operating systems have very different Docker installation process, refer to the above links for more details. 

 Once the Docker is installed and verified, run this command from the terminal::

    docker run -it --rm -u dafoamuser -v $HOME:/home/dafoamuser/mount -w /home/dafoamuser/mount dafoam/opt-packages:v1.1 bash

 It will first download the pre-compiled package (v1.1) from the Docker Hub if it has not been downloaded. Then it will start a Docker container (a light-weight virtual machine), mount your local computer's home directory to the container's ``mount`` directory, login to ``mount`` as dafoamuser, and set the relevant DAFoam environmental variables. Now you are ready to run DAFoam tutorials. Refer to :ref:`Tutorials` for more details.

- **Source code**

 The DAFoam source code is available at https://github.com/mdolab/dafoam. DAFoam depends on multiple prerequisites and packages. Refer to :ref:`Installation` for installation instructions.

