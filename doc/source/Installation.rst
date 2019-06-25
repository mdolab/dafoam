.. _Installation:

Installation 
------------

This section assumes you want to compile the DAFoam optimization package from the source. If you use the pre-compiled version, skip this section.

**DAFoam** runs on Linux systems and is based on **OpenFOAM-v1812**. You must install **OpenFOAM** and verify that it is working correctly. You also need to install the 3rd party and **MDOLab** packages before using **DAFoam** for optimization. Other dependencies include: 

- C/C++ compilers (gcc/g++ or icc/icpc)
  
- Fortran compiler (gfortran or ifort)
  
- MPI software (openmpi, mvapich2, or impi)
  
- Swig
  
- cmake

To compile the documentation, you also need:

- Doxygen 

- Graphviz

- Sphinx 

**NOTE:** Always try to use the system provided C/Fortran compiler and MPI software to compile DAFoam and all its dependencies. 
Using self-built C/Fortran and MPI may cause linking issues

|

To install the **DAFoam** package:

1. Compile **OpenFOAM-v1812** following this website: http://openfoamwiki.net/index.php/Installation/Linux **NOTE**: After the OpenFOAM installation is done, start a new session to install the rest packages; **DO NOT** load the OpenFOAM environment. This is to prevent environmental variable conflict between OpenFOAM and other packages.

2. Install **Anaconda** package for Python (https://repo.continuum.io/archive/Anaconda2-2.4.0-Linux-x86_64.sh). **NOTE**: Anaconda2-2.4.0 is recommended since the newer version may have libgfortran conflict.

3. Create a **packages** folder in your home dir, and install the following 3rd party packages

- **PETSc v3.6.4** (http://www.mcs.anl.gov/petsc/). Untar the package, go into petsc-3.6.4, and run::

   sed -i 's/ierr = MPI_Finalize();CHKERRQ(ierr);/\/\/ierr = MPI_Finalize();CHKERRQ(ierr);/g' src/sys/objects/pinit.c

  This will comment out line 1367 in src/sys/objects/pinit.c to prevent Petsc from conflicting with OpenFOAM MPI. After this, run::

   ./configure --with-shared-libraries --download-superlu_dist --download-parmetis --download-metis --with-fortran-interfaces --with-debugging=no --with-scalar-type=real --PETSC_ARCH=real-opt --download-fblaslapack
   
  and::

    make PETSC_DIR=$HOME/packages/petsc-3.6.4 PETSC_ARCH=real-opt all

  Add the following into your bashrc and source it::

    export PETSC_DIR=$HOME/packages/petsc-3.6.4
    export PETSC_ARCH=real-opt
    export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
    export PATH=$PETSC_DIR/$PETSC_ARCH/include:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib

- **cgnslib_3.2.1** (http://cgns.sourceforge.net/download.html). NOTE: The 3.2.1 version fortran include file is bad. After untaring, manually edit the cgnslib_f.h file in the src directory and remove all the comment lines at the beginning of the file starting with c. Then run::

    cmake .

  and::

    ccmake .

  A “GUI” appears and toggle ENABLE_FORTRAN by pressing [enter] (should be OFF when entering the screen for the first time, hence set it to ON). Type ‘c’ to reconfigure and ‘g’ to generate and exit. Then run::

    make

  Now add this into your bashrc and source it::

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/packages/cgnslib_3.2.1/src

- **mpi4py-1.3.1** (http://bitbucket.org/mpi4py/mpi4py/downloads/). Untar and run::
 
    python setup.py install --user
    
  This will install the package to the .local directory.
  
- **petsc4py-3.6.0** (http://bitbucket.org/petsc/petsc4py/downloads/). Untar and run::
 
    python setup.py install --user
    
  This will install the package to the .local directory.
  
3. Create a **repos** folder in your home dir, and install the following **MDOLab** packages. 

- First add this to your bashrc and source it::
 
     export PYTHONPATH=$PYTHONPATH:$HOME/repos/
   
- Get **baseClasses** (https://github.com/mdolab/baseclasses). No need to compile. 

- Get **pyGeo** (https://github.com/mdolab/pygeo). No need to compile.
 
- Get **openFoamMeshReader** (https://github.com/mdolab/openfoammeshreader). No need to compile.   

- Get **multipoint** (https://github.com/mdolab/multipoint). No need to compile.   

- Get **pySpline** (https://github.com/mdolab/pyspline). Run::
   
     cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
   
  and::
 
     make
    
- Get **pyHyp** (https://github.com/mdolab/pyHyp). Run::
   
     cp -r config/defaults/config.LINUX_GFORTRAN_OPENMPI.mk config/config.mk
   
  and::
 
     make
     
- Get **IDWarp** (https://github.com/mdolab/idwarp). Run::
     
     cp -r config/defaults/config.LINUX_GFORTRAN_OPENMPI.mk config/config.mk
     
  and::
   
     make
     
- Get **pyOptSparse** (https://github.com/mdolab/pyoptsparse). Run::
 
     python setup.py install --user
     
4. Download **DAFoam** (https://github.com/mdolab/dafoam) to $HOME/repos. First source the **OpenFOAM** environmental variables::

    source $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc
    
   Then run::
  
    ./Allwmake
    
   Next, go to dafoam/python/reg_tests and untar “input.tar.gz”. Finally, run the regression test there::
  
    python run_reg_tests.py
    
   Make sure the regression test passes. OK, the installation of **DAFoam** is finished.
