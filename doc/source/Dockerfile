FROM ubuntu:18.04

# Use ROOT
USER root

# Prerequisites
RUN apt-get update && apt-get install -y apt-utils --no-install-recommends

RUN apt-get install -y sudo build-essential flex bison cmake zlib1g-dev libboost-system-dev libboost-thread-dev libreadline-dev libncurses-dev libxt-dev qt5-default libqt5x11extras5-dev libqt5help5 qtdeclarative5-dev qttools5-dev libqtwebkit-dev freeglut3-dev libqt5opengl5-dev texinfo  libscotch-dev libcgal-dev gfortran swig wget git vim cmake-curses-gui libfl-dev --no-install-recommends

# Python
RUN apt-get install -y python3.6 python3-pip python3-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 10 && \ 
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

# Add a new user
RUN useradd -ms /bin/bash -G sudo -p dafoamuser dafoamuser

RUN echo "dafoamuser:dafoamuser"|chpasswd

# Swith to dafoamuser
USER dafoamuser

ENV HOME=/home/dafoamuser

# Numpy and Scipy
RUN pip install numpy==1.16.4 --user --no-cache

RUN pip install scipy==1.2.1 --user --no-cache

ENV PATH=$PATH:$HOME/.local/bin

# OpenMPI
ENV MPI_INSTALL_DIR=$HOME/packages/openmpi-1.10.7/opt-gfortran
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_INSTALL_DIR/lib
ENV PATH=$MPI_INSTALL_DIR/bin:$PATH

RUN mkdir -p $HOME/packages

RUN cd $HOME/packages && \
    wget https://download.open-mpi.org/release/open-mpi/v1.10/openmpi-1.10.7.tar.gz && \
    tar -xvf openmpi-1.10.7.tar.gz && cd openmpi-1.10.7 && \
    ./configure --prefix=$MPI_INSTALL_DIR && \
    make all install && \
    rm -rf ompi && \
    rm -rf $HOME/packages/openmpi-1.10.7.tar.gz

ENV LD_PRELOAD=$MPI_INSTALL_DIR/lib/libmpi.so

RUN pip install mpi4py==3.0.2 --user --no-cache

# Petsc
ENV PETSC_DIR=$HOME/packages/petsc-3.11.0
ENV PETSC_ARCH=real-opt
ENV PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
ENV PATH=$PETSC_DIR/$PETSC_ARCH/include:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib
ENV PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib

RUN cd $HOME/packages && \
    wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.11.0.tar.gz --no-check-certificate && \
    tar -xvaf petsc-3.11.0.tar.gz && cd petsc-3.11.0 && \
    sed -i 's/ierr = MPI_Finalize();CHKERRQ(ierr);/\/\/ierr = MPI_Finalize();CHKERRQ(ierr);/g' src/sys/objects/pinit.c && \
    ./configure --PETSC_ARCH=real-opt --with-scalar-type=real --with-debugging=0 --with-mpi-dir=$MPI_INSTALL_DIR --download-metis=yes --download-parmetis=yes --download-superlu_dist=yes --download-fblaslapack=yes --with-shared-libraries=yes --with-fortran-bindings=1 --with-cxx-dialect=C++11 && \
    make PETSC_DIR=$HOME/packages/petsc-3.11.0 PETSC_ARCH=real-opt all && \
    rm -rf src docs && \
    rm -rf $HOME/packages/petsc-3.11.0.tar.gz

RUN pip install petsc4py==3.11.0 --user --no-cache

# CGNS
ENV CGNS_HOME=$HOME/packages/CGNS-3.3.0/opt-gfortran
ENV PATH=$PATH:$CGNS_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib

RUN cd $HOME/packages && \
    wget https://github.com/CGNS/CGNS/archive/v3.3.0.tar.gz && \
    tar -xvaf v3.3.0.tar.gz && cd CGNS-3.3.0 && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCGNS_ENABLE_FORTRAN=1 -DCMAKE_INSTALL_PREFIX=$CGNS_HOME -DCGNS_BUILD_CGNSTOOLS=0 && \
    make all install && \
    rm -rf $HOME/packages/v3.3.0.tar.gz

# MACH framework
ENV PYTHONPATH=$PYTHONPATH:$HOME/repos

RUN mkdir -p $HOME/repos

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/baseclasses

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/pygeo

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/multipoint

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/pyspline && \
    cd pyspline && \
    cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
    make

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/pyhyp && \
    cd pyhyp && \
    cp -r config/defaults/config.LINUX_GFORTRAN_OPENMPI.mk config/config.mk && \
    make

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/cgnsutilities && \
    cd cgnsutilities && cp config.mk.info config.mk && \
    make

ENV PATH=$PATH:$HOME/repos/cgnsutilities/bin

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/idwarp && \
    cd idwarp && \
    cp -r config/defaults/config.LINUX_GFORTRAN_OPENMPI.mk config/config.mk && \
    make

RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/pyoptsparse && \
    cd pyoptsparse && pip install -r requirements.txt && rm -rf build && python setup.py install --user

# OpenFOAM
RUN mkdir -p $HOME/OpenFOAM 

RUN cd $HOME/OpenFOAM && \
    wget https://sourceforge.net/projects/openfoamplus/files/v1812/OpenFOAM-v1812.tgz/download --no-check-certificate -O OpenFOAM-v1812.tgz && \
    wget https://sourceforge.net/projects/openfoamplus/files/v1812/ThirdParty-v1812.tgz/download --no-check-certificate -O ThirdParty-v1812.tgz && \
    tar -xvf OpenFOAM-v1812.tgz && \
    tar -xvf ThirdParty-v1812.tgz && \
    cd $HOME/OpenFOAM/OpenFOAM-v1812 && \
    wget https://github.com/mdolab/dafoam/releases/download/v1.1.0/UPstream.C --no-check-certificate && \
    mv UPstream.C src/Pstream/mpi/UPstream.C && \
    . etc/bashrc && \
    export WM_NCOMPPROCS=4 && \
    ./Allwmake && \
    rm -rf $HOME/OpenFOAM/OpenFOAM-v1812.tgz $HOME/OpenFOAM/ThirdParty-v1812.tgz && \
    rm -rf $HOME/OpenFOAM/OpenFOAM-v1812/build && \
    rm -rf $HOME/OpenFOAM/ThirdParty-v1812/boost_1_64_0 $HOME/OpenFOAM/ThirdParty-v1812/openmpi-1.10.4 $HOME/OpenFOAM/ThirdParty-v1812/ParaView-v5.6.0

# pyOFM
RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/pyofm && \
    cd pyofm && \
    . $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc && \
    make

# DAFoam
RUN cd $HOME/repos && \
    git clone https://github.com/mdolab/dafoam && \
    . $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc && \
    cd $HOME/repos/dafoam && \
    make && \
    make clean

# Create a mount folder
RUN mkdir -p $HOME/mount

# Add environmental variables to bashrc
RUN echo '# f2py' >> $HOME/.bashrc
RUN echo 'PATH=$PATH:$HOME/.local/bin' >> $HOME/.bashrc
RUN echo '# OpenMPI-1.10.7' >> $HOME/.bashrc
RUN echo 'export MPI_INSTALL_DIR=$HOME/packages/openmpi-1.10.7/opt-gfortran' >> $HOME/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_INSTALL_DIR/lib' >> $HOME/.bashrc
RUN echo 'export PATH=$MPI_INSTALL_DIR/bin:$PATH' >> $HOME/.bashrc
RUN echo 'export LD_PRELOAD=$MPI_INSTALL_DIR/lib/libmpi.so' >> $HOME/.bashrc
RUN echo '# Petsc-3.11.0' >> $HOME/.bashrc
RUN echo 'export PETSC_DIR=$HOME/packages/petsc-3.11.0' >> $HOME/.bashrc
RUN echo 'export PETSC_ARCH=real-opt' >> $HOME/.bashrc
RUN echo 'export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH' >> $HOME/.bashrc
RUN echo 'export PATH=$PETSC_DIR/$PETSC_ARCH/include:$PATH' >> $HOME/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib' >> $HOME/.bashrc
RUN echo 'export PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib' >> $HOME/.bashrc
RUN echo '# CGNS-3.3.0' >> $HOME/.bashrc
RUN echo 'export CGNS_HOME=$HOME/packages/CGNS-3.3.0/opt-gfortran' >> $HOME/.bashrc
RUN echo 'export PATH=$PATH:$CGNS_HOME/bin' >> $HOME/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib' >> $HOME/.bashrc
RUN echo '# Python Path' >> $HOME/.bashrc
RUN echo 'export PYTHONPATH=$PYTHONPATH:$HOME/repos' >> $HOME/.bashrc
RUN echo '# cgnsUtilities' >> $HOME/.bashrc
RUN echo 'export PATH=$PATH:$HOME/repos/cgnsutilities/bin' >> $HOME/.bashrc
RUN echo '# OpenFOAM' >> $HOME/.bashrc
RUN echo '. $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc' >> $HOME/.bashrc
