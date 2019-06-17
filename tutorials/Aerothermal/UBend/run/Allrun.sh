#!/bin/bash

# pre-processing

# download the FFD folder
if [ ! -d FFD ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/UBend_FFD.tar.gz
  tar -xvf UBend_FFD.tar.gz
fi

# generate mesh
blockMesh
renumberMesh -overwrite
tar -xvf FFD.tar.gz

# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh &
mpirun -np 2 python runScript.py
