#!/bin/bash

# pre-processing

# generate mesh
cd constant
if [ ! -d triSurface ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/JBC_triSurface.tar.gz
  tar -xvf JBC_triSurface.tar.gz
fi
cd ..
blockMesh
snappyHexMesh -overwrite
renumberMesh -overwrite

if [ ! -d FFD ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/JBC_FFD.tar.gz
  tar -xvf JBC_FFD.tar.gz
fi

# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh &
mpirun -np 2 python runScript.py
