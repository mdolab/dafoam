#!/bin/bash

# pre-processing

# generate mesh
cd constant
if [ ! -d triSurface ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/DPW4_triSurface.tar.gz
  tar -xvf DPW4_triSurface.tar.gz
fi
cd ..
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
renumberMesh -overwrite
createPatch -overwrite

# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh &
mpirun -np 2 python runScript.py
