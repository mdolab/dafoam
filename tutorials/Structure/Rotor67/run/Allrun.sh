#!/bin/bash

# pre-processing

# generate mesh
cd constant
if [ ! -d polyMesh ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/Rotor67_Structure_polyMesh.tar.gz
  tar -xvf Rotor67_Structure_polyMesh.tar.gz
fi
cd ..
renumberMesh -overwrite

# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh &
mpirun -np 2 python runScript.py
