#!/bin/bash

# pre-processing

# generate mesh
python genAirFoilMesh.py
plot3dToFoam -noBlank volumeMesh.xyz
autoPatch 30 -overwrite
createPatch -overwrite
renumberMesh -overwrite

# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh &
mpirun -np 2 python runScript.py
