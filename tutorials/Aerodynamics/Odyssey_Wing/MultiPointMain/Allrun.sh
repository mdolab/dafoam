#!/bin/bash

nProcs=2

copyFC ()
{
  cp -r 0 ../FlowConfig$1/
  cp -r constant system ../FlowConfig$1/
  cd ../FlowConfig$1/
  sed -i "/numberOfSubdomains/c\numberOfSubdomains $nProcs;" system/decomposeParDict
  decomposePar
  cd ../MultiPointMain
}

# pre-processing
# download the surfaceMesh
if [ ! -f surfaceMesh.cgns ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/Odyssey_surfaceMesh.cgns.tar.gz
  tar -xvf Odyssey_surfaceMesh.cgns.tar.gz
fi
python runVolMesh.py
plot3dToFoam -noBlank volMesh.xyz
autoPatch 60 -overwrite
createPatch -overwrite
renumberMesh -overwrite
cp -r 0.orig 0

# for FC0
copyFC 0
# for FC1
copyFC 1


# these are the actually commands to run the case
./foamRunMultiPoint.sh &
mpirun -np $nProcs python runScript.py
