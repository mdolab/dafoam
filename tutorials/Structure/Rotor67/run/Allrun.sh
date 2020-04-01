#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

if [ -z "$1" ]; then
  echo "No argument supplied!"
  echo "Example: ./Allrun.sh 1"
  echo "This will run the case using 1 CPU core."
  exit
fi

# pre-processing

# generate mesh
echo "Downloading the mesh.."
cd constant
if [ ! -d polyMesh ]; then
  wget --no-check-certificate -O Rotor67_Structure_polyMesh.tar.gz https://umich.box.com/shared/static/f9otwqcqyq0326av6o4vmmxqpqiqjpyv.gz &> ../log.download
  tar -xvf Rotor67_Structure_polyMesh.tar.gz >> ../log.download
fi
cd ..
renumberMesh -overwrite  >> log.download

# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh $1 &
sleep 1
echo "Running the optimization. Check the log.opt file for the progress."
mpirun -np $1 python runScript.py &> log.opt 

