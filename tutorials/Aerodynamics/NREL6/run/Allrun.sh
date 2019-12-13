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

# get the geometry
echo "Downloading the geometry.."
cd constant
if [ ! -d NREL6_triSurface.tar.gz ]; then
  wget --no-check-certificate https://github.com/mdolab/dafoam_files/raw/master/tutorials/NREL6_triSurface.tar.gz &> ../log.download
  tar -xvf NREL6_triSurface.tar.gz >> ../log.download
fi
cd ../

# generate mesh
echo "Genearing the mesh.."
blockMesh > log.preProcessing
snappyHexMesh -overwrite >> log.preProcessing
renumberMesh -overwrite >> log.preProcessing
topoSet >> log.preProcessing

# copy initial and boundary condition files
cp -r 0.orig 0

# set the rotating velocity for MRF
setRotVelBCs -rotRad '(7.5 0 0)' -patchNames '(blade)' > log.preProcessing

# these are the actually commands to run the case
./foamRun.sh $1 &
sleep 1
echo "Running the optimization. Check the log.opt file for the progress."
mpirun -np $1 python runScript.py &> log.opt 

