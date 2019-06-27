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
cd constant
if [ ! -d triSurface ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/JBC_triSurface.tar.gz &> ../log.download
  tar -xvf JBC_triSurface.tar.gz >> ../log.download
fi
cd ..

if [ ! -d FFD ]; then
  wget https://github.com/mdolab/dafoam_files/raw/master/tutorials/JBC_FFD.tar.gz &>> log.download
  tar -xvf JBC_FFD.tar.gz >> log.download
fi

echo "Generating mesh.."
blockMesh > log.meshGeneration
snappyHexMesh -overwrite >> log.meshGeneration
renumberMesh -overwrite >> log.meshGeneration
echo "Generating mesh.. Done!"



# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh $1 &
sleep 1
mpirun -np $1 python runScript.py &> log.opt &
echo "Running the optimization. Check the log.opt file for the progress."
