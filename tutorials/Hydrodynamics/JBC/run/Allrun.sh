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
  wget --no-check-certificate -O JBC_triSurface.tar.gz https://umich.box.com/shared/static/q007vhg3bppwadz4dv3yfxbn660coz5l.gz &> ../log.download
  tar -xvf JBC_triSurface.tar.gz >> ../log.download
fi
cd ..

if [ ! -d FFD ]; then
  wget --no-check-certificate -O JBC_FFD.tar.gz https://umich.box.com/shared/static/7wfpijw4901yzvu7j70c3c4zr3cs2h4n.gz &>> log.download
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
echo "Running the optimization. Check the log.opt file for the progress."
mpirun -np $1 python runScript.py &> log.opt 

