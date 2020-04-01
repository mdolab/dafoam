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

# download the FFD folder
echo "Downloading the FFD.."
if [ ! -d FFD ]; then
  wget --no-check-certificate -O UBend_FFD.tar.gz https://umich.box.com/shared/static/rwt93cs8ea7kbvxqr1coy5t832v09x89.gz &> log.download
  tar -xvf UBend_FFD.tar.gz >> log.download
fi

# generate mesh
echo "Generating mesh.."
blockMesh > log.meshGeneration
renumberMesh -overwrite >> log.meshGeneration
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0

# these are the actually commands to run the case
./foamRun.sh $1 &
sleep 1
echo "Running the optimization. Check the log.opt file for the progress."
mpirun -np $1 python runScript.py &> log.opt 

