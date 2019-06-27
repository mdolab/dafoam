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

nProcs=$1

copyFC ()
{
  cp -r 0 ../FlowConfig$1/
  cp -r constant system ../FlowConfig$1/
  cd ../FlowConfig$1/
  sed -i "/numberOfSubdomains/c\numberOfSubdomains $nProcs;" system/decomposeParDict
  if [ $nProcs -ne 1 ]; then
    decomposePar
  fi
  cd ../MultiPointMain
}

# pre-processing
echo "Downloading the surface mesh.."
if [ ! -f surfaceMesh.cgns ]; then
  wget --no-check-certificate https://github.com/mdolab/dafoam_files/raw/master/tutorials/Odyssey_surfaceMesh.cgns.tar.gz &> log.download
  tar -xvf Odyssey_surfaceMesh.cgns.tar.gz >> log.download
fi
echo "Generating mesh.."
python runVolMesh.py > log.meshGeneration
plot3dToFoam -noBlank volMesh.xyz >> log.meshGeneration
autoPatch 60 -overwrite >> log.meshGeneration
createPatch -overwrite >> log.meshGeneration
renumberMesh -overwrite >> log.meshGeneration
echo "Generating mesh.. Done!"
cp -r 0.orig 0

echo "Copyinig the files to ../FlowConfig*"
# for FC0
copyFC 0
# for FC1
copyFC 1


# these are the actually commands to run the case
./foamRunMultiPoint.sh $nProcs &
mpirun -np $nProcs python runScript.py --nProcs=$nProcs &> log.opt &
echo "Running the optimization. Check the log.opt file for the progress."
