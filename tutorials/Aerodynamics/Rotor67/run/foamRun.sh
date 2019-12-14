#!/bin/bash

########## user input ##########
nProcs=$1
exec=mpirun
########## user input ##########

pFlag='-parallel'
if [ $1 -eq 1 ]; then
  pFlag=' '
fi

rm -rf runCheckMesh
rm -rf runFlowSolver
rm -rf runAdjointSolver
rm -rf runColoring
rm -rf jobFinished
rm -rf runSolidSolver
rm -rf runSolidAdjointSolver

solidAdjRunCounter=0
solidRunCounter=0

for n in `seq 0 1 1000000`; do

  if [ -e "runCheckMesh" ]
  then
    ${exec} -np $nProcs checkMesh $pFlag > checkMeshLog
    rm runCheckMesh
    touch jobFinished
  fi

  if [ -e "runColoring" ]
  then
    ${exec} -np $nProcs coloringSolverCompressible $pFlag > coloringLog
    rm runColoring
    touch jobFinished
  fi

  if [ -e "runFlowSolver" ]
  then
    ${exec} -np $nProcs turboDAFoam $pFlag > flowLog
    rm runFlowSolver
    touch jobFinished
  fi

  if [ -e "runAdjointSolver" ]
  then
    ${exec} -np $nProcs surfaceMeshTriangulate optShapes.stl -patches '(bladeps bladefillet bladess hub)' $pFlag > optShapesLog
    ${exec} -np $nProcs turboDAFoam $pFlag > adjointLog
    rm runAdjointSolver
    touch jobFinished
  fi
  
  sleep 5
  
done



