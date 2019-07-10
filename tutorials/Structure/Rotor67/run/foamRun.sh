#!/bin/bash

########## user input ##########
nProcs=$1
exec=mpirun
########## user input ##########

pFlag='-parallel'
if [ $1 -eq 1 ]; then
  pFlag=' '
fi

rm -f runCheckMesh
rm -f runFlowSolver
rm -f runAdjointSolver
rm -f runColoring

for n in `seq 0 1 1000000`; do

  if [ -e "runCheckMesh" ]
  then
    ${exec} -np $nProcs checkMesh $pFlag > checkMeshLog
    rm runCheckMesh
    touch jobFinished
  fi

  if [ -e "runColoring" ]
  then
    ${exec} -np $nProcs coloringSolverIncompressible $pFlag > coloringLog
    rm runColoring
    touch jobFinished
  fi

  if [ -e "runPotentialFoam" ]
  then
    ${exec} -np $nProcs potentialFoam -parallel > potentialFoamLog
    rm runPotentialFoam
    touch jobFinished
  fi

  if [ -e "runFlowSolver" ]
  then
    ${exec} -np $nProcs solidDisplacementDAFoam $pFlag > flowLog
    rm runFlowSolver
    touch jobFinished
  fi

  if [ -e "runAdjointSolver" ]
  then
    rm -rf optShapes*
    ${exec} -np $nProcs surfaceMeshTriangulate optShapes.stl -patches '(wing wingte)' $pFlag > optShapesLog
    ${exec} -np $nProcs solidDisplacementDAFoam $pFlag > adjointLog
    rm runAdjointSolver
    touch jobFinished
  fi
  
  sleep 5
  
done


