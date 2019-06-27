#!/bin/bash

################## user input ####################
nProcs=$1
nCases=2
exec=mpirun
outputPath=../optOutput/
################## user input ####################

if [ $1 -eq 1 ]
then
  jobFlag=
else
  jobFlag=-parallel 
fi

rm -f runCheckMesh
rm -f runColoring
rm -f runFlowSolver*
rm -f runAdjointSolver*
rm -f jobFinished*

# main loop
for (( n=0; n <=1000000; ++n ))
do

  # checkMesh
  if [ -e "runCheckMesh" ]
  then
    ${exec} -np ${nProcs} checkMesh ${jobFlag} > checkMeshLog
    rm runCheckMesh
    touch jobFinished
  fi
  
  # coloringSolver
  if [ -e "runColoring" ]
  then
    ${exec} -np ${nProcs} coloringSolverIncompressible ${jobFlag} > coloringLog
    rm runColoring
    touch jobFinished
  fi

  # loop over all multipoint cases
  for (( c=0; c<${nCases}; ++c )) 
  do

    # flow solver
    if [ -e "runFlowSolver_FC${c}" ]
    then
      # copy setupDicts
      cp system/* ../FlowConfig${c}/system/
      # copy updated points and BCs
      if [ $nProcs -eq 1 ]
      then
        cp constant/polyMesh/points.gz ../FlowConfig${c}/constant/polyMesh/
        cp 0/* ../FlowConfig${c}/0/
        rm -rf ../FlowConfig${c}/{1..9}*
      else  
        for (( p=0; p<${nProcs}; ++p )); do
            cp processor${p}/constant/polyMesh/points.gz ../FlowConfig${c}/processor${p}/constant/polyMesh/
            cp processor${p}/0/* ../FlowConfig${c}/processor${p}/0/
            rm -rf ../FlowConfig${c}/processor${p}/{1..9}*
        done
      fi
      # run flow solver
      cd ../FlowConfig${c}
      ${exec} -np ${nProcs} simpleDAFoam ${jobFlag} > flowLog
      # also copy the objFuncs.dat back to the main folder
      # in case we need to calculate alpha sens
      cp objFuncs.dat ../MultiPointMain/objFuncs.dat
      cd ../MultiPointMain
      rm runFlowSolver_FC${c}
      touch jobFinished_FC${c}

    fi
  
    # adjoint solver
    if [ -e "runAdjointSolver_FC${c}" ]
    then
      # copy setupDicts, colors, etc
      mv *.bin ../FlowConfig${c}/
      cp system/* ../FlowConfig${c}/system/
      cd ../FlowConfig${c}
      # run adjoint solver
      ${exec} -np ${nProcs} simpleDAFoam ${jobFlag} > adjointLog
      mv *.bin ../MultiPointMain/
      cd ../MultiPointMain
      rm runAdjointSolver_FC${c}
      touch jobFinished_FC${c}
    fi


  done
  
  sleep 5

done

