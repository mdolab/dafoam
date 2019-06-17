#!/bin/bash

while true 
do
    read -p "Delete everything and resume to the default setup (y/n)?" yn
    case $yn in
        [Yy]* ) 
            # clean everyting
            echo "Cleaning..."
            rm -rf 0
            rm -rf postProcessing
            rm -rf *.bin *.info *Log* *.dat 
            rm -rf jobFinished runCheckMesh* runFlowSolver* runAdjointSolver* runColoring
            rm -rf constant/polyMesh/
            rm -rf processor*
            rm -rf {1..9}*
            rm -rf ../FlowConfig*/*
            killall -9 foamRunMultiPoint.sh
            exit
            ;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

