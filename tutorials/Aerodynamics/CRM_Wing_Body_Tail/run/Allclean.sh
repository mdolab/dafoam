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
            rm -rf constant/extendedFeatureEdgeMesh
            rm -rf constant/triSurface/*eMesh*
            rm -rf *.bin *.info *Log* *.dat *.xyz *.stl figure.png
            rm -rf jobFinished runCheckMesh* runFlowSolver* runAdjointSolver* runColoring
            rm -rf constant/polyMesh/*
            rm -rf constant/triSurface
            rm -rf processor*
            rm -rf {1..9}*
            killall -9 foamRun.sh
            exit
            ;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

