#!/usr/bin/env bash

sed -i 's/EXE_INC =/EXE_INC = -fprofile-arcs -ftest-coverage/g' adjoint/Make/options
sed -i 's/-lfiniteVolume$(WM_AD_MODE)/-lfiniteVolume$(WM_AD_MODE) -lgcov/g' adjoint/Make/options

sed -i 's/EXE_INC =/EXE_INC = -fprofile-arcs -ftest-coverage/g' newTurbModels/incompressible/Make/options
sed -i 's/-lfiniteVolume$(WM_AD_MODE)/-lfiniteVolume$(WM_AD_MODE) -lgcov/g' newTurbModels/incompressible/Make/options

sed -i 's/EXE_INC =/EXE_INC = -fprofile-arcs -ftest-coverage/g' newTurbModels/compressible/Make/options
sed -i 's/-lfiniteVolume$(WM_AD_MODE)/-lfiniteVolume$(WM_AD_MODE) -lgcov/g' newTurbModels/compressible/Make/options

sed -i 's/EXE_INC =/EXE_INC = -fprofile-arcs -ftest-coverage/g' pyDASolvers/Make/options
sed -i 's/-lfiniteVolume$(WM_AD_MODE)/-lfiniteVolume$(WM_AD_MODE) -lgcov/g' pyDASolvers/Make/options


