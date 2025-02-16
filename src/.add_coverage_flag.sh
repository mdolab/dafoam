#!/usr/bin/env bash

sed -i 's/-std=c++11/-std=c++11 -fprofile-arcs -ftest-coverage/g' adjoint/Make/options
sed -i 's/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX)/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX) -lgcov/g' adjoint/Make/options

sed -i 's/-std=c++11/-std=c++11 -fprofile-arcs -ftest-coverage/g' newTurbModels/incompressible/Make/options
sed -i 's/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX)/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX) -lgcov/g' newTurbModels/incompressible/Make/options

sed -i 's/-std=c++11/-std=c++11 -fprofile-arcs -ftest-coverage/g' newTurbModels/compressible/Make/options
sed -i 's/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX)/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX) -lgcov/g' newTurbModels/compressible/Make/options

sed -i 's/-std=c++11/-std=c++11 -fprofile-arcs -ftest-coverage/g' pyDASolvers/Make/options
sed -i 's/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX)/-lfiniteVolume$(WM_CODI_AD_LIB_POSTFIX) -lgcov/g' pyDASolvers/Make/options


