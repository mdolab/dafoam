#!/bin/bash

set -e

export DAFOAM_NO_WARNINGS=1
unset WM_CODI_AD_MODE

# build 
. ~/packages/OpenFOAM/OpenFOAM-v1812/etc/bashrc
./Allclean
./Allmake

# build reversed mode
. ~/packages/OpenFOAM/OpenFOAM-v1812-ADR/etc/bashrc
./Allclean
./Allmake

# build forward mode
. ~/packages/OpenFOAM/OpenFOAM-v1812-ADF/etc/bashrc
./Allclean
./Allmake

unset WM_CODI_AD_MODE
. ~/packages/OpenFOAM/OpenFOAM-v1812/etc/bashrc
pip install .
