#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
import numpy as np
from pyofm import PYOFM

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.hisa/* 0/")
    os.system("cp -r system.hisa.unsteady/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

daOptions = {
    "solverName": "DAHisaFoam",
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

funcs = {}
DASolver.evalFunctions(funcs)

if abs(39135.17439747127 - funcs["CD"]) / 39135.17439747127 > 1e-8:
    print("DAInterFoam test failed!")
    exit(1)
else:
    print("DAInterFoam test passed!")
