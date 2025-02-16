#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")

if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [0.0]},
        "useWallFunction": False,
        "transport:nu": 1.5e-5,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 0.1,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setOption("function", {"CD": {"direction": [0.0, 1.0, 0.0]}})
DASolver.updateDAOption()
function = DASolver.getOption("function")
if function["CD"]["direction"][1] != 1.0:
    print("setOption failed")
    exit(1)
DASolver.calcPrimalResidualStatistics("print")
