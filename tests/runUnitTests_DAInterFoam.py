#!/usr/bin/env python

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
import numpy as np
from pyofm import PYOFM
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/DamBreak")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")
    replace_text_in_file("system/controlDict", "endTime         0.04;", "endTime         0.2;")
    replace_text_in_file("system/controlDict", "deltaT          0.004;", "deltaT          0.01;")


daOptions = {
    "solverName": "DAInterFoam",
    "useDdtCorr": True,
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["lowerWall"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "timeOp": "average",
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
DASolver.solver.calcPrimalResidualStatistics("print")

funcs = {}
DASolver.evalFunctions(funcs)

if abs(0.7314455555309382 - funcs["CD"]) / 0.7314455555309382 > 1e-8:
    print("DAInterFoam test failed!")
    exit(1)
else:
    print("DAInterFoam test passed!")

# read the U field at  2 and verify its norm
nLocalCells = DASolver.solver.getNLocalCells()
U = np.zeros(nLocalCells * 3)
ofm = PYOFM(gcomm)
ofm.readField("U", "volVectorField", "0.2", U)
UNorm = np.linalg.norm(U)
UNorm = gcomm.allreduce(UNorm, op=MPI.SUM)
print("UNorm", UNorm)

if abs(65.96431512444805 - UNorm) / 65.96431512444805 > 1e-8:
    print("DAInterFoam test failed!")
    exit(1)
else:
    print("DAInterFoam test passed!")
