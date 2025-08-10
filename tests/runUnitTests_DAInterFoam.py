#!/usr/bin/env python

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
import numpy as np
from pyofm import PYOFM

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/DamBreak")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")

daOptions = {
    "solverName": "DAInterFoam",
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

if abs(0.7724431359627557 - funcs["CD"]) / 0.7724431359627557 > 1e-10:
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

if abs(55.535762590061054 - UNorm) / 55.535762590061054 > 1e-10:
    print("DAInterFoam test failed!")
    exit(1)
else:
    print("DAInterFoam test passed!")
