#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
import numpy as np
from pyofm import PYOFM
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/NACA0012DynamicMeshV4")

if gcomm.rank == 0:
    os.system("rm -rf processor*")
    replace_text_in_file("system/controlDict", "SolverPerformance 1;", "SolverPerformance 0;")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DAPimpleDyMFoam",
    "dynamicMesh": {
        "active": True,
        "mode": "rotation",
        "center": [0.25, 0.0, 0.0],
        "axis": "z",
        "omega": -0.5,
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.deformDynamicMesh()
DASolver()

# read the U field at  0.01 and verify its norm
nLocalCells = DASolver.solver.getNLocalCells()
U = np.zeros(nLocalCells * 3)
ofm = PYOFM(gcomm)
ofm.readField("U", "volVectorField", "0.01", U)
UNorm = np.linalg.norm(U)
UNorm = gcomm.allreduce(UNorm, op=MPI.SUM)
print("UNorm", UNorm)

if abs(1176.2700572104368 - UNorm) / 1176.2700572104368 > 1e-6:
    print("DAPimpleDyMFoam test failed!")
    exit(1)
else:
    print("DAPimpleDyMFoam test passed!")
