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
    replace_text_in_file("system/controlDict", "SolverPerformance 1;", "SolverPerformance 0;")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DAPimpleDyMFoam",
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

# read the U field at  0.01 and verify its norm
nLocalCells = DASolver.solver.getNLocalCells()
U = np.zeros(nLocalCells * 3)
ofm = PYOFM(gcomm)
ofm.readField("U", "volVectorField", "0.01", U)
UNorm = np.linalg.norm(U)
UNorm = gcomm.allreduce(UNorm, op=MPI.SUM)
print("UNorm", UNorm)

if abs(1183.7173503783392 - UNorm) / 1183.7173503783392 > 1e-6:
    print("DAPimpleDyMFoam test failed!")
    exit(1)
else:
    print("DAPimpleDyMFoam test passed!")
