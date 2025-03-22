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

os.chdir("./reg_test_files-main/NACA0012UnsteadyV4")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DAIrkPimpleFoam",
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

# read the U field at  2 and verify its norm
nLocalCells = DASolver.solver.getNLocalCells()
U = np.zeros(nLocalCells * 3)
ofm = PYOFM(gcomm)
ofm.readField("U", "volVectorField", "0.005", U)
UNorm = np.linalg.norm(U)
UNorm = gcomm.allreduce(UNorm, op=MPI.SUM)
print("UNorm", UNorm)

if abs(1173.6440797856071 - UNorm) / 1173.6440797856071 > 1e-10:
    print("DAIrkPimpleFoam test failed!")
    exit(1)
else:
    print("DAIrkPimpleFoam test passed!")
