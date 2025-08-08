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
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

# read the U field at  2 and verify its norm
nLocalCells = DASolver.solver.getNLocalCells()
U = np.zeros(nLocalCells * 3)
ofm = PYOFM(gcomm)
ofm.readField("U", "volVectorField", "0.2", U)
UNorm = np.linalg.norm(U)
UNorm = gcomm.allreduce(UNorm, op=MPI.SUM)
print("UNorm", UNorm)

if abs(79.4050157054846 - UNorm) / 79.4050157054846 > 1e-10:
    print("DAInterFoam test failed!")
    exit(1)
else:
    print("DAInterFoam test passed!")
