#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/NACA0012DynamicMeshV4")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DAPimpleDyMFoam",
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
