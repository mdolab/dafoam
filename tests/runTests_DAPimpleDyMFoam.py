#!/usr/bin/env python
"""
Run Python tests for DAPimpleDyMFoam
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
from testFuncs import *

os.chdir("./input/NACA0012DynamicMesh")

replace_text_in_file("system/controlDict", "endTime         0.01;", "endTime         0.001;")

daOptions = {
    "solverName": "DAPimpleDyMFoam",
    "primalOnly": True,
    "useAD": {"mode": "fd"},
    "rigidBodyMotion": {
        "mode": "translationCoupled",
        "patchNames": ["wing"],
        "mass": 20.0,
        "damping": 0.0,
        "stiffness": 1000.0,
        "forceScale": 1.0,
        "V0": 0.1,
        "direction": [0.0, 1.0, 0.0],
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=MPI.COMM_WORLD)
DASolver.solvePrimal()

daOptions1 = {
    "solverName": "DAPimpleDyMFoam",
    "primalOnly": True,
    "useAD": {"mode": "fd"},
    "rigidBodyMotion": {
        "mode": "translation",
        "patchNames": ["wing"],
        "frequency": 1.0,
        "amplitude": 0.1,
        "phase": 0.0,
        "direction": [0.0, 1.0, 0.0],
    },
}
DASolver1 = PYDAFOAM(options=daOptions1, comm=MPI.COMM_WORLD)
DASolver1.solvePrimal()

daOptions2 = {
    "solverName": "DAPimpleDyMFoam",
    "primalOnly": True,
    "useAD": {"mode": "fd"},
}
DASolver2 = PYDAFOAM(options=daOptions2, comm=MPI.COMM_WORLD)
DASolver2.solvePrimal()

# we just run it without checking the output
funcs = {}
funcs["fail"] = 0

if MPI.COMM_WORLD.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
