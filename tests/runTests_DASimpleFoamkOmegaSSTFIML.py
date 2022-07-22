#!/usr/bin/env python
"""
Run Python tests for DASimpleFoam
"""

from mpi4py import MPI
from dafoam import PYDAFOAM

from testFuncs import *

os.chdir("./input/PeriodicHill")

gcomm = MPI.COMM_WORLD

if gcomm.rank == 0:
    os.system("rm -rf processor*")

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1e-3,
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["bottomWall"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1000.0,
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["bottomWall"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 1.0, 0.0],
                "scale": 1000.0,
                "addToAdjoint": True,
            }
        },
    },
    "fvSource": {
        "gradP": {
            "type": "uniformPressureGradient",
            "value": 6.8685e-06,
            "direction": [1.0, 0.0, 0.0],
        }
    },
}

# DAFoam
DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
DASolver()

funcs = {}
DASolver.evalFunctions(funcs, evalFuncs=["CD", "CL"])
if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
