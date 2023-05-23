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

dp0 = 6.8685e-06

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "useAD": {"mode": "reverse"},
    "printInterval": 10,
    "primalMinResTol": 1e-3,
    "primalBC": {
        "fvSource": {"value": dp0, "comp": 0},
    },
    "tensorflow": {
        "active": True,
        "modelName": "model",
        "nInputs": 9,
        "nOutputs": 1,
    },
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
}

# DAFoam
DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
DASolver()

funcs = {}
DASolver.evalFunctions(funcs, evalFuncs=["CD", "CL"])
if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
