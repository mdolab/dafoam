#!/usr/bin/env python
"""
Run Python tests for DASolidDisplacementFoam
"""

from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
import sys
import os
from pygeo import *
from pyspline import *
from idwarp import *
import numpy as np
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./input/flange")

TRef = 1.0

# test incompressible solvers
aeroOptions = {
    "designSurfaces": ["patch1"],
    "solverName": "DALaplacianFoam",
    "useAD": {"mode": "fd"},
    "unsteadyAdjoint": {"mode": "hybridAdjoint", "nTimeInstances": 3, "periodicity": 0.1},
    "objFunc": {
        "TVOL": {
            "part1": {
                "type": "variableVolSum",
                "source": "boxToCell",
                "min": [-50.0, -50.0, -50.0],
                "max": [50.0, 50.0, 50.0],
                "varName": "T",
                "varType": "scalar",
                "component": 0,
                "isSquare": 0,
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "debug": False,
    "primalMinResTol": 1e-16,
}
DASolver = PYDAFOAM(options=aeroOptions, comm=MPI.COMM_WORLD)
DASolver()
funcs = {}
evalFuncs = ["TVOL"]
DASolver.evalFunctions(funcs, evalFuncs)

if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
