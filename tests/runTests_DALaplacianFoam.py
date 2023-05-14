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

# test incompressible solvers
aeroOptions = {
    "designSurfaces": ["patch1"],
    "solverName": "DALaplacianFoam",
    "printIntervalUnsteady": 1,
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
                "divByTotalVol": 1,
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "HF": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["patch4"],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "TMEAN": {
            "part1": {
                "type": "patchMean",
                "source": "patchToFace",
                "patches": ["patch1"],
                "varName": "T",
                "varType": "scalar",
                "component": 0,
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "debug": False,
    "primalMinResTol": 1e-16,
}
DASolver = PYDAFOAM(options=aeroOptions, comm=MPI.COMM_WORLD)

nCells, nFaces = DASolver._getSurfaceSize(DASolver.couplingSurfacesGroup)
TGrad = np.ones(nFaces) * 5
DASolver.solver.setThermal("heatFlux", TGrad)

DASolver()
funcs = {}
evalFuncs = ["TVOL", "HF", "TMEAN"]
DASolver.evalFunctions(funcs, evalFuncs)

states = DASolver.vec2Array(DASolver.wVec)
volCoords = DASolver.vec2Array(DASolver.xvVec)
thermal = np.zeros(DASolver.solver.getNCouplingFaces())
DASolver.solver.getThermal("temperature", volCoords, states, thermal)
TNorm = np.linalg.norm(thermal / 100)
TNormSum = gcomm.allreduce(TNorm, op=MPI.SUM)
funcs["TFormSum"] = TNormSum

if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
