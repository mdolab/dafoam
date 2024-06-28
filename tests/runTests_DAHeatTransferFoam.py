#!/usr/bin/env python
"""
Run Python tests for DASolidDisplacementFoam
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import sys
import os
from pygeo import *
from pyspline import *
from idwarp import *
import numpy as np
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ChannelConjugateHeat/thermal")

aeroOptions = {
    "designSurfaces": ["channel_outer", "channel_inner", "channel_sides"],
    "solverName": "DAHeatTransferFoam",
    "objFunc": {
        "HF_INNER": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["channel_inner"],
                "scale": 1,
                "addToAdjoint": True,
            }
        },
        "HF_OUTER": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["channel_outer"],
                "scale": 1,
                "addToAdjoint": True,
            }
        },
        "R1": {
            "part1": {
                "type": "location",
                "source": "patchToFace",
                "patches": ["channel_inner"],
                "mode": "maxRadius",
                "axis": [0.0, 1.0, 0.0],
                "center": [0.5, 0.05, -0.05],
                "coeffKS": 1.0,
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "R1KS": {
            "part1": {
                "type": "location",
                "source": "patchToFace",
                "patches": ["channel_inner"],
                "mode": "maxRadiusKS",
                "axis": [0.0, 1.0, 0.0],
                "center": [0.505, 0.056, 0.026],
                "coeffKS": 1.0,
                "scale": 1.0,
                "snapCenter2Cell": True,
                "addToAdjoint": True,
            }
        },
        "IR1KS": {
            "part1": {
                "type": "location",
                "source": "patchToFace",
                "patches": ["channel_inner"],
                "mode": "maxInverseRadiusKS",
                "axis": [0.0, 1.0, 0.0],
                "center": [0.5, 0.05, -0.05],
                "coeffKS": 1.0,
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "debug": False,
    "primalMinResTol": 1e-12,
    "couplingInfo": {
        "aerothermal": {
            "active": True,
            "couplingSurfaceGroups": {
                "wallGroup": ["channel_outer"],
            },
        }
    },
    "fvSource": {
        "source1": {
            "type": "heatSource",
            "source": "cylinderToCell",
            "p1": [0.6, 0.055, 0.025],
            "p2": [1.0, 0.055, 0.025],
            "radius": 0.01,
            "power": 1000.0,
        },
        "source2": {
            "type": "heatSource",
            "source": "cylinderSmooth",
            "center": [0.2, 0.055, 0.025],
            "axis": [1.0, 0.0, 0.0],
            "length": 0.4,
            "radius": 0.005,
            "power": 1000.0,
            "eps": 0.001,
        },
    },
}
DASolver = PYDAFOAM(options=aeroOptions, comm=MPI.COMM_WORLD)

DASolver()
funcs = {}
evalFuncs = ["HF_INNER", "HF_OUTER", "R1", "R1KS", "IR1KS"]
DASolver.evalFunctions(funcs, evalFuncs)

# test getThermal and setThermal
states = DASolver.vec2Array(DASolver.wVec)
volCoords = DASolver.vec2Array(DASolver.xvVec)
thermal = np.ones(DASolver.solver.getNCouplingFaces() * 2)
DASolver.solver.setThermal(thermal)
DASolver.solver.getThermal(volCoords, states, thermal)
TNorm = np.linalg.norm(thermal)
TNormSum = gcomm.allreduce(TNorm, op=MPI.SUM)
funcs["TNormSum"] = TNormSum

if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
