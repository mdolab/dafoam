#!/usr/bin/env python
"""
Run Python tests for DASimpleFoam
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import numpy as np
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./input/NACA0012")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.sstlm constant/turbulenceProperties")

replace_text_in_file("system/fvSchemes", "meshWave", "meshWaveFrozen")

aeroOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaceFamily": "designSurface",
    "designSurfaces": ["wing"],
    "primalMinResTol": 1e-10,
    "primalBC": {
        "UIn": {"variable": "U", "patches": ["inout"], "value": [9.9985, 0.1745, 0.0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "fixedDirection",
                "direction": [0.999847695156392, 0.017452406437284, 0.0],
                "scale": 1.0 / (0.5 * 10 * 10 * 0.1),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "fixedDirection",
                "direction": [-0.017452406437284, 0.999847695156392, 0.0],
                "scale": 1.0 / (0.5 * 10 * 10 * 0.1),
                "addToAdjoint": True,
            }
        },
    },
}
DASolver = PYDAFOAM(options=aeroOptions, comm=MPI.COMM_WORLD)
DASolver()
funcs = {}
evalFuncs = ["CD", "CL"]
DASolver.evalFunctions(funcs, evalFuncs)

# Run
if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
