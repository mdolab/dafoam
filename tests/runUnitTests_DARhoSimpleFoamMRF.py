#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
import os
import numpy as np
from dafoam import PYDAFOAM

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/CompressorFluid")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor* *.bin")
    os.system("cp -r 0.compressible 0")
    os.system("cp -r 0/U.subsonic 0/U")
    os.system("cp -r 0/p.subsonic 0/p")
    os.system("cp -r constant/thermophysicalProperties.e constant/thermophysicalProperties")
    os.system("cp -r constant/MRFProperties.subsonic constant/MRFProperties")
    os.system("cp -r system/fvSolution.subsonic system/fvSolution")
    os.system("cp -r system/fvSchemes.subsonic system/fvSchemes")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

daOptions = {
    "designSurfaces": ["blade"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-11,
    "primalMinResTolDiff": 1e4,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [0.0, 0.0, 100.0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [300.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [101325.0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [1e-4]},
        "useWallFunction": True,
    },
    "function": {
        "CMZ": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["blade"],
            "axis": [0.0, 0.0, 1.0],
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0 / (0.5 * 10.0 * 10.0 * 1.0 * 1.0),
        },
    },
    "decomposeParDict": {"preservePatches": ["per1", "per2"]},
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
funcs = {}
DASolver.evalFunctions(funcs)

print(funcs)

if abs(funcs["CMZ"] - 0.13739475620718933) / 0.13739475620718933 > 1e-8:
    print("DARhoSimpleFoamMRF test failed!")
    exit(1)
else:
    print("DARhoSimpleFoamMRF test passed!")
