#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")

# NOTE: we will test DAFunction for incompressible, compressible, and solid solvers

# ********************
# incompressible tests
# ********************
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [0.0]},
        "useWallFunction": False,
        "transport:nu": 1.5e-5,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 0.1,
        },
        "CMZ": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["walls"],
            "axis": [0.0, 0.0, 1.0],
            "center": [0.5, 0.5, 0.5],
            "scale": 1.0,
        },
        "TP1": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["inlet"],
            "scale": 1.0 / (0.5 * U0 * U0),
        },
        "HFX": {
            "type": "wallHeatFlux",
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 0.001,
        },
        "PMean": {
            "type": "patchMean",
            "source": "patchToFace",
            "patches": ["inlet"],
            "varName": "p",
            "varType": "scalar",
            "component": 0,
            "scale": 1.0,
        },
        "UMean": {
            "type": "patchMean",
            "source": "patchToFace",
            "patches": ["outlet"],
            "varName": "U",
            "varType": "vector",
            "component": 0,
            "scale": 1.0,
        },
        "skewness": {
            "type": "meshQualityKS",
            "source": "allCells",
            "coeffKS": 10.0,
            "metric": "faceSkewness",
            "scale": 1.0,
        },
        "nonOrtho": {
            "type": "meshQualityKS",
            "source": "allCells",
            "coeffKS": 1.0,
            "metric": "nonOrthoAngle",
            "scale": 1.0,
        },
        "faceOrthogonality": {
            "type": "meshQualityKS",
            "source": "allCells",
            "coeffKS": 1.0,
            "metric": "faceOrthogonality",
            "scale": 1.0,
        },
        "RMax": {
            "type": "location",
            "source": "patchToFace",
            "patches": ["walls"],
            "mode": "maxRadius",
            "axis": [0.0, 0.0, 1.0],
            "center": [0.5, 0.5, 0.5],
            "scale": 1.0,
        },
        "RMaxKS": {
            "type": "location",
            "source": "patchToFace",
            "patches": ["walls"],
            "mode": "maxRadiusKS",
            "axis": [0.0, 0.0, 1.0],
            "center": [0.5, 0.5, 0.5],
            "coeffKS": 20.0,
            "scale": 1.0,
            "snapCenter2Cell": False,
        },
        "IRMaxKS": {
            "type": "location",
            "source": "patchToFace",
            "patches": ["walls"],
            "mode": "maxInverseRadiusKS",
            "axis": [0.0, 0.0, 1.0],
            "center": [0.5, 0.5, 0.5],
            "coeffKS": 20.0,
            "scale": 1.0,
        },
        "PVolSum": {
            "type": "variableVolSum",
            "source": "allCells",
            "varName": "p",
            "varType": "scalar",
            "component": 0,
            "isSquare": 0,
            "multiplyVol": 1,
            "divByTotalVol": 0,
            "scale": 1.0,
        },
        "UVolSum": {
            "type": "variableVolSum",
            "source": "boxToCell",
            "min": [0.2, 0.2, 0.3],
            "max": [0.8, 0.8, 0.9],
            "varName": "U",
            "varType": "vector",
            "component": 0,
            "isSquare": 1,
            "multiplyVol": 0,
            "divByTotalVol": 1,
            "scale": 1.0,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

funcs = {}
DASolver.evalFunctions(funcs)

if gcomm.rank == 0:
    print(funcs)

funcs_ref = {
    "CD": 2.2801022493682037,
    "CMZ": 7.1768354637131795,
    "TP1": 2.5561992647012914,
    "HFX": 1.5730364723151125,
    "PMean": 77.80996323506456,
    "UMean": 15.469850053816028,
    "skewness": 1.3140396235456926,
    "nonOrtho": 26.209186150313975,
    "faceOrthogonality": 7.948460246568271,
    "RMax": 1.295712391984845,
    "RMaxKS": 0.8057075945057248,
    "IRMaxKS": 9.132901616926853,
    "PVolSum": 23.576101529517096,
    "UVolSum": 2004.7819430730992,
}

fail = 0
failedFunc = []
for func in list(funcs_ref.keys()):
    diff = abs(funcs_ref[func] - funcs[func]) / funcs_ref[func]
    if diff > 1e-10:
        fail += 1
        failedFunc.append(func)

if fail != 0:
    if gcomm.rank == 0:
        print("DAFunction incomp test failed for ", failedFunc)
    exit(1)
else:
    if gcomm.rank == 0:
        print("DAFunction incomp test passed!")


# ********************
# compressible tests
# ********************
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 100.0

daOptions = {
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [310.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [101325.0]},
        "useWallFunction": True,
    },
    "function": {
        "HFX": {
            "type": "wallHeatFlux",
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 1.0,
        },
        "TTR": {
            "type": "totalTemperatureRatio",
            "source": "patchToFace",
            "patches": ["inlet", "outlet"],
            "inletPatches": ["inlet"],
            "outletPatches": ["outlet"],
            "scale": 1.0,
        },
        "MFR": {
            "type": "massFlowRate",
            "source": "patchToFace",
            "patches": ["inlet"],
            "scale": -1.0,
        },
        "TPR": {
            "type": "totalPressureRatio",
            "source": "patchToFace",
            "patches": ["inlet", "outlet"],
            "inletPatches": ["inlet"],
            "outletPatches": ["outlet"],
            "scale": 1.0,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

funcs = {}
DASolver.evalFunctions(funcs)

if gcomm.rank == 0:
    print(funcs)

funcs_ref = {"HFX": 15.494946599784122, "TTR": 1.000000415241806, "MFR": 128.22449523986853, "TPR": 0.9934250594975235}

fail = 0
failedFunc = []
for func in list(funcs_ref.keys()):
    diff = abs(funcs_ref[func] - funcs[func]) / funcs_ref[func]
    if diff > 1e-10:
        fail += 1
        failedFunc.append(func)

if fail != 0:
    if gcomm.rank == 0:
        print("DAFunction comp test failed for ", failedFunc)
    exit(1)
else:
    if gcomm.rank == 0:
        print("DAFunction comp test passed!")
