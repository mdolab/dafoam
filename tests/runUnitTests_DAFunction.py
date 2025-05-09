#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")

# NOTE: we will test DAFunction for incompressible, compressible, and solid solvers

# ********************
# incompressible tests
# ********************
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    replace_text_in_file(
        "0/T", "fixedValue ;", "fixedWallHeatFlux; heatFlux 8.2; nu 1.5e-5; Pr 0.7; Prt 0.85; Cp 1004.0;"
    )
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
            "scale": 1.0,
        },
        "PMean": {
            "type": "patchMean",
            "source": "patchToFace",
            "patches": ["inlet"],
            "varName": "p",
            "varType": "scalar",
            "index": 0,
            "scale": 1.0,
        },
        "UMean": {
            "type": "patchMean",
            "source": "patchToFace",
            "patches": ["outlet"],
            "varName": "U",
            "varType": "vector",
            "index": 0,
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
            "snapCenter2Cell": True,
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
            "index": 0,
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
            "index": 0,
            "isSquare": 1,
            "multiplyVol": 0,
            "divByTotalVol": 1,
            "scale": 1.0,
        },
        "HVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "wallHeatFlux",
            "varType": "scalar",
            "indices": [0],
            "timeDependentRefData": False,
        },
        "PVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "field",
            "varName": "p",
            "varType": "scalar",
            "indices": [0],
            "timeDependentRefData": False,
        },
        "PProbe": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "probePoint",
            "probePointCoords": [[0.51, 0.52, 0.53], [0.2, 0.3, 0.4]],
            "varName": "p",
            "varType": "scalar",
            "indices": [0],
            "timeDependentRefData": False,
        },
        "UOutVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["outlet"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "U",
            "varType": "vector",
            "indices": [0],
            "timeDependentRefData": False,
        },
        "ResNorm": {
            "type": "residualNorm",
            "source": "allCells",
            "scale": 1.0,
            "resWeight": {"URes": 0.1, "pRes": 0.01, "phiRes": 10.0, "TRes": 0.01, "nuTildaRes": 100.0},
            "timeDependentRefData": False,
            "timeOp": "average",
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
    "HFX": 8.200000003826663,
    "PMean": 77.80996323506456,
    "UMean": 15.469850053816028,
    "skewness": 1.3140396235456926,
    "nonOrtho": 26.209186150313975,
    "faceOrthogonality": 7.948460246568271,
    "RMax": 1.295712391984845,
    "RMaxKS": 0.8579812873447494,
    "IRMaxKS": 9.132901616926853,
    "PVolSum": 23.576101529517096,
    "UVolSum": 2004.7819430730992,
    "HVar": 67.2400000627573,
    "PVar": 2.476982282327677,
    "PProbe": 3.6866882754983203,
    "UOutVar": 0.5085431532392312,
    "ResNorm": 0.5124351660034533,
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
        "HVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "wallHeatFlux",
            "varType": "scalar",
            "indices": [0],
            "timeDependentRefData": False,
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
    "HFX": 2564.397361965973,
    "TTR": 1.0002129350727795,
    "MFR": 128.2374484095998,
    "TPR": 0.9933893131111589,
    "HVar": 12470853.906556124,
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
        print("DAFunction comp test failed for ", failedFunc)
    exit(1)
else:
    if gcomm.rank == 0:
        print("DAFunction comp test passed!")
