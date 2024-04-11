#!/usr/bin/env python
"""
Run Python tests for DARhoSimpleFoam
"""

from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
import sys
import os
from pygeo import *
from pyspline import *
from idwarp import *
from pyoptsparse import Optimization, OPT
import numpy as np
from testFuncs import *
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)

calcFDSens = 0
if len(sys.argv) != 1:
    if sys.argv[1] == "calcFDSens":
        calcFDSens = 1

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/CurvedCubeSnappyHexMesh")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.compressible 0")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWaveFrozen;", "meshWave;")
    os.system("rhoSimpleFoam")
    os.system("getFIData -refFieldName wallHeatFlux -refFieldType scalar -time 2000")
    os.system("mv 2000/*Data.gz 0/")
    os.system("cp -r constant/turbulenceProperties.sst constant/turbulenceProperties")
    os.system("rm -rf 2000")

replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

U0 = 50.0
p0 = 101325.0
T0 = 300.0
A0 = 1.0
rho0 = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DARhoSimpleFoam",
    "designSurfaces": ["wallsbump"],
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1e-12,
    "primalBC": {
        "UIn": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [T0]},
        "useWallFunction": False,
    },
    "regressionModel": {
        "active": True,
        "modelType": "neuralNetwork",
        "inputNames": ["KoU2", "ReWall", "CoP", "TauoK"],
        "outputName": "betaFIOmega",
        "hiddenLayerNeurons": [10, 10],
        "inputShift": [0.0, 0.0, 0.0, 0.0],
        "inputScale": [1.0, 1.0, 0.0001, 1.0],
        "outputShift": 1.0,
        "outputScale": 1.0,
        "activationFunction": "tanh",
        "printInputInfo": False,
        "defaultOutputValue": 1.0,
        "writeFeatures": True,
    },
    "objFunc": {
        "VAR": {
            "flux": {
                "type": "variance",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "scale": 1.0,
                "mode": "surface",
                "varName": "wallHeatFlux",
                "varType": "scalar",
                "surfaceNames": ["wallsbump"],
                "addToAdjoint": True,
                "timeDependentRefData": False,
            },
        },
    },
    "adjStateOrdering": "cell",
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": 1e-4, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "parameter": {"designVarType": "RegPar"},
    },
}


def regModel(val, DASolver):
    for idxI in range(len(val)):
        val1 = float(val[idxI])
        DASolver.setRegressionParameter(idxI, val1)


# DAFoam
DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
nParameters = DASolver.solver.getNRegressionParameters()
parameter0 = np.ones(nParameters) * 0.01
DASolver.addInternalDV("parameter", parameter0, regModel, lower=-10, upper=10, scale=100.0)
# set evalFuncs
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# optFuncs
optFuncs.DASolver = DASolver
optFuncs.DVGeo = None
optFuncs.DVCon = None
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# Run
if calcFDSens == 1:
    optFuncs.calcFDSens(objFun=optFuncs.calcObjFuncValues, fileName="sensFD.txt")
    exit(0)
else:
    DASolver.runColoring()
    iDV = DASolver.getInternalDVDict()
    funcs = {}
    funcs, fail = optFuncs.calcObjFuncValues(iDV)
    funcsSens = {}
    funcsSens, fail = optFuncs.calcObjFuncSens(iDV, funcs)

    norm = np.linalg.norm(funcsSens["VAR"]["parameter"])
    funcsSens["VAR"]["parameter"] = norm

    # test RBF regression
    aeroOptions["regressionModel"] = {
        "active": True,
        "modelType": "radialBasisFunction",
        "inputNames": ["KoU2", "ReWall", "CoP", "TauoK"],
        "outputName": "betaFIOmega",
        "nRBFs": 20,
        "inputShift": [0.0, 0.0, 0.0, 0.0],
        "inputScale": [1.0, 1.0, 0.0001, 1.0],
        "outputShift": 1.0,
        "outputScale": 1.0,
        "printInputInfo": True,
        "defaultOutputValue": 1.0,
        "writeFeatures": True,
    }

    DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
    nParameters = DASolver.solver.getNRegressionParameters()
    parameter0 = np.ones(nParameters) * 0.03
    DASolver.addInternalDV("parameter", parameter0, regModel, lower=-10, upper=10, scale=100.0)
    DASolver()
    funcs1 = {}
    DASolver.evalFunctions(funcs1, evalFuncs=["VAR"])
    funcs["VAR_RBF"] = funcs1["VAR"]

    # test ReLU
    aeroOptions["regressionModel"] = {
        "active": True,
        "modelType": "neuralNetwork",
        "inputNames": ["KoU2", "ReWall", "CoP", "TauoK"],
        "outputName": "betaFIOmega",
        "hiddenLayerNeurons": [10, 10],
        "inputShift": [0.0, 0.0, 0.0, 0.0],
        "inputScale": [1.0, 1.0, 0.0001, 1.0],
        "outputShift": 1.0,
        "outputScale": 1.0,
        "activationFunction": "ReLU",
        "leakyCoeff": 0.1,
        "printInputInfo": False,
        "defaultOutputValue": 1.0,
    }

    DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
    nParameters = DASolver.solver.getNRegressionParameters()
    parameter0 = np.ones(nParameters) * 0.05
    DASolver.addInternalDV("parameter", parameter0, regModel, lower=-10, upper=10, scale=100.0)
    DASolver()
    funcs2 = {}
    DASolver.evalFunctions(funcs2, evalFuncs=["VAR"])
    funcs["VAR_ReLU"] = funcs2["VAR"]

    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-4, 1e-6)
