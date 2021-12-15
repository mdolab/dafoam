#!/usr/bin/env python
"""
Run Python tests for DAPisoFoam
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

calcFDSens = 0
if len(sys.argv) != 1:
    if sys.argv[1] == "calcFDSens":
        calcFDSens = 1

gcomm = MPI.COMM_WORLD

os.chdir("./input/CurvedCubeHexMesh")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.unsteady 0")
    os.system("cp -r system/controlDict.unsteady system/controlDict")
    os.system("cp -r system/fvSchemes.unsteady system/fvSchemes")
    os.system("cp -r system/fvSolution.unsteady system/fvSolution")
    os.system("cp -r constant/turbulenceProperties.safv3 constant/turbulenceProperties")

# test incompressible solvers
daOptions = {
    "solverName": "DAPisoFoam",
    "designSurfaceFamily": "designSurface",
    "designSurfaces": ["wallsbump"],
    "useAD": {"mode": "fd"},
    "unsteadyAdjoint": {"mode": "hybridAdjoint", "nTimeInstances": 3, "periodicity": 1.0},
    "fvSource": {
        "point1": {
            "type": "actuatorPoint",
            "smoothFunction": "hyperbolic",
            "center": [0.5, 0.5, 0.5],  # center and size define a rectangular
            "size": [0.2, 0.2, 0.2],
            "amplitude": [0.0, 0.2, 0.0],
            "phase": 0.0,
            "thrustDirIdx": 0,
            "periodicity": 1.0,
            "eps": 10.0,
            "scale": 10.0,  # scale the source such the integral equals desired thrust
        },
        "point2": {
            "type": "actuatorPoint",
            "smoothFunction": "gaussian",
            "center": [0.5, 0.5, 0.5],  # center and size define a rectangular
            "amplitude": [0.0, 0.2, 0.0],
            "phase": 3.1415926,
            "thrustDirIdx": 0,
            "periodicity": 1.0,
            "eps": 0.1,
            "scale": 10.0,  # scale the source such the integral equals desired thrust
        },
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wallsbump"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wallsbump"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 1.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "primalMinResTol": 1e-16,
    "normalizeStates": {"U": 1.0, "p": 1.0, "nuTilda": 0.1, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-7, "FFD": 1e-3, "ACTP": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {"shapey": {"designVarType": "FFD"}, "actuator": {"actuatorName": "point2", "designVarType": "ACTP"}},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}

# DVGeo
FFDFile = "./FFD/bumpFFD.xyz"
DVGeo = DVGeometry(FFDFile)
DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")


def actuator(val, geo):
    actX = float(val[0])
    actY = float(val[1])
    actZ = float(val[2])
    actAx = float(val[3])
    actAy = float(val[4])
    actAz = float(val[5])
    actT = float(val[6])
    actPhase = float(val[7])
    actScale = float(val[8])
    DASolver.setOption(
        "fvSource",
        {
            "point2": {
                "type": "actuatorPoint",
                "smoothFunction": "gaussian",
                "center": [actX, actY, actZ],
                "amplitude": [actAx, actAy, actAz],
                "phase": actPhase,
                "thrustDirIdx": 0,
                "periodicity": actT,
                "eps": 0.1,
                "scale": actScale,
            },
        },
    )
    DASolver.updateDAOption()


iVol = 0
pts = DVGeo.getLocalIndex(iVol)
indexList = pts[:, 1, 1].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape
DVGeo.addGeoDVLocal("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
# actuator point parameter
DVGeo.addGeoDVGlobal(
    "actuator",
    value=[0.5, 0.5, 0.5, 0.0, 0.2, 0.0, 1.0, 3.1415926, 10.0],
    func=actuator,
    lower=-100.0,
    upper=100.0,
    scale=1.0,
)

# DAFoam
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.addFamilyGroup(DASolver.getOption("designSurfaceFamily"), DASolver.getOption("designSurfaces"))
DASolver.printFamilyList()
DASolver.setMesh(mesh)
# set evalFuncs
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# DVCon
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
[p0, v1, v2] = DASolver.getTriangulatedMeshSurface(groupName=DASolver.getOption("designSurfaceFamily"))
surf = [p0, v1, v2]
DVCon.setSurface(surf)

# optFuncs
def setObjFuncsUnsteady(DASolver, funcs, evalFuncs):
    nTimeInstances = DASolver.getOption("unsteadyAdjoint")["nTimeInstances"]
    for func in evalFuncs:
        avgObjVal = 0.0
        for i in range(nTimeInstances):
            avgObjVal += DASolver.getTimeInstanceObjFunc(i, func)
        funcs[func] = avgObjVal / nTimeInstances

    funcs["fail"] = False


def setObjFuncsSensUnsteady(CFDSolver, funcs, funcsSensAllTimeInstances, funcsSensCombined):

    nTimeInstances = 1.0 * len(funcsSensAllTimeInstances)
    for funcsSens in funcsSensAllTimeInstances:
        for objFunc in funcsSens:
            if objFunc != "fail":
                funcsSensCombined[objFunc] = {}
                for dv in funcsSens[objFunc]:
                    funcsSensCombined[objFunc][dv] = np.zeros_like(funcsSens[objFunc][dv], dtype="d")

    for funcsSens in funcsSensAllTimeInstances:
        for objFunc in funcsSens:
            if objFunc != "fail":
                for dv in funcsSens[objFunc]:
                    funcsSensCombined[objFunc][dv] += funcsSens[objFunc][dv] / nTimeInstances

    funcsSensCombined["fail"] = False

    if gcomm.rank == 0:
        print(funcsSensCombined)
    return


optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm
optFuncs.setObjFuncsUnsteady = setObjFuncsUnsteady
optFuncs.setObjFuncsSensUnsteady = setObjFuncsSensUnsteady

# Run
if calcFDSens == 1:
    optFuncs.calcFDSens(objFun=optFuncs.calcObjFuncValuesUnsteady, fileName="sensFD.txt")
else:
    DASolver.runColoring()
    xDV = DVGeo.getValues()
    funcs = {}
    funcs, fail = optFuncs.calcObjFuncValuesUnsteady(xDV)
    funcsSens = {}
    funcsSens, fail = optFuncs.calcObjFuncSensUnsteady(xDV, funcs)
    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-4, 1e-6)
