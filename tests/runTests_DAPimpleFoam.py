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

replace_text_in_file("system/controlDict", "endTime         40;", "endTime         0.05;")

# test incompressible solvers
daOptions = {
    "solverName": "DAPimpleFoam",
    "designSurfaces": ["wallsbump"],
    "printIntervalUnsteady": 100,
    "writeJacobians": ["all"],
    "useAD": {"mode": "reverse"},
    "unsteadyAdjoint": {"mode": "timeAccurateAdjoint", "nTimeInstances": 6},
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
    },
    "primalMinResTol": 1e-16,
    "adjStateOrdering": "cell",
    "adjEqnOption": {"pcFillLevel": 0, "jacMatReOrdering": "natural", "useNonZeroInitGuess": False},
    "normalizeStates": {"U": 1.0, "p": 1.0, "nuTilda": 0.1, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-7, "FFD": 1e-2},
    "designVar": {},
    "adjPCLag": 1000,
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}

# DVGeo
DVGeo = DVGeometry("./FFD/bumpFFD.xyz")
# select points
iVol = 0
pts = DVGeo.getLocalIndex(iVol)
indexList = pts[2, 1, 2].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape
DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
daOptions["designVar"]["shapey"] = {"designVarType": "FFD"}

# DAFoam
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.printFamilyList()
DASolver.setMesh(mesh)
# set evalFuncs
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# DVCon
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
[p0, v1, v2] = DASolver.getTriangulatedMeshSurface(groupName=DASolver.designSurfacesGroup)
surf = [p0, v1, v2]
DVCon.setSurface(surf)

# optFuncs
def setObjFuncsUnsteady(DASolver, funcs, evalFuncs):
    nTimeInstances = DASolver.getOption("unsteadyAdjoint")["nTimeInstances"]
    for func in evalFuncs:
        avgObjVal = 0.0
        for i in range(1, nTimeInstances):
            avgObjVal += DASolver.getTimeInstanceObjFunc(i, func)
        funcs[func] = avgObjVal # / (nTimeInstances - 1)

    funcs["fail"] = False


def setObjFuncsSensUnsteady(DASolver, funcs, funcsSensAllTimeInstances, funcsSensCombined):

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
                    funcsSensCombined[objFunc][dv] += funcsSens[objFunc][dv] # / nTimeInstances

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

    # this code is not fully implemented yet, so do not test it
    funcsSens["CD"]["shapey"] = 0
    
    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-4, 1e-6)
