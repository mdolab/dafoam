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

os.chdir("./input/pitzDailyScalarTransport")

TRef = 1.0

# test incompressible solvers
aeroOptions = {
    "designSurfaces": ["upperWall"],
    "solverName": "DAScalarTransportFoam",
    "printIntervalUnsteady": 1,
    "primalBC": {"T0": {"variable": "T", "patches": ["inlet"], "value": [TRef]}},
    "unsteadyAdjoint": {"mode": "timeAccurate"},
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
                "divByTotalVol": 0,
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "debug": False,
    "primalMinResTol": 1e-16,
    # "adjStateOrdering": "cell",
    "adjEqnOption": {"pcFillLevel": 0, "jacMatReOrdering": "natural", "useNonZeroInitGuess": False},
    "normalizeStates": {"U": 1.0, "p": 1.0, "nuTilda": 0.1, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-7, "FFD": 1e-2},
    "designVar": {},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, -0.0005]], [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0005]]],
}

# DVGeo
DVGeo = DVGeometry("./FFD/bumpFFD.xyz")
# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")
# select points
iVol = 0
pts = DVGeo.getLocalIndex(iVol)
indexList = pts[3, 0, 0].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape
# DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
# aeroOptions["designVar"]["shapey"] = {"designVarType": "FFD"}


def tin(val, geo):
    inletT = float(val[0])
    DASolver.setOption("primalBC", {"T0": {"variable": "T", "patches": ["inlet"], "value": [inletT]}})
    DASolver.updateDAOption()


DVGeo.addGlobalDV("tbc", [TRef], tin, lower=0.0, upper=100.0, scale=1.0)
aeroOptions["designVar"]["tbc"] = {"designVarType": "BC", "patches": ["inlet"], "variable": "T", "comp": 0}

# DAFoam
DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
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
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm
# Run
DASolver()
funcs = {}
evalFuncs = ["TVOL"]
DASolver.evalFunctions(funcs, evalFuncs)

if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-8, 1e-10)
