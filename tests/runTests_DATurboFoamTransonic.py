#!/usr/bin/env python
"""
Run Python tests for DATurboFoam with transonic conditions
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

checkRegVal = 1
if len(sys.argv) == 1:
    checkRegVal = 1
elif sys.argv[1] == "noCheckVal":
    checkRegVal = 0
else:
    print("sys.argv %s not valid!" % sys.argv[1])
    exit(1)

gcomm = MPI.COMM_WORLD

os.chdir("./input/CompressorFluid")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.compressible 0")
    os.system("cp -r 0/U.transonic 0/U")
    os.system("cp -r 0/p.transonic 0/p")
    os.system("cp -r constant/thermophysicalProperties.h constant/thermophysicalProperties")
    os.system("cp -r constant/MRFProperties.transonic constant/MRFProperties")
    os.system("cp -r system/fvSolution.transonic system/fvSolution")
    os.system("cp -r system/fvSchemes.transonic system/fvSchemes")

# test incompressible solvers
aeroOptions = {
    "solverName": "DATurboFoam",
    "flowCondition": "Compressible",
    "turbulenceModel": "SpalartAllmaras",
    "designSurfaceFamily": "designSurface",
    "designSurfaces": ["blade"],
    "primalMinResTol": 1e-12,
    "primalVarBounds": {
        "UUpperBound": 800.0,
        "ULowerBound": -800.0,
        "pUpperBound": 1000000.0,
        "pLowerBound": 20000.0,
        "hUpperBound": 500000.0,
        "hLowerBound": 50000.0,
        "rhoUpperBound": 10.0,
        "rhoLowerBound": 0.2,
    },
    "objFunc": {
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["blade"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0 / (0.5 * 10.0 * 10.0 * 1.0 * 1.0),
                "addToAdjoint": True,
            }
        },
        "TPR": {
            "part1": {
                "type": "totalPressureRatio",
                "source": "patchToFace",
                "patches": ["inlet", "outlet"],
                "inletPatches": ["inlet"],
                "outletPatches": ["outlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "TTR": {
            "part1": {
                "type": "totalTemperatureRatio",
                "source": "patchToFace",
                "patches": ["inlet", "outlet"],
                "inletPatches": ["inlet"],
                "outletPatches": ["outlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "MFR": {
            "part1": {
                "type": "massFlowRate",
                "source": "patchToFace",
                "patches": ["inlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"U": 100.0, "p": 100000.0, "nuTilda": 1e-3, "phi": 1.0, "T": 300.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "transonicPCOption": 1,
    # Design variable setup
    "designVar": {
        "shapey": {"designVarType": "FFD"},
        "shapez": {"designVarType": "FFD"}
    },
    "decomposeParDict": {"preservePatches": ["per1", "per2"]}
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "openfoam",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}

# DVGeo
FFDFile = "./FFD/localFFD.xyz"
DVGeo = DVGeometry(FFDFile)
# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[1:3, :, 1].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
DVGeo.addGeoDVLocal("shapez", lower=-1.0, upper=1.0, axis="z", scale=1.0, pointSelect=PS)

# DAFoam
DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
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
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# Run
DASolver.runColoring()
xDV = DVGeo.getValues()
funcs = {}
funcs, fail = optFuncs.calcObjFuncValues(xDV)
funcsSens = {}
funcsSens, fail = optFuncs.calcObjFuncSens(xDV, funcs)

if checkRegVal:
    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-6, 1e-8)


