#!/usr/bin/env python
"""
Run Python tests for DASimpleFoam
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

os.chdir("./reg_test_files-main/CompressorFluid")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r constant/MRFProperties.incompressible constant/MRFProperties")
    os.system("cp -r system/fvSolution.incompressible system/fvSolution")
    os.system("cp -r system/fvSchemes.incompressible system/fvSchemes")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["blade"],
    "useAD": {"mode": "fd"},
    "primalMinResTol": 1e-12,
    "objFunc": {
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["blade"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0 / (0.5 * 20.0 * 20.0 * 1.0 * 1.0),
                "addToAdjoint": True,
            }
        },
        "PWR": {
            "part1": {
                "type": "power",
                "source": "patchToFace",
                "patches": ["blade"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"U": 10, "p": 50.0, "nuTilda": 1e-3, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "dynAdjustTol": False},
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
    "fileType": "OpenFOAM",
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
DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
DVGeo.addLocalDV("shapez", lower=-1.0, upper=1.0, axis="z", scale=1.0, pointSelect=PS)

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
if calcFDSens == 1:
    optFuncs.calcFDSens(objFun=optFuncs.calcObjFuncValues, fileName="sensFD.txt")
else:
    DASolver.runColoring()
    xDV = DVGeo.getValues()
    funcs = {}
    funcs, fail = optFuncs.calcObjFuncValues(xDV)
    funcsSens = {}
    funcsSens, fail = optFuncs.calcObjFuncSens(xDV, funcs)
    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-4, 1e-6)
