#!/usr/bin/env python
"""
Run Python tests for DASimpleFoamFieldInversionObjectiveFunctions
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

os.chdir("./input/NACA0012FieldInversion")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    # create dummy reference fields for field inversion
    os.system("cp 0/p 0/surfacePressure")
    os.system("cp 0/p 0/surfacePressureRef")
    os.system("cp 0/p 0/surfaceFriction")
    os.system("cp 0/p 0/surfaceFrictionRef")
    os.system("cp 0/p 0/profileRefFieldInversion")

    
replace_text_in_file("0/surfacePressure", "    object      p;", "    object      surfacePressure;")
replace_text_in_file("0/surfacePressureRef", "    object      p;", "    object      surfacePressureRef;")
replace_text_in_file("0/surfaceFriction", "    object      p;", "    object      surfaceFriction;")
replace_text_in_file("0/surfaceFrictionRef", "    object      p;", "    object      surfaceFrictionRef;")
replace_text_in_file("0/profileRefFieldInversion", "    object      p;", "    object      profileRefFieldInversion;")

U0 = 10.0
p0 = 0.0
rho0 = 1.0
nuTilda0 = 4.5e-5
A0 = 0.1
alpha0 = 5.139186
LRef = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "useAD": {"mode": "reverse"},
    "designSurfaces": ["wing"],
    "primalMinResTol": 1e-12,
    "writeJacobians": ["all"],
    "writeSensMap": ["betaSA", "alphaPorosity"],
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "FI": {
            "varRefFieldInversion": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "varTypeFieldInversion": "volume",
                "stateName": "U",
                "stateRefName": "varRefFieldInversion",
                "stateType": "vector",
                "scale": 1.0,
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1.0,
            },
            "profileRefFieldInversion":
            {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "varTypeFieldInversion": "profile",
                "stateName": "U",
                "stateRefName": "profileRefFieldInversion", # dummy 
                "stateType": "scalar",
                "scale": 1.0,
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1.0,
            },
            "surfacePressure": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "varTypeFieldInversion": "surface",
                "stateType": "surfacePressure",
                "stateName": "surfacePressure",
                "stateRefName": "surfacePressureRef", # dummy
                "patchNames": ["wing"],
                "pRef": 0,
                "scale": 1.0 / (rho0 * U0 * U0),
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1.0,
            },
            "surfaceFriction": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "varTypeFieldInversion": "surface",
                "stateType": "surfaceFriction",
                "stateName": "surfaceFriction",
                "stateRefName": "surfaceFrictionRef", # dummy
                "patchNames": ["wing"],
                "scale": 1.0 / (rho0 * U0 * U0),
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1.0,
            },
            "wallShearStress": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "varTypeFieldInversion": "surface",
                "stateType": "wallShearStress",
                "stateName": "surfaceFriction",
                "stateRefName": "surfaceFrictionRef", # dummy
                "patchNames": ["wing"],
                "scale": 1.0 / (rho0 * U0 * U0),
                "wssDir": [-1.0, 0.0, 0.0],
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1.0,
            },
            "aeroCoeff":{
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "varTypeFieldInversion": "surface",
                "stateType": "aeroCoeff",
                "stateName": "aeroCoeff",
                "stateRefName": "aeroCoeff",
                "aeroCoeffRef": 1, # dummy
                "direction": [1.0, 0.0, 0.0], 
                "patchNames": ["wing"],
                "scale": 1.0 / (rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1.0,
            },
            "beta": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "varTypeFieldInversion": "volume",
                "stateName": "betaFieldInversion",
                "stateRefName": "betaRefFieldInversion",
                "stateType": "scalar",
                "scale": 0.01,
                "addToAdjoint": True,
                "weightedSum": False,
            },
        },
    },
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "beta": {"designVarType": "Field", "fieldName": "betaFieldInversion", "fieldType": "scalar"},
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}

# DVGeo
FFDFile = "./FFD/wingFFD.xyz"
DVGeo = DVGeometry(FFDFile)

# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")


def betaFieldInversion(val, geo):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"betaFieldInversion", v, idxI)
        DASolver.updateBoundaryConditions(b"betaFieldInversion", b"scalar")

# select points
nCells = 4032
beta0 = np.ones(nCells, dtype="d")
DVGeo.addGlobalDV("beta", value=beta0, func=betaFieldInversion, lower=1e-5, upper=10.0, scale=1.0)


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
if calcFDSens == 1:
    optFuncs.calcFDSens(objFun=optFuncs.calcObjFuncValues, fileName="sensFD.txt")
else:
    DASolver.runColoring()
    xDV = DVGeo.getValues()
    funcs = {}
    funcs, fail = optFuncs.calcObjFuncValues(xDV)
    funcsSens = {}
    funcsSens, fail = optFuncs.calcObjFuncSens(xDV, funcs)
    # we dont want to save all 4K sens, therefore, we compute
    # the L2 norm of the sens and save it to disk as ref
    betaSens = funcsSens["FI"]["beta"]
    funcsSens["FI"]["beta"] = np.zeros(1, "d")
    funcsSens["FI"]["beta"][0] = np.linalg.norm(betaSens)


    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-4, 1e-6)