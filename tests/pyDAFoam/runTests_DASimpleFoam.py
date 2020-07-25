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

checkRegVal = 1
if len(sys.argv) == 1:
    checkRegVal = 1
elif sys.argv[1] == "noCheckVal":
    checkRegVal = 0
else:
    print("sys.argv %s not valid!" % sys.argv[1])
    exit(1)

gcomm = MPI.COMM_WORLD

os.chdir("../input/NACA0012")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")

UmagIn = 10.0
pIn = 0.0
nuTildaIn = 4.5e-5
ARef = 0.1
alpha0 = 5.0
LRef = 1.0
CL_target = 0.5
CM_target = 0.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "flowCondition": "Incompressible",
    "turbulenceModel": "SpalartAllmaras",
    "designSurfaceFamily": "designSurface",
    "designSurfaces": ["wing"],
    "primalMinResTol": 1e-12,
    "primalBC": {
        "UIn": {"variable": "U", "patch": "inout", "value": [UmagIn, 0.0, 0.0]},
        "pIn": {"variable": "p", "patch": "inout", "value": [pIn]},
        "nuTildaIn": {"variable": "nuTilda", "patch": "inout", "value": [nuTildaIn], "useWallFunction": True},
    },
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusToCell",
            "p1": [-0.4, -0.1, 0.05],  # p1 and p2 define the axis and width
            "p2": [-0.1, -0.1, 0.05],  # p2-p1 should be streamwise
            "innerRadius": 0.01,
            "outerRadius": 0.5,
            "rotDir": "left",
            "scale": 50.0,
            "POD": 0.7,
        },
        "disk2": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusToCell",
            "p1": [-0.4, 0.1, 0.05],
            "p2": [-0.1, 0.1, 0.05],
            "innerRadius": 0.01,
            "outerRadius": 0.5,
            "rotDir": "right",
            "scale": 25.0,
            "POD": 1.0,
        },
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "alpha",
                "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "alphaName": "alpha",
                "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef),
                "addToAdjoint": True,
            }
        },
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["wing"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.25, 0.0, 0.05],
                "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef * LRef),
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"U": UmagIn, "p": UmagIn * UmagIn / 2.0, "nuTilda": nuTildaIn * 10.0, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "shapey": {"designVarType": "FFD"},
        "alpha": {"designVarType": "AOA", "patch": "inout", "flowAxis": "x", "normalAxis": "y"},
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "openfoam",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}

optOptions = {
    "ACC": 1.0e-5,  # convergence accuracy
    "MAXIT": 1,  # max optimization iterations
    "IFILE": "opt_SLSQP.out",
}

# DVGeo
FFDFile = "./FFD/wingFFD.xyz"
DVGeo = DVGeometry(FFDFile)

# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")


def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(UmagIn * np.cos(aoa)), float(UmagIn * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"UIn": {"variable": "U", "patch": "inout", "value": inletU}})
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[:, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
DVGeo.addGeoDVGlobal("alpha", [alpha0], alpha, lower=-10.0, upper=10.0, scale=1.0)

# DAFoam
DASolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.addFamilyGroup(DASolver.getOption("designSurfaceFamily"), DASolver.getOption("designSurfaces"))
DASolver.printFamilyList()
DASolver.setMesh(mesh)
# set evalFuncs
evalFuncs = []
objFuncs = DASolver.getOption("objFunc")
for funcName in objFuncs:
    for funcPart in objFuncs[funcName]:
        if objFuncs[funcName][funcPart]["addToAdjoint"] is True:
            if funcName not in evalFuncs:
                evalFuncs.append(funcName)

# DVCon
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
[p0, v1, v2] = DASolver.getTriangulatedMeshSurface(groupName=DASolver.getOption("designSurfaceFamily"))
surf = [p0, v1, v2]
DVCon.setSurface(surf)

leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]

DVCon.addVolumeConstraint(leList, teList, nSpan=2, nChord=10, lower=1.0, upper=3, scaled=True)
DVCon.addThicknessConstraints2D(leList, teList, nSpan=2, nChord=10, lower=0.8, upper=3.0, scaled=True)

# Create a linear constraint so that the curvature at the symmetry plane is zero
nFFDs_x = 5
pts1 = DVGeo.getLocalIndex(0)
indSetA = []
indSetB = []
for i in range(nFFDs_x):
    for j in [0, 1]:
        indSetA.append(pts1[i, j, 1])
        indSetB.append(pts1[i, j, 0])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0)

# Create a linear constraint so that the leading and trailing edges do not change
pts1 = DVGeo.getLocalIndex(0)
indSetA = []
indSetB = []
for i in [0, nFFDs_x - 1]:
    for k in [0]:  # do not constrain k=1 because it is linked in the above symmetry constraint
        indSetA.append(pts1[i, 0, k])
        indSetB.append(pts1[i, 1, k])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=1.0, lower=0.0, upper=0.0)

# optFuncs
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# Optimize
DASolver.runColoring()
optProb = Optimization("opt", optFuncs.calcObjFuncValues, comm=gcomm)
DVGeo.addVariablesPyOpt(optProb)
DVCon.addConstraintsPyOpt(optProb)
# Add objective
optProb.addObj("CD", scale=1)
# Add physical constraints
optProb.addCon("CL", lower=CL_target, upper=CL_target, scale=1)
optProb.addCon("CMZ", lower=CM_target, upper=CM_target, scale=1)

if gcomm.rank == 0:
    print(optProb)

opt = OPT("slsqp", options=optOptions)
histFile = os.path.join("./", "slsqp_hist.hst")
sol = opt(optProb, sens=optFuncs.calcObjFuncSens, storeHistory=histFile)

if gcomm.rank == 0:
    print(sol)

if checkRegVal:
    xDVs = DVGeo.getValues()
    if gcomm.rank == 0:
        reg_write_dict(xDVs, 1e-6, 1e-8)
