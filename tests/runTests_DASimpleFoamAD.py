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

os.chdir("./input/NACA0012")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.safv3 constant/turbulenceProperties")

U0 = 10.0
p0 = 0.0
k0 = 0.18
omega0 = 1225.0
A0 = 0.1
alpha0 = 5.0
LRef = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["wing"],
    "useAD": {"mode": "reverse"},
    "writeJacobians": ["all"],
    "writeSensMap": ["shapey"],
    "primalMinResTol": 1e-12,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "k0": {"variable": "k", "patches": ["inout"], "value": [k0]},
        "omega0": {"variable": "omega", "patches": ["inout"], "value": [omega0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "alpha",
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
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
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "COP": {
            "part1": {
                "type": "centerOfPressure",
                "source": "patchToFace",
                "patches": ["wing"],
                "axis": [1.0, 0.0, 0.0],
                "forceAxis": [0.0, 1.0, 0.0],
                "center": [0, 0, 0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "k": k0, "omega": omega0, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3, "ACTD": 1.0e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "shapey": {"designVarType": "FFD"},
        "alpha": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
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


def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[1:4, 1, 0].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
DVGeo.addGlobalDV("alpha", [alpha0], alpha, lower=-10.0, upper=10.0, scale=1.0)

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
    alphaSet = optFuncs.solveCL(0.5, "alpha", "CL", tol=1e-2)
    if abs(alphaSet - 5.139885) > 1e-3:
        exit(1)
    alpha([alpha0], None)
    xDVs = DVGeo.getValues()
    xDVs["alpha"][0] = alpha0
    DVGeo.setDesignVars(xDVs)
    funcs = {}
    funcs, fail = optFuncs.runPrimal()
    funcsSens = {}
    funcsSens, fail = optFuncs.runAdjoint(fileName="totalSens.txt")
    optFuncs.calcFDSens(fileName="totalSensFD.txt")

    # Force calculation routines
    # Compute force
    forces = DASolver.getForces()
    fNorm = np.linalg.norm(forces.flatten())
    fNormSum = gcomm.allreduce(fNorm, op=MPI.SUM)
    funcs["forces"] = fNormSum

    # Compute dForcedxV
    fBar = np.ones(np.size(forces.flatten()))
    fBarVec = DASolver.array2Vec(fBar)
    dForcedXv = DASolver.xvVec.duplicate()
    dForcedXv.zeroEntries()
    DASolver.solverAD.calcdForcedXvAD(DASolver.xvVec, DASolver.wVec, fBarVec, dForcedXv)
    xVBar = DASolver.vec2Array(dForcedXv)
    xVBarNorm = np.linalg.norm(xVBar.flatten())
    xVBarNormSum = gcomm.allreduce(xVBarNorm, op=MPI.SUM)
    funcsSens["dForcedxV"] = xVBarNormSum

    # Compute dForcedW
    fBar = np.ones(np.size(forces.flatten()))
    fBarVec = DASolver.array2Vec(fBar)
    dForcedW = DASolver.wVec.duplicate()
    dForcedW.zeroEntries()
    DASolver.solverAD.calcdForcedWAD(DASolver.xvVec, DASolver.wVec, fBarVec, dForcedW)
    wBar = DASolver.vec2Array(dForcedW)
    wBarNorm = np.linalg.norm(wBar.flatten())
    wBarNormSum = gcomm.allreduce(wBarNorm, op=MPI.SUM)
    funcsSens["dForcedW"] = wBarNormSum

    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-5, 1e-7)

        f = open("totalSensFD.txt")
        lines = f.readlines()
        f.close()
        line4 = float(lines[4])
        if abs(line4 - 0.002349192076814971) / line4 > 1e-3:
            exit(1)
