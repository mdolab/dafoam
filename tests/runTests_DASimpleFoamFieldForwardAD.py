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
import numpy as np
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./input/NACA0012BetaSA")

replace_text_in_file("constant/turbulenceProperties", "RASModel             SpalartAllmarasFv3Beta;", "RASModel             SpalartAllmarasFv3FieldInversion;")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    # rename reference U field 
    os.system("cp 0/varRefFieldInversion 0/UData")

replace_text_in_file("0/varRefFieldInversion", "    object      varRefFieldInversion;", "    object      UData;")

U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
A0 = 0.1
alpha0 = 5.139186
LRef = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["wing"],
    "primalMinResTol": 1e-12,
    "useAD": {"mode": "forward", "dvName": "beta", "seedIndex": 0},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
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
                "addToAdjoint": False,
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
                "addToAdjoint": False,
            }
        },
        "FI": {
            "part1": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "data": "UData",
                "scale": 1.0,
                "addToAdjoint": True,
                "weightedSum": False,
            },
            "part2": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "data": "beta",
                "scale": 0.01,
                "addToAdjoint": True,
                "weightedSum": False,
            },
        },
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["wing"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.25, 0.0, 0.05],
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * 1.0),
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "beta": {"designVarType": "Field", "fieldName": "betaFieldInversion", "fieldType": "scalar"},
        "alphaPorosity": {"designVarType": "Field", "fieldName": "alphaPorosity", "fieldType": "scalar"},
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


def betaFieldInversion(val, geo):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"betaFieldInversion", v, idxI)
        DASolver.updateBoundaryConditions(b"betaFieldInversion", b"scalar")


def alphaPorosity(val, geo):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"alphaPorosity", v, idxI)
        DASolver.updateBoundaryConditions(b"alphaPorosity", b"scalar")


# select points
DVGeo.addGlobalDV("alpha", [alpha0], alpha, lower=-10.0, upper=10.0, scale=1.0)
nCells = 4032
beta0 = np.ones(nCells, dtype="d")
DVGeo.addGlobalDV("beta", value=beta0, func=betaFieldInversion, lower=1e-5, upper=10.0, scale=1.0)

alphaPorosity0 = np.zeros(nCells, dtype="d")
DVGeo.addGlobalDV("alphaPorosity", value=alphaPorosity0, func=alphaPorosity, lower=0, upper=100.0, scale=1.0)

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

funcsSens = {}
funcsSens["CD"] = {}
funcsSens["CL"] = {}
funcsSens["CMZ"] = {}

# Run
# beta0
DASolver()
funcsSens["CD"]["beta"] = [] 
funcsSens["CL"]["beta"] = []
funcsSens["CMZ"]["beta"] = []
funcsSens["CD"]["beta"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["beta"].append(DASolver.getForwardADDerivVal("CL"))
funcsSens["CMZ"]["beta"].append(DASolver.getForwardADDerivVal("CMZ"))
# beta100
DASolver.setOption("useAD", {"dvName": "beta", "seedIndex": 100})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["beta"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["beta"].append(DASolver.getForwardADDerivVal("CL"))
funcsSens["CMZ"]["beta"].append(DASolver.getForwardADDerivVal("CMZ"))
# beta1000
DASolver.setOption("useAD", {"dvName": "beta", "seedIndex": 1000})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["beta"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["beta"].append(DASolver.getForwardADDerivVal("CL"))
funcsSens["CMZ"]["beta"].append(DASolver.getForwardADDerivVal("CMZ"))

# alphaPorosity0
DASolver.setOption("useAD", {"dvName": "alphaPorosity", "seedIndex": 0})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["alphaPorosity"] = [] 
funcsSens["CL"]["alphaPorosity"] = []
funcsSens["CMZ"]["alphaPorosity"] = []
funcsSens["CD"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CL"))
funcsSens["CMZ"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CMZ"))
# alphaPorosity100
DASolver.setOption("useAD", {"dvName": "alphaPorosity", "seedIndex": 100})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CL"))
funcsSens["CMZ"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CMZ"))
# alphaPorosity1000
DASolver.setOption("useAD", {"dvName": "alphaPorosity", "seedIndex": 1000})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CL"))
funcsSens["CMZ"]["alphaPorosity"].append(DASolver.getForwardADDerivVal("CMZ"))

if gcomm.rank == 0:
    reg_write_dict(funcsSens, 1e-5, 1e-7)
