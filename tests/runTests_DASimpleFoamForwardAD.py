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

os.chdir("./input/NACA0012")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
A0 = 0.1
alpha0 = 3.0
LRef = 1.0

# test incompressible solvers
daOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["wing"],
    "primalMinResTol": 1e-12,
    "useAD": {"mode": "forward", "dvName": "shape", "seedIndex": 0},
    "primalBC": {
        "UIn": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": False,
    },
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusSmooth",
            "center": [-0.5, 0.0, 0.05],
            "direction": [1.0, 0.0, 0.0],
            "innerRadius": 0.01,
            "outerRadius": 0.4,
            "rotDir": "right",
            "scale": 10.0,
            "POD": 0.8,
            "eps": 0.1,
            "expM": 1.0,
            "expN": 0.5,
            "adjustThrust": 1,
            "targetThrust": 0.2,
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
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "designVar": {},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}

# DVGeo
DVGeo = DVGeometry("./FFD/wingFFD.xyz")
# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")


def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


def pitch(val, geo):
    for i in range(nTwists):
        geo.rot_z["bodyAxis"].coef[i] = -val[0]


def actuator(val, geo):
    actX = float(val[0])
    actY = float(val[1])
    actZ = float(val[2])
    actR1 = float(val[3])
    actR2 = float(val[4])
    actScale = float(val[5])
    actPOD = float(val[6])
    actExpM = float(val[7])
    actExpN = float(val[8])
    DASolver.setOption(
        "fvSource",
        {
            "disk1": {
                "type": "actuatorDisk",
                "source": "cylinderAnnulusSmooth",
                "center": [actX, actY, actZ],
                "direction": [1.0, 0.0, 0.0],
                "innerRadius": actR1,
                "outerRadius": actR2,
                "rotDir": "right",
                "scale": actScale,
                "POD": actPOD,
                "eps": 0.1,  # eps should be of cell size
                "expM": actExpM,
                "expN": actExpN,
                "adjustThrust": 1,
                "targetThrust": 0.2,
            },
        },
    )
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[1:4, 1, 0].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape
DVGeo.addGeoDVLocal("shape", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
daOptions["designVar"]["shape"] = {"designVarType": "FFD"}
# pitch
DVGeo.addGeoDVGlobal("pitch", np.zeros(1), pitch, lower=-10.0, upper=10.0, scale=1.0)
daOptions["designVar"]["pitch"] = {"designVarType": "FFD"}
# AOA
DVGeo.addGeoDVGlobal("alpha", value=[alpha0], func=alpha, lower=0.0, upper=10.0, scale=1.0)
daOptions["designVar"]["alpha"] = {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"}
# Actuator
DVGeo.addGeoDVGlobal(
    "actuator",
    value=[-0.5, 0.0, 0.05, 0.01, 0.4, 10.0, 0.8, 1.0, 0.5],
    func=actuator,
    lower=-100.0,
    upper=100.0,
    scale=1.0,
)
daOptions["designVar"]["actuator"] = {"actuatorName": "disk1", "designVarType": "ACTD"}
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
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

funcsSens = {}
funcsSens["CD"] = {}
funcsSens["CL"] = {}

# Run
# shape
DASolver()
funcsSens["CD"]["shape"] = DASolver.getForwardADDerivVal("CD")
funcsSens["CL"]["shape"] = DASolver.getForwardADDerivVal("CL")
# pitch
DASolver.setOption("useAD", {"dvName": "pitch"})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["pitch"] = DASolver.getForwardADDerivVal("CD")
funcsSens["CL"]["pitch"] = DASolver.getForwardADDerivVal("CL")
# aoa
DASolver.setOption("useAD", {"dvName": "alpha"})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["alpha"] = DASolver.getForwardADDerivVal("CD")
funcsSens["CL"]["alpha"] = DASolver.getForwardADDerivVal("CL")
# ACTD
funcsSens["CD"]["actuator"] = [] 
funcsSens["CL"]["actuator"] = []
# 0th
DASolver.setOption("useAD", {"dvName": "actuator", "seedIndex": 0})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["actuator"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["actuator"].append(DASolver.getForwardADDerivVal("CL"))
# 3th
DASolver.setOption("useAD", {"dvName": "actuator", "seedIndex": 3})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["actuator"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["actuator"].append(DASolver.getForwardADDerivVal("CL"))
# 5th
DASolver.setOption("useAD", {"dvName": "actuator", "seedIndex": 5})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["actuator"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["actuator"].append(DASolver.getForwardADDerivVal("CL"))
# 7th
DASolver.setOption("useAD", {"dvName": "actuator", "seedIndex": 7})
DASolver.updateDAOption()
DASolver()
funcsSens["CD"]["actuator"].append(DASolver.getForwardADDerivVal("CD"))
funcsSens["CL"]["actuator"].append(DASolver.getForwardADDerivVal("CL"))

if gcomm.rank == 0:
    reg_write_dict(funcsSens, 1e-5, 1e-7)
