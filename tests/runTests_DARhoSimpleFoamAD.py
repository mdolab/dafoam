#!/usr/bin/env python
"""
Run Python tests for DARhoSimpleFoam
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

calcFDSens = 0
if len(sys.argv) != 1:
    if sys.argv[1] == "calcFDSens":
        calcFDSens = 1

gcomm = MPI.COMM_WORLD

os.chdir("./input/CurvedCubeSnappyHexMesh")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.compressible 0")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

replace_text_in_file("system/fvSchemes", "meshWave", "meshWaveFrozen")

U0 = 50.0
p0 = 101325.0
T0 = 300.0
A0 = 1.0
rho0 = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DARhoSimpleFoam",
    "designSurfaces": ["wallsbump"],
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1e-12,
    "primalBC": {
        "UIn": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [T0]},
        "useWallFunction": False,
    },
    "primalVarBounds": {
        "UMax": 1000.0,
        "UMin": -1000.0,
        "pMax": 500000.0,
        "pMin": 20000.0,
        "eMax": 500000.0,
        "eMin": 100000.0,
        "rhoMax": 5.0,
        "rhoMin": 0.2,
    },
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusSmooth",
            "center": [0.5, 0.5, 0.5],
            "direction": [1.0, 0.0, 0.0],
            "innerRadius": 0.05,
            "outerRadius": 0.6,
            "rotDir": "right",
            "scale": 10.0,
            "POD": 0.7,
            "eps": 0.1,
            "expM": 1.0,
            "expN": 0.5,
            "adjustThrust": 1,
            "targetThrust": 1.2,
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
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            },
            "part2": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["walls", "frontandback"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            },
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["walls", "frontandback", "wallsbump"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 1.0, 0.0],
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["walls", "frontandback", "wallsbump"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.5, 0.5, 0.5],
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * 1.0),
                "addToAdjoint": True,
            }
        },
        "VOL": {
            "part1": {
                "type": "variableVolSum",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "varName": "U",
                "varType": "vector",
                "component": 0,
                "isSquare": 1,
                "scale": 1.0,
                "addToAdjoint": True,
            },
            "part2": {
                "type": "variableVolSum",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "varName": "p",
                "varType": "scalar",
                "component": 0,
                "isSquare": 0,
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
        "VOL1": {
            "part1": {
                "type": "variableVolSum",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "varName": "fvSource",
                "varType": "vector",
                "component": 0,
                "isSquare": 0,
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
    },
    "adjStateOrdering": "cell",
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": 1e-4, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "shapey": {"designVarType": "FFD"},
        "uin": {"designVarType": "BC", "patches": ["inlet"], "variable": "U", "comp": 0},
        "actuator": {"actuatorName": "disk1", "designVarType": "ACTD"},
    },
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
DVGeo.addRefAxis("dummyAxis", xFraction=0.25, alignIndex="k")


def uin(val, geo):
    inletU = float(val[0])
    DASolver.setOption("primalBC", {"UIn": {"variable": "U", "patches": ["inlet"], "value": [inletU, 0.0, 0.0]}})
    DASolver.updateDAOption()


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
    T = float(val[9])
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
                "targetThrust": T,
            },
        },
    )
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[1:3, 1, 1:3].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
DVGeo.addGlobalDV("uin", [U0], uin, lower=0.0, upper=100.0, scale=1.0)
# actuator
DVGeo.addGlobalDV(
    "actuator",
    value=[0.5, 0.5, 0.5, 0.05, 0.6, 10.0, 0.7, 1.0, 0.5, 1.2],
    func=actuator,
    lower=-100.0,
    upper=100.0,
    scale=1.0,
)

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
    exit(0)
else:
    DASolver.runColoring()
    xDV = DVGeo.getValues()
    funcs = {}
    funcs, fail = optFuncs.calcObjFuncValues(xDV)
    funcsSens = {}
    funcsSens, fail = optFuncs.calcObjFuncSens(xDV, funcs)
    psiVec = DASolver.wVec.duplicate()
    psiVec.set(1.0)
    prodVec = DASolver.wVec.duplicate()
    DASolver.solverAD.calcdRdWTPsiAD(DASolver.xvVec, DASolver.wVec, psiVec, prodVec)
    if abs(prodVec.norm() - 4351730273.14912) / 4351730273.14912 > 1e-3:
        print("prodVec.norm() failed!", prodVec.norm())
        exit(1)
    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-5, 1e-7)
    
    # just run runLowOrderPrimal4PC
    DASolver.setOption("runLowOrderPrimal4PC", {"active": True})
    optFuncs.runLowOrderPrimal4PC()
