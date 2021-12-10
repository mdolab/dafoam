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

U0 = 50.0
p0 = 101325.0
T0 = 300.0
A0 = 1.0
rho0 = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DARhoSimpleFoam",
    "designSurfaceFamily": "designSurface",
    "useAD": {"mode": "fd"},
    "designSurfaces": ["wallsbump"],
    "writeJacobians": ["all"],
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
            "source": "cylinderAnnulusToCell",
            "p1": [0.3, 0.5, 0.5],  # p1 and p2 define the axis and width
            "p2": [0.7, 0.5, 0.5],  # p2-p1 should be streamwise
            "innerRadius": 0.01,
            "outerRadius": 0.6,
            "rotDir": "left",
            "scale": 10.0,
            "POD": 0.7,
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
    },
    "adjStateOrdering": "cell",
    "debug": True,
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": 1e-4, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "shapey": {"designVarType": "FFD"},
        "uin": {"designVarType": "BC", "patches": ["inlet"], "variable": "U", "comp": 0},
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


def uin(val, geo):
    inletU = val[0]
    DASolver.setOption("primalBC", {"UIn": {"variable": "U", "patches": ["inlet"], "value": [inletU, 0.0, 0.0]}})
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[1:3, 1, 1:3].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
DVGeo.addGeoDVGlobal("uin", [U0], uin, lower=0.0, upper=100.0, scale=1.0)

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
    exit(0)
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


# *************************************************************
# Unit tests for functions that are not called in the above run
# *************************************************************

# test point2Vec functions
xvSize = len(DASolver.xv) * 3
xvVec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
xvVec.setSizes((xvSize, PETSc.DECIDE), bsize=1)
xvVec.setFromOptions()

DASolver.solver.ofMesh2PointVec(xvVec)
xvVecNorm = xvVec.norm(1)
DASolver.solver.pointVec2OFMesh(xvVec)
DASolver.solver.ofMesh2PointVec(xvVec)
xvVecNorm1 = xvVec.norm(1)

if xvVecNorm != xvVecNorm1:
    exit(1)

# test res2Vec functions
rSize = DASolver.solver.getNLocalAdjointStates()
rVec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
rVec.setSizes((rSize, PETSc.DECIDE), bsize=1)
rVec.setFromOptions()

DASolver.solver.ofResField2ResVec(rVec)
rVecNorm = rVec.norm(1)
DASolver.solver.resVec2OFResField(rVec)
DASolver.solver.ofResField2ResVec(rVec)
rVecNorm1 = rVec.norm(1)

if rVecNorm != rVecNorm1:
    exit(1)

# Test vector IO functions
DASolver.solver.writeVectorASCII(rVec, b"rVecRead")
DASolver.solver.writeVectorBinary(rVec, b"rVecRead")
rVecRead = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
rVecRead.setSizes((rSize, PETSc.DECIDE), bsize=1)
rVecRead.setFromOptions()
DASolver.solver.readVectorBinary(rVecRead, b"rVecRead")
rVecNormRead = rVecRead.norm(1)

if rVecNorm != rVecNormRead:
    exit(1)

# Test vector IO in pyDAFoam.py
DASolver.writePetscVecMat("rVecRead", rVec, mode="ASCII")
DASolver.writePetscVecMat("rVecRead", rVec, mode="Binary")
rVecRead = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
rVecRead.setSizes((rSize, PETSc.DECIDE), bsize=1)
rVecRead.setFromOptions()
DASolver.readPetscVecMat("rVecRead", rVec)
rVecNormRead = rVecRead.norm(1)

if rVecNorm != rVecNormRead:
    exit(1)

# Test matrix IO functions
lRow = gcomm.rank + 1
testMat = PETSc.Mat().create(PETSc.COMM_WORLD)
testMat.setSizes(((lRow, None), (None, gcomm.size)))
testMat.setFromOptions()
testMat.setPreallocationNNZ((gcomm.size, gcomm.size))
testMat.setUp()
Istart, Iend = testMat.getOwnershipRange()
for i in range(Istart, Iend):
    testMat[i, gcomm.rank] = gcomm.rank
testMat.assemblyBegin()
testMat.assemblyEnd()
DASolver.solver.writeMatrixASCII(testMat, b"testMat")
DASolver.solver.writeMatrixBinary(testMat, b"testMat")
testMatNorm = testMat.norm(0)

testMat1 = PETSc.Mat().create(PETSc.COMM_WORLD)
testMat1.setSizes(((lRow, None), (None, gcomm.size)))
testMat1.setFromOptions()
testMat1.setPreallocationNNZ((gcomm.size, gcomm.size))
testMat1.setUp()
DASolver.solver.readMatrixBinary(testMat1, b"testMat")
testMatNorm1 = testMat1.norm(0)

if testMatNorm != testMatNorm1:
    exit(1)

# call inferface functions
DVNames, DVSizes = DASolver.getDVsCons()
if DVNames[0] != "uin" or DVNames[1] != "shapey":
    exit(1)
if DVSizes[0] != 1 or DVSizes[1] != 4:
    exit(1)

states = DASolver.getStates()
statesMean = states.mean()
statesVec = DASolver.array2Vec(states)
states1 = DASolver.vec2Array(statesVec)
statesMean1 = states1.mean()
if statesMean != statesMean1:
    exit(1)

residuals = DASolver.getResiduals()
residualsMean = residuals.mean()

DASolver.setStates(states)

residuals1 = DASolver.getResiduals()
residualsMean1 = residuals1.mean()

if residualsMean != residualsMean1:
    exit(1)

# Additional test for a failed mesh
# perturb a large value for design variable to make a failed mesh
xDV["shapey"][0] = 1000.0
funcs1 = {}
funcs1, fail1 = optFuncs.calcObjFuncValues(xDV)

# the checkMesh utility should detect failed mesh
if fail1 is False:
    exit(1)
