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

checkRegVal = 1
if len(sys.argv) == 1:
    checkRegVal = 1
elif sys.argv[1] == "noCheckVal":
    checkRegVal = 0
else:
    print("sys.argv %s not valid!" % sys.argv[1])
    exit(1)

gcomm = MPI.COMM_WORLD

os.chdir("./input/CurvedCubeSnappyHexMesh")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.compressible 0")

UmagIn = 50.0
pIn = 101325.0
nuTildaIn = 4.5e-5
TIn = 300.0
ARef = 1.0
rhoRef = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DARhoSimpleFoam",
    "flowCondition": "Compressible",
    "turbulenceModel": "SpalartAllmaras",
    "designSurfaceFamily": "designSurface",
    "designSurfaces": ["wallsbump"],
    "primalMinResTol": 1e-12,
    "multiPoint": True,
    "nMultiPoints": 1,
    "primalBC": {
        "UIn": {"variable": "U", "patch": "inlet", "value": [UmagIn, 0.0, 0.0]},
        "pIn": {"variable": "p", "patch": "outlet", "value": [pIn]},
        "TIn": {"variable": "T", "patch": "inlet", "value": [TIn]},
        "nuTildaIn": {"variable": "nuTilda", "patch": "inlet", "value": [nuTildaIn]},
        "useWallFunction": False,
    },
    "primalVarBounds": {
        "UUpperBound": 1000.0,
        "ULowerBound": -1000.0,
        "pUpperBound": 500000.0,
        "pLowerBound": 20000.0,
        "eUpperBound": 500000.0,
        "eLowerBound": 100000.0,
        "rhoUpperBound": 5.0,
        "rhoLowerBound": 0.2,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wallsbump"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0 / (0.5 * rhoRef * UmagIn * UmagIn * ARef),
                "addToAdjoint": True,
            },
            "part2": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["walls", "frontandback"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0 / (0.5 * rhoRef * UmagIn * UmagIn * ARef),
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
                "scale": 1.0 / (0.5 * rhoRef * UmagIn * UmagIn * ARef),
                "addToAdjoint": True,
            }
        },
    },
    "adjStateOrdering": "cell",
    "debug": True,
    "normalizeStates": {"U": UmagIn, "p": pIn, "nuTilda": nuTildaIn * 10.0, "phi": 1.0, "T": TIn},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "gmresAbsTol": 1.0e-15, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "shapey": {"designVarType": "FFD"},
        "uin": {"designVarType": "BC", "patch": "inlet", "variable": "U", "comp": 0},
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "openfoam",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}

# DVGeo
FFDFile = "./FFD/bumpFFD.xyz"
DVGeo = DVGeometry(FFDFile)


def uin(val, geo):
    inletU = val[0]
    DASolver.setOption("primalBC", {"UIn": {"variable": "U", "patch": "inlet", "value": [inletU, 0.0, 0.0]}})
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[1:3, 1, 1:3].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
DVGeo.addGeoDVGlobal("uin", [UmagIn], uin, lower=0.0, upper=100.0, scale=1.0)

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

# optFuncs
# provide a function to set primal conditions
def setMultiPointCondition(xDV, index):
    pass


# provide a function to assemble the funcs from MP
def setMultiPointObjFuncs(funcs, funcsMP, index):
    for key in funcs:
        funcsMP[key] = funcs[key]
    return


# provide a function to assemble the funcs from MP
def setMultiPointObjFuncsSens(xDVs, funcsMP, funcsSens, funcsSensMP, index):
    for key in funcsSens:
        funcsSensMP[key] = funcsSens[key]
    return


optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm
optFuncs.setMultiPointCondition = setMultiPointCondition
optFuncs.setMultiPointObjFuncs = setMultiPointObjFuncs
optFuncs.setMultiPointObjFuncsSens = setMultiPointObjFuncsSens

# Run
DASolver.runColoring()
xDV = DVGeo.getValues()
funcs = {}
funcs, fail = optFuncs.calcObjFuncValuesMP(xDV)
funcsSens = {}
funcsSens, fail = optFuncs.calcObjFuncSensMP(xDV, funcs)

if checkRegVal:
    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-6, 1e-8)


# *************************************************************
# Unit tests for functions that are not called in the above run
# *************************************************************

# Additional test for a failed mesh
# perturb a large value for design variable to make a failed mesh
xDV["shapey"][0] = 1000.0
funcs1 = {}
funcs1, fail1 = optFuncs.calcObjFuncValues(xDV)

if checkRegVal:
    # the checkMesh utility should detect failed mesh
    if fail1 is False:
        exit(1)

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

if checkRegVal:
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

if checkRegVal:
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

if checkRegVal:
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

if checkRegVal:
    if testMatNorm != testMatNorm1:
        exit(1)
