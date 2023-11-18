#!/usr/bin/env python
"""
Run Python tests for DAPisoFoam
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

calcFDSens = 0
if len(sys.argv) != 1:
    if sys.argv[1] == "calcFDSens":
        calcFDSens = 1

gcomm = MPI.COMM_WORLD

os.chdir("./input/NACA0012Unsteady")

if gcomm.rank == 0:
    os.system("rm -rf processor*")
    os.system("./preProcessing.sh")

twist0 = 30
U0 = 10
nCells = 4032
alpha0 = 0

# test incompressible solvers
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DAPimpleFoam",
    "primalBC": {"U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0, 0]}, "useWallFunction": True},
    "unsteadyAdjoint": {"mode": "timeAccurate", "objFuncStartTime": 0.01, "objFuncEndTime": 0.045},
    "printIntervalUnsteady": 1,
    "objFunc": {
        "UVar": {
            "part1": {
                "type": "variance",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "scale": 1.0,
                "varName": "U",
                "varType": "vector",
                "addToAdjoint": True,
                "timeOperator": "average",
            },
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-8,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
    },
    "normalizeStates": {
        "U": 10,
        "p": 50,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "designVar": {
        "beta": {"designVarType": "Field", "fieldName": "betaFI", "fieldType": "scalar"},
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}

# =============================================================================
# Design variable setup
# =============================================================================
DVGeo = DVGeometry("./FFD/FFD.xyz")
DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")


def betaFieldInversion(val, geo):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"betaFI", v, idxI)
        DASolver.updateBoundaryConditions(b"betaFI", b"scalar")


beta0 = np.ones(nCells, dtype="d")
beta0[0] = 1.0
DVGeo.addGlobalDV("beta", value=beta0, func=betaFieldInversion, lower=1e-5, upper=10.0, scale=1.0)

# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.printFamilyList()
DASolver.setMesh(mesh)
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# =============================================================================
# Constraint setup
# =============================================================================
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
DVCon.setSurface(DASolver.getTriangulatedMeshSurface(groupName=DASolver.designSurfacesGroup))

# =============================================================================
# Initialize optFuncs for optimization
# =============================================================================
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# Run
if calcFDSens == 1:
    optFuncs.calcFDSens()
else:
    DASolver.runColoring()
    xDV = DVGeo.getValues()
    funcs = {}
    funcs, fail = optFuncs.calcObjFuncValues(xDV)
    funcsSens = {}
    funcsSens, fail = optFuncs.calcObjFuncSens(xDV, funcs)

    betaNorm = np.linalg.norm(funcsSens["UVar"]["beta"])
    funcsSens["UVar"]["beta"] = betaNorm

    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-4, 1e-6)
