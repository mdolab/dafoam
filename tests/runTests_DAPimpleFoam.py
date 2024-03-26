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

os.chdir("./reg_test_files-main/NACA0012Unsteady")

if gcomm.rank == 0:
    os.system("rm -rf processor*")

twist0 = 30
U0 = 10
alpha0 = 0

# test incompressible solvers
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DAPimpleFoam",
    "primalBC": {"U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0, 0]}, "useWallFunction": True},
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "objFuncTimeOperator": "average",
    },
    "printIntervalUnsteady": 1,
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusSmooth",
            "center": [-0.55, 0.0, 0.05],
            "direction": [1.0, 0.0, 0.0],
            "innerRadius": 0.01,
            "outerRadius": 0.4,
            "rotDir": "right",
            "scale": 100.0,
            "POD": 0.0,
            "eps": 0.1,  # eps should be of cell size
            "expM": 1.0,
            "expN": 0.5,
            "adjustThrust": 0,
            "targetThrust": 1.0,
        },
    },
    "regressionModel": {
        "active": True,
        "modelType": "neuralNetwork",
        "inputNames": ["VoS", "PoD", "chiSA", "pGradStream", "PSoSS", "SCurv", "UOrth"],
        "outputName": "betaFINuTilda",
        "hiddenLayerNeurons": [5, 5],
        "inputShift": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "inputScale": [1.0, 0.00001, 0.01, 1.0, 1.0, 1.0, 1.0],
        "outputShift": 1.0,
        "outputScale": 1.0,
        "activationFunction": "tanh",
        "printInputInfo": False,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "alpha",
                "scale": 1.0,
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
                "scale": 1.0,
                "addToAdjoint": False,
            }
        },
        "CMZVAR": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["wing"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.25, 0.0, 0.05],
                "scale": 1.0,
                "addToAdjoint": True,
                "calcRefVar": True,
                "ref": [0.1, 0.05, 0.04, 0.02, 0.02, 0.01, 0.0, -0.01, -0.01, -0.02]
            }
        }
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
        "uin": {"designVarType": "BC", "patches": ["inout"], "variable": "U", "comp": 0},
        "twist": {"designVarType": "FFD"},
        "alpha": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
        "actuator": {
            "actuatorName": "disk1",
            "designVarType": "ACTD",
            "comps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
        "parameter": {"designVarType": "RegPar"},
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
# twist
def twist(val, geo):
    for i in range(2):
        geo.rot_z["bodyAxis"].coef[i] = -val[0]


DVGeo.addGlobalDV("twist", [twist0], twist, lower=-100.0, upper=100.0, scale=1.0)


def uin(val, geo):
    inletU = float(val[0])
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": [inletU, 0, 0]}})
    DASolver.updateDAOption()


DVGeo.addGlobalDV("uin", [U0], uin, lower=0.0, upper=100.0, scale=1.0)


def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


DVGeo.addGlobalDV("alpha", value=[alpha0], func=alpha, lower=0.0, upper=10.0, scale=1.0)


def actuator(val, geo):
    actX = float(val[0])
    actY = float(val[1])
    actZ = float(val[2])
    actDirx = float(val[3])
    actDiry = float(val[4])
    actDirz = float(val[5])
    actR1 = float(val[6])
    actR2 = float(val[7])
    actScale = float(val[8])
    actPOD = float(val[9])
    actExpM = float(val[10])
    actExpN = float(val[11])
    T = float(val[12])
    DASolver.setOption(
        "fvSource",
        {
            "disk1": {
                "type": "actuatorDisk",
                "source": "cylinderAnnulusSmooth",
                "center": [actX, actY, actZ],
                "direction": [actDirx, actDiry, actDirz],
                "innerRadius": actR1,
                "outerRadius": actR2,
                "rotDir": "right",
                "scale": actScale,
                "POD": actPOD,
                "eps": 0.1,  # eps should be of cell size
                "expM": actExpM,
                "expN": actExpN,
                "adjustThrust": 0,
                "targetThrust": T,
            },
        },
    )
    DASolver.updateDAOption()


# actuator
DVGeo.addGlobalDV(
    "actuator",
    value=[-0.55, 0.0, 0.05, 1.0, 0.0, 0.0, 0.01, 0.4, 100.0, 0.0, 1.0, 0.5, 1.0],
    func=actuator,
    lower=-100.0,
    upper=100.0,
    scale=1.0,
)


def regModel(val, geo):
    for idxI in range(len(val)):
        val1 = float(val[idxI])
        DASolver.setRegressionParameter(idxI, val1)


# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)

nParameters = DASolver.solver.getNRegressionParameters()

parameter0 = np.ones(nParameters) * 0.1
DASolver.addInternalDV("parameter", parameter0, regModel, lower=-100.0, upper=100.0, scale=1.0)

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
    iDV = DASolver.getInternalDVDict()
    allDV = {**xDV, **iDV}
    funcs = {}
    funcs, fail = optFuncs.calcObjFuncValues(allDV)
    funcsSens = {}
    funcsSens, fail = optFuncs.calcObjFuncSens(allDV, funcs)

    parameterNormU = np.linalg.norm(funcsSens["CD"]["parameter"])
    funcsSens["CD"]["parameter"] = parameterNormU

    parameterNormM = np.linalg.norm(funcsSens["CMZVAR"]["parameter"])
    funcsSens["CMZVAR"]["parameter"] = parameterNormM

    if gcomm.rank == 0:
        reg_write_dict(funcs, 1e-8, 1e-10)
        reg_write_dict(funcsSens, 1e-4, 1e-6)
