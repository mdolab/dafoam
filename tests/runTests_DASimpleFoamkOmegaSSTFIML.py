#!/usr/bin/env python
"""
Run Python tests for DASimpleFoam
"""

from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
from pygeo import *
from idwarp import *
from testFuncs import *
import numpy as np

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/NACA0012")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.sst constant/turbulenceProperties")

replace_text_in_file(
    "constant/turbulenceProperties",
    "RASModel             kOmegaSST;",
    "RASModel             kOmegaSSTFIML;",
)

U0 = 10.0
p0 = 0.0
k0 = 0.18
omega0 = 1225.0
A0 = 0.1
alpha0 = 1.0
LRef = 1.0

# test incompressible solvers
aeroOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 5e-5,
    "primalMinResTolDiff": 1e4,
    "tensorflow": {"active": True, "model": {"predictBatchSize": 100}},
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
    },
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "k": k0, "omega": omega0 / 10, "phi": 1.0},
    "adjEqnOption": {"gmresRelTol": 1.0e-5, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "designVar": {
        "alpha": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
    },
    "fvSource": {
        "gradP": {
            "type": "uniformPressureGradient",
            "value": 1e-6,
            "direction": [1.0, 0.0, 0.0],
        }
    },
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}

# DAFoam
FFDFile = "./FFD/wingFFD.xyz"
DVGeo = DVGeometry(FFDFile)

# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")


def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


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
DASolver.runColoring()
xDV = DVGeo.getValues()
funcs = {}
funcs, fail = optFuncs.calcObjFuncValues(xDV)
funcsSens = {}
funcsSens, fail = optFuncs.calcObjFuncSens(xDV, funcs)

if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-6, 1e-10)
    reg_write_dict(funcsSens, 1e-3, 1e-6)
