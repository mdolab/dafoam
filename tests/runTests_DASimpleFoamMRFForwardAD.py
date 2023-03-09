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

os.chdir("./input/CompressorFluid")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r constant/MRFProperties.incompressible constant/MRFProperties")
    os.system("cp -r system/fvSolution.incompressible system/fvSolution")
    os.system("cp -r system/fvSchemes.incompressible system/fvSchemes")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

MRF0 = -100.0

# test incompressible solvers
daOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["blade"],
    "primalMinResTol": 1e-12,
    "useAD": {"mode": "forward", "dvName": "MRF", "seedIndex": 0},
    "primalBC": {
        "MRF": MRF0,
    },
    "objFunc": {
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["blade"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "designVar": {"MRF": {"designVarType": "BC"},},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}

# DVGeo
DVGeo = DVGeometry("./FFD/localFFD.xyz")
# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")

def MRF(val, geo):
    DASolver.setOption("primalBC", {"MRF": float(val[0])})
    DASolver.updateDAOption()

DVGeo.addGlobalDV("MRF", [MRF0], MRF, lower=-1000.0, upper=1000.0, scale=1.0)

# DAFoam
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
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
funcsSens["CMZ"] = {}

# Run
# MRF
DASolver()
funcsSens["CMZ"]["MRF"] = DASolver.getForwardADDerivVal("CMZ")

if gcomm.rank == 0:
    reg_write_dict(funcsSens, 1e-4, 1e-5)
