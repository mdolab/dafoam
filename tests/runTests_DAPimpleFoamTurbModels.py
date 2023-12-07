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
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)

gcomm = MPI.COMM_WORLD

os.chdir("./input/NACA0012Unsteady")

funcs = {}

for model in ["SA", "KE", "KW", "SST"]:

    if gcomm.rank == 0:
        os.system("rm -rf processor* *.bin")
        os.system("cp constant/turbulenceProperties_%s constant/turbulenceProperties" % model)

    twist0 = 30
    U0 = 10
    alpha0 = 0

    # test incompressible solvers
    daOptions = {
        "designSurfaces": ["wing"],
        "solverName": "DAPimpleFoam",
        "primalBC": {
            "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0, 0]},
            "k0": {"variable": "k", "patches": ["inout"], "value": [0.015]},
            "omega0": {"variable": "omega", "patches": ["inout"], "value": [100]},
            "epsilon0": {"variable": "epsilon", "patches": ["inout"], "value": [0.135]},
            "useWallFunction": True,
        },
        "unsteadyAdjoint": {"mode": "timeAccurate", "PCMatPrecomputeInterval": 5, "PCMatUpdateInterval": 1, "objFuncTimeOperator": "average"},
        "printIntervalUnsteady": 1,
        "primalVarBounds": {"kMin": -1e16, "omegaMin": -1e16, "epsilonMin": -1e16},
        "useConstrainHbyA": True,
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

    # =============================================================================
    # Design variable setup
    # =============================================================================
    DVGeo = DVGeometry("./FFD/FFD.xyz")
    DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")

    def alpha(val, geo):
        aoa = val[0] * np.pi / 180.0
        inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
        DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
        DASolver.updateDAOption()

    DVGeo.addGlobalDV("alpha", value=[alpha0], func=alpha, lower=0.0, upper=10.0, scale=1.0)

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

    xDV = DVGeo.getValues()
    optFuncs.calcObjFuncValues(xDV)

    DASolver.runColoring()

    dRdWTPC = PETSc.Mat().create(PETSc.COMM_WORLD)
    DASolver.solver.calcdRdWT(DASolver.xvVec, DASolver.wVec, 1, dRdWTPC)
    DASolver.solver.calcPCMatWithFvMatrix(dRdWTPC)

    funcs[model] = dRdWTPC.norm()

if gcomm.rank == 0:
    reg_write_dict(funcs, 1e-4, 1e-6)
