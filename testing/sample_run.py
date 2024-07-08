#!/usr/bin/env python
"""
DAFoam run script for the periodic hills case
"""

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
import sys
import os
import argparse
from pygeo import *
from pyspline import *
from idwarp import *
from pyoptsparse import Optimization, OPT
import numpy as np



# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--opt", help="optimizer to use", type=str, default="snopt")
parser.add_argument("--task", help="type of run to do", type=str, default="runPrimal")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# Define the global parameters here
U0 = 1.0
p0 = 0.0
nuTilda0 = 0.024
T0 = 1
J0 = 525.61604 # this is the actual baseline norm
J1 = 300000

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["obstacle"],
    "solverName": "DASimpleTFoam",
    "useAD": {"mode": "reverse"},
    "writeSensMap": ["fvSource"],
    "primalMinResTol": 1.0e-5,
    "fvSource": {
        "source1": {
            "type": "passiveScalar",
            "source": "boxToCell",
 	 "min": [-0.5, -0.1, -0.1], # p1 and p2 define the axis and width
            "max": [-0.6, 0.1, 0.1], # p2-p1 should be the axis of the cylinder
            "power": 5.0,  # here we should prescribe the power in W
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 2, "jacMatReOrdering": "natural", "gmresMaxIters": 3000},
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
        "T": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-7, "FFD": 1e-3},
    "adjPCLag": 100,
    "designVar": {
        #"fvSource": {"designVarType": "Field", "fieldName": "fvSource", "fieldType": "vector"},
        #"fTSource": {"designVarType": "Field", "fieldName": "fTSource", "fieldType": "scalar"}
    },
    "adjPCLag": 100,


}

# mesh warping parameters, users need to manually specify the symmetry plane and their normals
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.5], [0.0, 0.0, 1.0]], [[0.0, 0.0, -0.5], [0.0, 0.0, -1.0]]],
}

# options for optimizers
if args.opt == "snopt":
    optOptions = {
        "Major feasibility tolerance": 1.0e-10,
        "Major optimality tolerance": 1.0e-10,
        "Minor feasibility tolerance": 1.0e-10,
        "Verify level": -1,
        "Function precision": 1.0e-10,
        "Major iterations limit": 200,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
        #"Print frequency": 1,
        #"Summary frequency": 1,
        #"System information": 1,
        #"Verify level": 3,
        #"Major Print level": 11,
        #"Minor print level": 11,
        #"Scale option": 2,
        #"Solution":1,
    }
elif args.opt == "ipopt":
    optOptions = {
        "tol": 1.0e-7,
        "constr_viol_tol": 1.0e-7,
        "max_iter": 200,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.opt == "slsqp":
    optOptions = {
        "ACC": 1.0e-7,
        "MAXIT": 50,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("opt arg not valid!")
    exit(0)


# =============================================================================
# Design variable setup
# =============================================================================
DVGeo = DVGeometry("./FFD/cylinderFFD.xyz")
DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")

def fvSource(val, geo):
    for idxI, v in enumerate(val):
        cellI = idxI // 3
        compI = idxI % 3
        DASolver.setFieldValue4GlobalCellI(b"fvSource", v, cellI, compI)


nCells = 58600
fvSource0 = np.zeros(nCells*3, dtype="d")
DVGeo.addGlobalDV("fvSource", value=fvSource0, func=fvSource, lower=-5, upper=5, scale=10)
"""

def betaFieldInversion(val, geo):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"betaFieldInversion", v, idxI)
        DASolver.updateBoundaryConditions(b"betaFieldInversion", b"scalar")

# select points
nCells = 58000

beta0 = np.ones(nCells, dtype="d")


DVGeo.addGlobalDV("beta", value=beta0, func=betaFieldInversion, lower=-30, upper=30.0, scale=0.011)
"""

"""
def fTSource(val, geo):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"fTSource", v, idxI)
        DASolver.updateBoundaryConditions(b"fTSource", b"scalar")

# select points
nCells = 58600

fTSource0 = np.zeros(nCells, dtype="d")


DVGeo.addGlobalDV("fTSource", value=fTSource0, func=fTSource, lower=-5, upper=5, scale=1)
"""

# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.printFamilyList()
DASolver.setMesh(mesh)
# set evalFuncs
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# =============================================================================
# Constraint setup
# =============================================================================
# DVCon
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
[p0, v1, v2] = DASolver.getTriangulatedMeshSurface(groupName=DASolver.designSurfacesGroup)
surf = [p0, v1, v2]
DVCon.setSurface(surf)

# =============================================================================
# Initialize optFuncs for optimization
# =============================================================================
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# =============================================================================
# Task
# =============================================================================
if args.task == "opt":

    optProb = Optimization("opt", objFun=optFuncs.calcObjFuncValues, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    optProb.addObj("FI", scale=1)

    if gcomm.rank == 0:
        print(optProb)

    DASolver.runColoring()

    opt = OPT(args.opt, options=optOptions)
    histFile = "./%s_hist.hst" % args.opt
    sol = opt(optProb, sens=optFuncs.calcObjFuncSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

elif args.task == "runPrimal":

    optFuncs.runPrimal()

elif args.task == "runAdjoint":

    optFuncs.runAdjoint()

else:
    print("task arg not found!")
    exit(0)


# ============================================================
# Tests. Uncomment this section to perform checks on the code
# ============================================================

"""

a = DASolver.testFseq
b = DASolver.testCseq
np.savetxt('field.out', a, delimiter = ',')
np.savetxt('c.out', b, delimiter = ',')


a = DASolver.u_interp_full
b = DASolver.v_interp_full
c = DASolver.nans_u
d = DASolver.nans_v
e = DASolver.u_interp_nearest
f = DASolver.v_interp_nearest
g = DASolver.mask

np.savetxt('interp_U.out', a, delimiter = ',')
np.savetxt('interp_V.out', b, delimiter = ',')
np.savetxt('nans_U.out', c, delimiter = ',')
np.savetxt('nans_V.out', d, delimiter = ',')
np.savetxt('interp_U_nearest.out', e, delimiter = ',')
np.savetxt('interp_V_nearest.out', f, delimiter = ',')
np.savetxt('mask.out', g , delimiter = ',')
"""

