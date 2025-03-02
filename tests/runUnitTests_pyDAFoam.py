#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
import numpy as np

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")

if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [0.0]},
        "useWallFunction": False,
        "transport:nu": 1.5e-5,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 0.1,
            "dummy1": {},
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
DASolver.setOption("function", {"CD": {"direction": [0.0, 1.0, 0.0]}})
DASolver.updateDAOption()
function = DASolver.getOption("function")
if function["CD"]["direction"][1] != 1.0:
    print("setOption failed")
    exit(1)
# test 3 levels of subDict
DASolver.setOption("function", {"CD": {"dummy1": {"dummy2": {"dummy3": 2.5}}}})
DASolver.updateDAOption()
function = DASolver.getOption("function")
if function["CD"]["dummy1"]["dummy2"]["dummy3"] != 2.5:
    print("setOption failed")
    exit(1)
DASolver.calcPrimalResidualStatistics("print")
p = np.zeros(DASolver.solver.getNLocalCells())
U = np.zeros(3 * DASolver.solver.getNLocalCells())
DASolver.solver.updateBoundaryConditions("p", "scalar")
DASolver.solver.updateBoundaryConditions("U", "vector")
DASolver.solver.getOFField("p", "scalar", p)
DASolver.solver.getOFField("U", "vector", U)
normP = np.linalg.norm(p)
normP = gcomm.allreduce(normP, op=MPI.SUM)
normU = np.linalg.norm(U)
normU = gcomm.allreduce(normU, op=MPI.SUM)

print(normP, normU)

if abs(1779.052677196137 - normP) / normP > 1e-8:
    print("pyDAFoam failed")
elif abs(546.9793586769085 - normU) / normU > 1e-8:
    print("pyDAFoam failed")
else:
    print("pyDAFoam passed!")
