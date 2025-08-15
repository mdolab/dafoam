#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
import os
import numpy as np
from testFuncs import *

import openmdao.api as om
from openmdao.api import Group
from mphys.multipoint import Multipoint
from dafoam.mphys.mphys_dafoam import DAFoamBuilderUnsteady
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/pitzDailyScalarTransport")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")

TRef = 1.0

daOptions = {
    "designSurfaces": ["upperWall"],
    "solverName": "DAScalarTransportFoam",
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "UIn"},
    "primalBC": {"T0": {"variable": "T", "patches": ["inlet"], "value": [TRef]}},
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 1000,
        "PCMatUpdateInterval": 1000,
        "readZeroFields": True,
    },
    "function": {
        "TVOL": {
            "type": "variableVolSum",
            "source": "allCells",
            "varName": "T",
            "varType": "scalar",
            "index": 0,
            "isSquare": 0,
            "divByTotalVol": 0,
            "scale": 1.0,
            "timeOp": "average",
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "normalizeStates": {"T": 1.0},
    "inputInfo": {
        "TIn": {
            "type": "patchVar",
            "varName": "T",
            "varType": "scalar",
            "patches": ["inlet"],
            "components": ["solver", "function"],
        },
    },
    "unsteadyCompOutput": {
        "obj": ["TVOL"],
    },
}


class Top(Group):
    def setup(self):

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.add_subsystem(
            "cruise",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=None),
            promotes=["*"],
        )

    def configure(self):

        # add the design variables to the dvs component's output
        self.dvs.add_output("TIn", val=np.array([1.0]))

        # define the design variables to the top level
        self.add_design_var("TIn", lower=-50.0, upper=50.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("obj", scaler=1.0)
        # self.add_constraint("CL", equals=0.3)


funcDict = {}
derivDict = {}

dvNames = ["TIn"]
dvIndices = [[0]]
funcNames = ["cruise.solver.obj"]

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
