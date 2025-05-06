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

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible.unsteady/* system/")
    replace_text_in_file("system/fvSchemes", "meshWaveFrozen;", "meshWave;")

# aero setup
U0 = 10.0
nCells = 343
nFields = 3

daOptions = {
    "solverName": "DAPimpleFoam",
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "beta"},
    "primalBC": {
        "useWallFunction": False,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "readZeroFields": True,
        "additionalOutput": ["betaFINuTilda"],
        "reduceIO": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "timeOp": "average",
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "beta": {
            "type": "fieldUnsteady",
            "fieldName": "betaFINuTilda",
            "fieldType": "scalar",
            "stepInterval": 5,
            "components": ["solver", "function"],
            "distributed": 0,
            "interpolationMethod": "linear",
        },
    },
    "unsteadyCompOutput": {
        "CD": ["CD"],
    },
}


class Top(Group):
    def setup(self):

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.add_subsystem(
            "scenario",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=None),
            promotes=["*"],
        )

    def configure(self):

        # add the design variables to the dvs component's output
        beta0 = np.ones(nCells * nFields)
        for i in range(nCells, 2 * nCells):
            beta0[i] = 1.1
        for i in range(2 * nCells, 3 * nCells):
            beta0[i] = 1.3
        self.dvs.add_output("beta", val=beta0)

        # define the design variables to the top level
        self.add_design_var("beta", lower=-3, upper=3, scaler=1.0, indices=[100, 400, 700])

        # add constraints and the objective
        self.add_objective("CD", scaler=1.0)


funcDict = {}
derivDict = {}

dvNames = ["beta"]
dvIndices = [[100, 400, 700]]
funcNames = [
    "scenario.solver.CD",
]

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)


# ************** RBF **************
daOptions["inputInfo"] = {
    "beta_rbf": {
        "type": "fieldUnsteady",
        "fieldName": "betaFINuTilda",
        "fieldType": "scalar",
        "stepInterval": 5,
        "components": ["solver", "function"],
        "distributed": 0,
        "interpolationMethod": "rbf",
        "offset": 1.0,
    }
}
daOptions["useAD"] = {"mode": "reverse", "seedIndex": 0, "dvName": "beta_rbf"}


class Top1(Group):
    def setup(self):

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.add_subsystem(
            "scenario",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=None),
            promotes=["*"],
        )

    def configure(self):

        # add the design variables to the dvs component's output
        # generate some variation in time
        beta_rbf0 = np.ones(nCells * nFields * 2)
        for i in range(nCells, 2 * nCells):
            beta_rbf0[i] = 5.0
        for i in range(2 * nCells, 3 * nCells):
            beta_rbf0[i] = 10.0
        for i in range(3 * nCells, 4 * nCells):
            beta_rbf0[i] = 1.0
        for i in range(4 * nCells, 5 * nCells):
            beta_rbf0[i] = 5.0
        for i in range(5 * nCells, 6 * nCells):
            beta_rbf0[i] = 10.0
        beta_rbf0 = beta_rbf0 * 0.1

        self.dvs.add_output("beta_rbf", val=beta_rbf0)

        # define the design variables to the top level
        self.add_design_var("beta_rbf", lower=-3, upper=3, scaler=1.0, indices=[100, 700, 1400])

        # add constraints and the objective
        self.add_objective("CD", scaler=1.0)


funcDict = {}
derivDict = {}

dvNames = ["beta_rbf"]
dvIndices = [[100, 700, 1400]]
funcNames = [
    "scenario.solver.CD",
]

# run the adjoint and forward ref
run_tests(om, Top1, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
