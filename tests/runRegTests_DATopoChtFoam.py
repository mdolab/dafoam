#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
import os
import numpy as np
from testFuncs import *

import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ChannelTopoCht")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor* *.bin")
    os.system("cp -r 0_orig 0")

# aero setup
U0 = 0.1
p0 = 0.0
nuTilda0 = 4.5e-5
nCells = 1600

daOptions = {
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "solverName": "DATopoChtFoam",
    "primalMinResTol": 1.0e-10,
    "function": {
        "TP1": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["inlet"],
            "scale": 1.0,
        },
        "TP2": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["outlet"],
            "scale": 1.0,
        },
        "TMean": {
            "type": "patchMean",
            "source": "patchToFace",
            "patches": ["outlet"],
            "varName": "T",
            "varType": "scalar",
            "index": 0,
            "scale": 1.0,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-8,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "gmresMaxIters": 2000,
        "gmresRestart": 2000,
    },
    "normalizeStates": {
        "U": U0,
        "p": 0.1,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
        "T": 300.0,
    },
    "inputInfo": {
        "eta": {
            "type": "field",
            "fieldName": "eta",
            "fieldType": "scalar",
            "components": ["solver", "function"],
            "distributed": 0,
        },
        "u_in": {
            "type": "patchVar",
            "varName": "U",
            "varType": "vector",
            "patches": ["inlet"],
            "components": ["solver", "function"],
        },
    },
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

    def configure(self):

        # add the design variables to the dvs component's output
        self.dvs.add_output("eta", val=np.ones(nCells))
        self.dvs.add_output("u_in", val=np.array([0.1, 0, 0]))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("eta", "scenario1.eta")
        self.connect("u_in", "scenario1.u_in")

        # define the design variables to the top level
        self.add_design_var("eta", lower=0, upper=1, scaler=1.0)
        self.add_design_var("u_in", lower=0, upper=1, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.TMean", scaler=1.0)


funcDict = {}
derivDict = {}

dvNames = ["u_in", "eta"]
dvIndices = [[0], [0]]
funcNames = [
    "scenario1.aero_post.functionals.TMean",
    "scenario1.aero_post.functionals.TP2",
]

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(funcDict, 1e-8, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
