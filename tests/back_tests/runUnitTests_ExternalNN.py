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
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "tensorflow": {
        "active": True,
        "dummy_nn_model": {
            "predictBatchSize": 1000,
            "nInputs": 2,
        },
    },
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
        "transport:nu": 1.5e-5,
    },
    "regressionModel": {
        "active": True,
        "dummy_nn_model": {
            "modelType": "externalTensorFlow",
            "inputNames": ["VoS", "PoD"],
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [5, 5],
            "inputShift": [0.0, 0.0],
            "inputScale": [1.0, 1.0],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "writeFeatures": True,
            "outputUpperBound": 1e2,
            "outputLowerBound": -1e2,
            "defaultOutputValue": 1.0,
        },
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0,
        },
    },
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(daOptions, None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

    def configure(self):

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.3)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
CD = prob.get_val("cruise.aero_post.CD")[0]
print("CD, ", CD)
if abs(25.09280970753588 - CD) / 25.09280970753588 > 1e-6:
    print("externalNN test failed!")
    exit(1)
else:
    print("externalNN test passed!")
