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
nuTilda0 = 4.5e-4

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-11,
    "primalMinResTolDiff": 1e4,
    "debug": True,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusToCell",
            "p1": [0.3, 0.5, 0.5],  # p1 and p2 define the axis and width
            "p2": [0.7, 0.5, 0.5],  # p2-p1 should be streamwise
            "innerRadius": 0.01,
            "outerRadius": 0.6,
            "rotDir": "left",
            "scale": 10.0,
            "POD": 0.7,
        },
        "disk2": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusSmooth",
            "center": [0.5, 0.5, 0.5],
            "direction": [1.0, 0.0, 0.0],
            "innerRadius": 0.01,
            "outerRadius": 0.4,
            "rotDir": "right",
            "scale": 100.0,
            "POD": 0.8,
            "eps": 0.1,  # eps should be of cell size
            "expM": 1.0,
            "expN": 0.5,
            "adjustThrust": 1,
            "targetThrust": 1.0,
        },
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 0.1,
        },
    },
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(daOptions, None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))


prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")
prob.run_model()
CD = prob.get_val("cruise.aero_post.CD")


print("CD", CD)
if (abs(CD - 2.4300457) / (CD + 1e-16)) > 1e-6:
    print("ActuatorDisk test failed!")
    exit(1)
else:
    print("ActuatorDisk test passed!")
