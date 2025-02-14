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

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ChannelConjugateHeat/thermal")

daOptions = {
    "designSurfaces": ["channel_outer", "channel_inner", "channel_sides"],
    "solverName": "DAHeatTransferFoam",
    "function": {
        "HFX": {
            "type": "wallHeatFlux",
            "source": "patchToFace",
            "patches": ["channel_inner"],
            "scale": 1,
        },
    },
    "primalMinResTol": 1e-12,
    "fvSource": {
        "source1": {
            "type": "heatSource",
            "source": "cylinderToCell",
            "p1": [0.6, 0.055, 0.025],
            "p2": [1.0, 0.055, 0.025],
            "radius": 0.01,
            "power": 1000.0,
        },
        "source2": {
            "type": "heatSource",
            "source": "cylinderSmooth",
            "center": [0.201, 0.056, 0.026],
            "axis": [1.0, 0.0, 0.0],
            "length": 0.4,
            "radius": 0.005,
            "power": 1000.0,
            "eps": 0.001,
            "snapCenter2Cell": True,
        },
    },
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(daOptions, None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

    def configure(self):
        self.add_objective("cruise.aero_post.HFX", scaler=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()

if gcomm.rank == 0:
    funcDict = {}
    funcDict["HFX"] = prob.get_val("cruise.aero_post.HFX")
    reg_write_dict(funcDict, 1e-10, 1e-12)
