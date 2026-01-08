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

os.chdir("./reg_test_files-main/ChannelConjugateHeatV4/thermal")
if gcomm.rank == 0:
    os.system("rm -rf processor*")

daOptions = {
    "designSurfaces": ["channel_outer", "channel_inner", "channel_sides"],
    "solverName": "DAHeatTransferFoam",
    "debug": True,
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
            "radius": 0.05,
            "power": 1000.0,
        },
        "source2": {
            "type": "heatSource",
            "source": "cylinderSmooth",
            "center": [0.201, 0.056, 0.026],
            "axis": [1.0, 0.0, 0.0],
            "length": 0.4,
            "radius": 0.05,
            "power": 1000.0,
            "eps": 0.01,
            "snapCenter2Cell": True,
        },
    },
    "inputInfo": {
        "heat_source": {
            "type": "fvSourcePar",
            "fvSourceName": "source2",
            "indices": [0, 6],
            "components": ["solver", "function"],
        },
    },
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(daOptions, None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

    def configure(self):
        self.dvs.add_output("heat_source", val=np.array([0.3, 0.45]))
        self.connect("heat_source", "cruise.heat_source")
        self.add_design_var("heat_source", lower=-50.0, upper=50.0, scaler=1.0)
        self.add_objective("cruise.aero_post.HFX", scaler=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# verify the total derivatives against the finite-difference
prob.run_model()
totals = prob.compute_totals()

HFX = prob.get_val("cruise.aero_post.HFX")[0]
print("HFX", HFX)
if (abs(HFX - 180000.0) / (HFX + 1e-16)) > 1e-6:
    print("DAHeatTransferFoam test failed!")
    exit(1)
else:
    print("DAHeatTransferFoam test passed!")
