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

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible.unsteady/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DAPimpleFoam",
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "primalBC": {
        # "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "readZeroFields": True,
    },
    "fvSource": {
        "line1": {
            "type": "actuatorLine",
            "center": [0.5, 0.5, 0.0],
            "direction": [1.0, 0.0, 0.0],
            "initial": [0.0, 1.0, 0.0],
            "innerRadius": 0.01,
            "outerRadius": 0.8,
            "scale": 1.0,  # scale the source such the integral equals desired thrust
            "POD": 1.0,
            "expM": 1.0,
            "expN": 0.5,
            "rpm": 60.0,
            "phase": 0.5,
            "eps": 0.2,
            "nBlades": 5,
            "rotDir": "right",
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
            "timeOp": "average",
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0,
            "timeOp": "average",
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "actuator_line": {
            "type": "fvSourcePar",
            "fvSourceName": "line1",
            "indices": [0, 1, 2, 17],  # centerX, centerY, centerZ, scale
            "components": ["solver", "function"],
        },
    },
    "unsteadyCompOutput": {
        "CD": ["CD"],
    },
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}


class Top(Group):
    def setup(self):

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.add_subsystem(
            "cruise",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=meshOptions),
            promotes=["*"],
        )

    def configure(self):

        # add the design variables to the dvs component's output
        self.dvs.add_output("actuator_line", val=np.array([0.4, 0.4, 0.1, 0.1]))

        # define the design variables to the top level
        self.add_design_var("actuator_line", lower=-50.0, upper=50.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("CD", scaler=1.0)
        # self.add_constraint("CL", equals=0.3)


prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")
prob.run_model()
CD = prob.get_val("CD")

print("CD", CD)
if (abs(CD - 29.264778550278) / (CD + 1e-16)) > 1e-8:
    print("ActuatorLine test failed!")
    exit(1)
else:
    print("ActuatorLine test passed!")
