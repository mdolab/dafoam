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
from dafoam.mphys.mphys_dafoam import DAFoamSolverUnsteady
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/NACA0012Unsteady")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0
p0 = 0.0
A0 = 0.1
nuTilda0 = 4.5e-5

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DAPimpleFoam",
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": True,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "functionTimeOperator": "average",
    },
    "printInterval": 1,
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural", "dynAdjustTol": False},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "solverInput": {
        # "aero_vol_coords": {"type": "volCoord"},
        "patchV": {"type": "patchVelocity", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
    },
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}


class Top(Group):
    def setup(self):

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.add_subsystem(
            "cruise", DAFoamSolverUnsteady(solver_options=daOptions, mesh_options=meshOptions), promotes=["*"]
        )

    def configure(self):

        # add the design variables to the dvs component's output
        self.dvs.add_output("patchV", val=np.array([10.0, 20.0]))

        # define the design variables to the top level
        self.add_design_var("patchV", lower=-50.0, upper=50.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("CD", scaler=1.0)
        self.add_constraint("CL", equals=0.3)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("CD")
    funcDict["CL"] = prob.get_val("CL")
    reg_write_dict(funcDict, 1e-10, 1e-12)
