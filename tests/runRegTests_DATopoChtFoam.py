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

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("constant/turbulenceProperties", "SpalartAllmaras;", "dummy;")
    # replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
nSelectedCells = 32
nPatchFaces = 49

daOptions = {
    "solverName": "DATopoChtFoam",
    "primalMinResTol": 1.0e-11,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
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
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3, "T": 300.0},
    "inputInfo": {
        "eta": {
            "type": "field",
            "fieldName": "eta",
            "fieldType": "scalar",
            "distributed": False,
            "cellSetName": "myCellSet",
            "components": ["solver", "function"],
        },
        "patchUField": {
            "type": "patchField",
            "patches": ["inlet"],
            "fieldName": "U",
            "fieldType": "vector",
            "distributed": False,
            "indices": [0],
            "components": ["solver", "function"],
        },
    },
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(daOptions, None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

    def configure(self):

        self.dvs.add_output("eta", val=np.ones(nSelectedCells))
        self.dvs.add_output("patchUField", val=np.ones(nPatchFaces) * U0)

        # manually connect the dvs output to the geometry and cruise
        self.connect("eta", "cruise.eta")
        self.connect("patchUField", "cruise.patchUField")
        # define the design variables to the top level
        self.add_design_var("eta", lower=0.0, upper=1, scaler=1.0, indices=[0, 15, 30])
        self.add_design_var("patchUField", lower=-100, upper=100, scaler=1.0, indices=[20])

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

prob.run_model()
results = prob.check_totals(
    of=["cruise.aero_post.CD", "cruise.aero_post.TMean"],
    wrt=["eta", "patchUField"],
    compact_print=True,
    step=1e-3,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("cruise.aero_post.CD")
    funcDict["TMean"] = prob.get_val("cruise.aero_post.TMean")
    derivDict = {}
    derivDict["CD"] = {}
    derivDict["CD"]["eta-Adjoint"] = results[("cruise.aero_post.CD", "eta")]["J_fwd"][0]
    derivDict["CD"]["eta-FD"] = results[("cruise.aero_post.CD", "eta")]["J_fd"][0]
    derivDict["CD"]["patchUField-Adjoint"] = results[("cruise.aero_post.CD", "patchUField")]["J_fwd"][0]
    derivDict["CD"]["patchUField-FD"] = results[("cruise.aero_post.CD", "patchUField")]["J_fd"][0]
    derivDict["TMean"] = {}
    derivDict["TMean"]["eta-Adjoint"] = results[("cruise.aero_post.TMean", "eta")]["J_fwd"][0]
    derivDict["TMean"]["eta-FD"] = results[("cruise.aero_post.TMean", "eta")]["J_fd"][0]
    derivDict["TMean"]["patchUField-Adjoint"] = results[("cruise.aero_post.TMean", "patchUField")]["J_fwd"][0]
    derivDict["TMean"]["patchUField-FD"] = results[("cruise.aero_post.TMean", "patchUField")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-6, 1e-8)
