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
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "reg_model"},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": False,
        "transport:nu": 1.5e-5,
    },
    "regressionModel": {
        "active": True,
        "reg_model": {
            "modelType": "neuralNetwork",
            "inputNames": ["VoS", "PoD", "chiSA", "pGradStream", "PSoSS", "SCurv", "UOrth"],
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [5, 5],
            "inputShift": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
        "UVar": {
            "type": "variance",
            "source": "boxToCell",
            "min": [0.2, 0.2, 0.3],
            "max": [0.8, 0.8, 0.9],
            "scale": 1.0,
            "mode": "field",
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1, 2],
            "timeDependentRefData": False,
        },
        "PVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "p",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "UProbe": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "probePoint",
            "probePointCoords": [[0.51, 0.52, 0.53], [0.2, 0.3, 0.4]],
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1],
            "timeDependentRefData": False,
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "reg_model": {"type": "regressionPar", "components": ["solver", "function"]},
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

        self.mphys_add_scenario("scenario", ScenarioAerodynamic(aero_builder=dafoam_builder))

    def configure(self):

        nParameters = self.scenario.coupling.solver.DASolver.getNRegressionParameters("reg_model")
        parameter0 = np.ones(nParameters) * 0.005
        self.dvs.add_output("reg_model", val=parameter0)
        self.connect("reg_model", "scenario.reg_model")

        # define the design variables to the top level
        self.add_design_var("reg_model", lower=-100.0, upper=100.0, scaler=1.0, indices=[0, 50])
        # add constraints and the objective
        self.add_objective("scenario.aero_post.UVar", scaler=1.0)
        self.add_constraint("scenario.aero_post.PVar", equals=0.3)
        self.add_constraint("scenario.aero_post.UProbe", equals=0.3)


funcDict = {}
derivDict = {}

dvNames = ["reg_model"]
dvIndices = [[0, 50]]
funcNames = [
    "scenario.aero_post.functionals.UVar",
    "scenario.aero_post.functionals.PVar",
    "scenario.aero_post.functionals.UProbe",
]

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
