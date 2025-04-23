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
    os.system("pimpleFoam")
    os.system("getFIData -refFieldName U -refFieldType vector")
    os.system("getFIData -refFieldName p -refFieldType scalar")
    os.system("getFIData -refFieldName wallShearStress -refFieldType vector")
    # os.system("getFIData -refFieldName wallHeatFlux -refFieldType scalar")
    # os.system("decomposePar -time '0:'")
    os.system("cp constant/turbulenceProperties.sst constant/turbulenceProperties")
    # os.system("rm -rf 0.0* 0.1")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0

daOptions = {
    "solverName": "DAPimpleFoam",
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "primalBC": {
        "useWallFunction": False,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "readZeroFields": True,
    },
    "regressionModel": {
        "active": True,
        "reg_model": {
            "modelType": "neuralNetwork",
            "inputNames": ["KoU2", "ReWall", "TauoK"],
            "outputName": "betaFIK",
            "hiddenLayerNeurons": [5, 5],
            "inputShift": [0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0],
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
            "source": "allCells",
            "scale": 1.0,
            "mode": "field",
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1, 2],
            "timeDependentRefData": True,
            "timeOp": "average",
        },
        "PVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "p",
            "varType": "scalar",
            "timeDependentRefData": True,
            "timeOp": "average",
        },
        "UProbe": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "probePoint",
            "probePointCoords": [[0.5, 0.5, 0.5], [0.3, 0.2, 0.1]],
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1],
            "timeDependentRefData": True,
            "timeOp": "average",
        },
        "wallShearStressVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "wallShearStress",
            "varType": "vector",
            "indices": [0, 1],
            "timeDependentRefData": True,
            "timeOp": "average",
        },
        # "wallHeatFlux": {
        #     "type": "variance",
        #     "source": "patchToFace",
        #     "patches": ["walls"],
        #     "scale": 1.0,
        #     "mode": "surface",
        #     "varName": "wallHeatFlux",
        #     "varType": "scalar",
        #     "indices": [0],
        #     "timeDependentRefData": True,
        #     "timeOp": "average",
        # },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "reg_model": {"type": "regressionPar", "components": ["solver", "function"]},
    },
    "unsteadyCompOutput": {
        "UVar": ["UVar", "UProbe"],
        "PVar": ["PVar"],
        "wallShearStressVar": ["wallShearStressVar"],
    },
    "decomposeParDict": {"args": ["-time", "0:"]},
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
        nParameters = self.scenario.solver.DASolver.getNRegressionParameters("reg_model")
        parameter0 = np.ones(nParameters) * 0.005
        self.dvs.add_output("reg_model", val=parameter0)

        # define the design variables to the top level
        self.add_design_var("reg_model", lower=-100.0, upper=100.0, scaler=1.0, indices=[0, 50])

        # add constraints and the objective
        # add constraints and the objective
        self.add_objective("UVar", scaler=1.0)
        self.add_constraint("PVar", equals=0.3)
        self.add_constraint("wallShearStressVar", equals=0.3)


funcDict = {}
derivDict = {}

dvNames = ["reg_model"]
dvIndices = [[0, 50]]
funcNames = [
    "scenario.solver.UVar",
    "scenario.solver.PVar",
    "scenario.solver.wallShearStressVar",
    # "scenario.solver.wallHeatFlux"
]

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
