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
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")
    os.system("cp -r constant/turbulenceProperties.safv3 constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 50.0
p0 = 101325.0
nuTilda0 = 4.5e-4

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-11,
    "primalMinResTolDiff": 1e4,
    "useConstrainHbyA": False,
    "printDAOptions": False,
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
        "thermo:mu": 1.0e-5,
    },
    "fvSource": {
        "disk1": {
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
            "adjustThrust": 0,
            "targetThrust": 1.0,
        },
    },
    "function": {
        "CMZ": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["walls"],
            "axis": [0.0, 0.0, 1.0],
            "center": [0.5, 0.5, 0.5],
            "scale": 1.0,
        },
        "TP1": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["inlet"],
            "scale": 1.0 / (0.5 * U0 * U0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": p0, "phi": 1.0, "nuTilda": 1e-3, "T": 300.0},
    "inputInfo": {
        "actuator_disk": {
            "type": "fvSourcePar",
            "fvSourceName": "disk1",
            "indices": [0, 7],
            "components": ["solver", "function"],
        },
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inlet"],
            "flowAxis": "x",
            "normalAxis": "y",
            "components": ["solver", "function"],
        },
        "nutilda_in": {
            "type": "patchVar",
            "varName": "nuTilda",
            "varType": "scalar",
            "patches": ["inlet"],
            "components": ["solver", "function"],
        },
    },
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # create the mesh and cruise scenario because we only have one analysis point
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/FFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # add the dv_geo object to the builder solver. This will be used to write deformed FFDs and forward AD
        self.cruise.coupling.solver.add_dvgeo(self.geometry.DVGeo)

        # geometry setup

        # add the design variables to the dvs component's output
        self.dvs.add_output("actuator_disk", val=np.array([0.5, 0.4]))
        self.dvs.add_output("patchV", val=np.array([50.0, 0.0]))
        self.dvs.add_output("nutilda_in", val=np.array([nuTilda0]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("actuator_disk", "cruise.actuator_disk")
        self.connect("patchV", "cruise.patchV")
        self.connect("nutilda_in", "cruise.nutilda_in")

        # define the design variables to the top level
        self.add_design_var("actuator_disk", lower=-50.0, upper=50.0, scaler=1.0)
        self.add_design_var("patchV", lower=-50.0, upper=50.0, scaler=1.0)
        self.add_design_var("nutilda_in", lower=-50.0, upper=50.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.TP1", scaler=1.0)


funcDict = {}
derivDict = {}

dvNames = ["actuator_disk", "patchV", "nutilda_in"]
dvIndices = [[0, 1], [0, 1], [0]]
funcNames = [
    "cruise.aero_post.functionals.TP1",
    "cruise.aero_post.functionals.CMZ",
]

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(derivDict, 1e-8, 1e-12)
