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
    os.system("rm -rf processor* *.bin")
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")
    os.system("cp -r constant/turbulenceProperties.sst constant/turbulenceProperties")
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
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
        "thermo:mu": 1.0e-5,
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
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": p0, "phi": 1.0, "nuTilda": 1e-3, "T": 300.0},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inlet"],
            "flowAxis": "x",
            "normalAxis": "y",
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
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[1, 0, 1].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.zeros(1))
        self.dvs.add_output("patchV", val=np.array([U0, 0.0]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")
        self.connect("patchV", "cruise.patchV")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("patchV", lower=-50.0, upper=50.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.3)


funcDict = {}
derivDict = {}

dvNames = ["shape", "patchV"]
dvIndices = [[0], [0, 1]]
funcNames = ["cruise.aero_post.functionals.CL", "cruise.aero_post.functionals.CD"]

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict)

# write the test results
if gcomm.rank == 0:
    reg_write_dict(derivDict, 1e-8, 1e-12)
