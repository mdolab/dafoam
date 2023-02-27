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

os.chdir("./input/Wing")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")

# aero setup
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
CL_target = 0.2
pitch0 = 2.0
rho0 = 1.0
A0 = 45.5

daOptions = {
    "designSurfaces": ["wing", "wing_te"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-10,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "couplingInfo": {
        "aeroacoustic": {
            "pRef": 0.0,
            "bladeSurface": {
                "patchNames": ["wing"]
            },
        },
    },
    "objFunc": {
        "Lift": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing", "wing_te"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 1.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {
        "gmresRelTol": 1.0e-8,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-6},
    "checkMeshThreshold": {
        "maxAspectRatio": 5000.0,
        "maxNonOrth": 70.0,
        "maxSkewness": 8.0,
        "maxIncorrectlyOrientedFaces": 0,
    },
    "designVar": {
        "twist": {"designVarType": "FFD"},
    },
    "adjPCLag": 1,
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": False,
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        super().configure()

        self.cruise.aero_post.mphys_add_funcs()

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # create constraint DV setup
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # geometry setup

        # Create reference axis
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Set up global design variables
        def twist(val, geo):
            for i in range(nRefAxPts):
                geo.rot_z["wingAxis"].coef[i] = -val[i]

        self.geometry.nom_addGlobalDV(dvName="twist", value=np.array([pitch0] * nRefAxPts), func=twist)

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([pitch0] * nRefAxPts))
        self.connect("twist", "geometry.twist")

        # define the design variables
        self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.bladeSurface.fAcou", scaler=1.0, index=0)
        self.add_constraint("cruise.aero_post.bladeSurface.xAcou", equals=1.0, scaler=1.0, indices=[0])
        self.add_constraint("cruise.aero_post.bladeSurface.nAcou", equals=1.0, scaler=1.0, indices=[0])
        self.add_constraint("cruise.aero_post.bladeSurface.aAcou", equals=1.0, scaler=1.0, indices=[0])
        self.add_constraint("cruise.aero_post.Lift", equals=1.0, scaler=1.0)

prob = om.Problem()
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.opt_settings = {
    "ACC": 1.0e-5,
    "MAXIT": 2,
    "IFILE": "opt_SLSQP.txt",
}
prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]


prob.add_recorder(om.SqliteRecorder("cases.sql"))
prob.recording_options["includes"] = []
prob.recording_options["record_objectives"] = True
prob.recording_options["record_constraints"] = True

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aerostruct.html")
prob.run_model()
totals = prob.compute_totals()

if gcomm.rank == 0:
    derivDict = {}
    derivDict["Lift"] = {}
    derivDict["Lift"]["twist"] = totals[("cruise.aero_post.functionals.Lift", "dvs.twist")]
    derivDict["xAcou"] = {}
    derivDict["xAcou"]["twist"] = totals[("cruise.aero_post.bladeSurface.xAcou", "dvs.twist")]
    derivDict["nAcou"] = {}
    derivDict["nAcou"]["twist"] = totals[("cruise.aero_post.bladeSurface.nAcou", "dvs.twist")]
    derivDict["aAcou"] = {}
    derivDict["aAcou"]["twist"] = totals[("cruise.aero_post.bladeSurface.aAcou", "dvs.twist")]
    derivDict["fAcou"] = {}
    derivDict["fAcou"]["twist"] = totals[("cruise.aero_post.bladeSurface.fAcou", "dvs.twist")]
    # reg_write_dict(derivDict, 1e-4, 1e-6)
