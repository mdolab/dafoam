#!/usr/bin/env python
"""
Run Python tests for DASimpleFoam
"""

import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from testFuncs import *

gcomm = MPI.COMM_WORLD

os.chdir("./input/CurvedCubeHexMesh")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system/fvSchemes.fp system/fvSchemes")
    os.system("cp -r system/fvSolution.fp system/fvSolution")
    os.system("cp -r constant/turbulenceProperties.safv3 constant/turbulenceProperties")

U0 = 1.0

# test incompressible solvers
daOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaceFamily": "designSurface",
    "designSurfaces": ["wallsbump"],
    "adjEqnSolMethod": "fixedPoint",
    "primalMinResTol": 1e-10,
    "adjUseColoring": False,
    "normalizeResiduals": ["tmp"],
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wallsbump"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {
        "relaxP": 0.3,
        "relaxU": 0.3,
        "relaxPhi": 0.3,
        "relaxNuTilda": 0.3,
        "fpMaxIters": 2,
        "fpRelTol": 1e-5,
    },
    # Design variable setup
    "designVar": {
        "shape": {"designVarType": "FFD"},
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}

# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/bumpFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the cruise
        # scenario group
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        # configure and setup perform a similar function, i.e., initialize the optimization.
        # But configure will be run after setup

        # add the objective function to the cruise scenario
        self.cruise.aero_post.mphys_add_funcs()

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        nShapes = self.geometry.nom_addLocalDV(dvName="shape", axis="z")

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("cruise.aero_post.CD", scaler=1.0)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys.html")

# initialize the optimization function
optFuncs = OptFuncs(daOptions, prob)

prob.add_recorder(om.SqliteRecorder("cases.sql"))
prob.recording_options["includes"] = []
prob.recording_options["record_objectives"] = True
prob.recording_options["record_constraints"] = True
prob.recording_options["record_derivatives"] = True

prob.run_model()
totals = prob.compute_totals()
    
prob.record("final")
prob.cleanup()

cr = om.CaseReader("cases.sql")

# get list of cases recorded on problem
problem_cases = cr.list_cases("problem")
cr.list_source_vars("problem")
case = cr.get_case("final")

finalObj = case.get_objectives()

if gcomm.rank == 0:
    reg_write_dict(finalObj, 1e-8, 1e-10)
