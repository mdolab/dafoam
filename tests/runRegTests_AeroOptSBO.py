#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

import os
import argparse
import numpy as np
import json
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from testFuncs import *
from pygeo import geo_utils
from dafoam.pyDAFoam import surrogateOptimization

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/NACA0012V4")
if gcomm.rank == 0:
    os.system("rm -rf 0 system processor* *.bin")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    # replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0
p0 = 0.0
A0 = 0.1
twist0 = 3.25
LRef = 1.0
nuTilda0 = 4.5e-5

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-10,
    "primalMinResTolDiff": 1e4,
    "writeDeformedFFDs": True,
    "writeDeformedConstraints": True,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
        "transport:nu": 1.5e-5,
    },
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
        "skewness": {
            "type": "meshQualityKS",
            "source": "allCells",
            "coeffKS": 10.0,
            "metric": "faceSkewness",
            "scale": 1.0,
        },
        "nonOrtho": {
            "type": "meshQualityKS",
            "source": "allCells",
            "coeffKS": 1.0,
            "metric": "nonOrthoAngle",
            "scale": 1.0,
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inout"],
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
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
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

        self.add_subsystem("LoD", om.ExecComp("val=CL/CD"))

    def configure(self):

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # create constraint DV setup
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # add the dv_geo object to the builder solver. This will be used to write deformed FFDs
        self.cruise.coupling.solver.add_dvgeo(self.geometry.DVGeo)

        # add the dv_con object to the builder solver. This will be used to write deformed constraints
        self.cruise.coupling.solver.add_dvcon(self.geometry.DVCon)

        # geometry setup

        # Select all points
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.zeros(nShapes))
        self.dvs.add_output("patchV", val=np.array([10.0, 3.0]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")
        self.connect("patchV", "cruise.patchV")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=0.01)
        self.add_design_var("patchV", lower=-50.0, upper=50.0, scaler=1.0, indices=[1])

        # add constraints and the objective
        self.connect("cruise.aero_post.CD", "LoD.CD")
        self.connect("cruise.aero_post.CL", "LoD.CL")
        self.add_objective("LoD.val", scaler=1.0)


# openmdao setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")

# surrogate optimization setup
xlimits = np.array([[0, 0.1]] * 22)  # DV bounds
xlimits[-1] = [3.0, 3.3]  # adjust DV bounds
xlimits[-2] = [10, 10 + 1e-9]
xlimits[0:4] = [0.0, 0.0 + 1e-9]
xlimits[4:8] = [0.03, 0.04]
xlimits[8:10] = [0.001, 0.002]
xlimits[10:12] = [0.004, 0.005]
xlimits[12:16] = [-0.016, 0.0]
xlimits[16:20] = [0.0, 0.0 + 1e-9]

surrogateOptions = {
    "optType": "constrained",  # type of optimization problem (constrained or unconstrained)
    "criterion": "EI",  # criterion for next evaluation point determination -> EGO algorithm
    "iters": 1,  # num iterations to optimize function
    "numDOE": 2,  # number of sampling points
    "seed": 41,  # seed value to reproduce results
    "dvNames": ["shape", "patchV"],  # names of design variables
    "dvSizes": [20, 2],  # number of points for each design variable
    "dvBounds": xlimits,  # design variable bounds
    "objFunc": "LoD.val",  # objective function
    "maxObj": True,  # maximize lift over drag
    "cons": ["cruise.aero_post.CL"],  # quantity to constrain
    "conWeights": [1e8],  # constraint weight
    "consEqs": ["x - 0.3"],  # constraint equation(s)
}

surrogateOptimization(surrogateOptions, prob)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("cruise.aero_post.CD")
    funcDict["CL"] = prob.get_val("cruise.aero_post.CL")
    reg_write_dict(funcDict, 1e-5, 1e-10)
