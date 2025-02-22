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

os.chdir("./reg_test_files-main/NACA0012V4")
if gcomm.rank == 0:
    os.system("rm -rf 0 system processor* *.bin")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

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

        # setup the symmetry constraint to link the y displacement between k=0 and k=1
        nFFDs_x = pts.shape[0]
        nFFDs_y = pts.shape[1]
        indSetA = []
        indSetB = []
        for i in range(nFFDs_x):
            for j in range(nFFDs_y):
                indSetA.append(pts[i, j, 0])
                indSetB.append(pts[i, j, 1])
        self.geometry.nom_addLinearConstraintsShape("linearcon", indSetA, indSetB, factorA=1.0, factorB=-1.0)

        # setup the volume and thickness constraints
        leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
        teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=2, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=2, nChord=10)
        # add the LE/TE constraints
        self.geometry.nom_add_LETEConstraint("lecon", volID=0, faceID="iLow", topID="k")
        self.geometry.nom_add_LETEConstraint("tecon", volID=0, faceID="iHigh", topID="k")

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
        self.add_objective("LoD.val", scaler=-1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.3)
        self.add_constraint("cruise.aero_post.skewness", upper=4.0)
        self.add_constraint("cruise.aero_post.nonOrtho", upper=70.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.linearcon", equals=0.0, scaler=1.0, linear=True)


prob = om.Problem()
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "IPOPT"
prob.driver.opt_settings = {
    "tol": 1.0e-5,
    "constr_viol_tol": 1.0e-5,
    "max_iter": 2,
    "print_level": 5,
    "mu_strategy": "adaptive",
    "limited_memory_max_history": 10,
    "nlp_scaling_method": "none",
    "alpha_for_y": "full",
    "recalc_y": "yes",
}
prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

optFuncs = OptFuncs(daOptions, prob)

optFuncs.findFeasibleDesign(["cruise.aero_post.CL"], ["patchV"], designVarsComp=[1], targets=[0.3])

prob.run_driver()

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("cruise.aero_post.CD")
    funcDict["CL"] = prob.get_val("cruise.aero_post.CL")
    reg_write_dict(funcDict, 1e-5, 1e-10)
