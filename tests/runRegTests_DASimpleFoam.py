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

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/NACA0012")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor* *.bin")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0
p0 = 0.0
A0 = 0.1
twist0 = 3.0
LRef = 1.0
nuTilda0 = 4.5e-5

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
        "transport:nu": 1.5e-5,
    },
    "function": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "patchVelocityInputName": "patchV",
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "patchVelocityInputName": "patchV",
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "dynAdjustTol": False},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "solverInput": {
        "aero_vol_coords": {"type": "volCoord"},
        "patchV": {"type": "patchVelocity", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
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

        self.cruise.aero_post.mphys_add_funcs()

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # geometry setup

        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        shapes.append({pts[2, 1, 0]: dir_y, pts[2, 1, 1]: dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.zeros(1))
        self.dvs.add_output("patchV", val=np.array([10.0, 3.0]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")
        self.connect("patchV", "cruise.patchV")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("patchV", lower=-50.0, upper=50.0, scaler=1.0)

        # add constraints and the objective
        self.connect("cruise.aero_post.CD", "LoD.CD")
        self.connect("cruise.aero_post.CL", "LoD.CL")
        self.add_objective("LoD.val", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.3)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["LoD.val", "cruise.aero_post.CL"],
    wrt=["patchV", "shape"],
    compact_print=True,
    step=1e-3,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("cruise.aero_post.CD")
    funcDict["CL"] = prob.get_val("cruise.aero_post.CL")
    derivDict = {}
    derivDict["LoD"] = {}
    derivDict["LoD"]["shape"] = results[("LoD.val", "shape")]["J_fwd"][0]
    derivDict["LoD"]["patchV"] = results[("LoD.val", "patchV")]["J_fwd"][0]
    derivDict["CL"] = {}
    derivDict["CL"]["shape"] = results[("cruise.aero_post.CL", "shape")]["J_fwd"][0]
    derivDict["CL"]["patchV"] = results[("cruise.aero_post.CL", "patchV")]["J_fwd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
