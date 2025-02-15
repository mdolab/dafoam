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

os.chdir("./reg_test_files-main/NACA0012V4")
if gcomm.rank == 0:
    os.system("rm -rf 0 system processor* *.bin")
    os.system("cp -r 0.compressible 0")
    os.system("cp -r system.transonic system")
    os.system("cp -r constant/turbulenceProperties.sst constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 240.0
p0 = 101325.0
T0 = 300.0
A0 = 0.1
twist0 = 3.0
LRef = 1.0
nuTilda0 = 4.5e-5

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-11,
    "primalMinResTolDiff": 1e4,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
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
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-11, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "adjStateOrdering": "cell",
    "transonicPCOption": 1,
    "normalizeStates": {"U": U0, "p": p0, "phi": 1.0, "T": T0, "nuTilda": 1e-3},
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

        # geometry setup
        self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Set up global design variables. We dont change the root twist
        def twist(val, geo):
            for i in range(2):
                geo.rot_z["wingAxis"].coef[i] = -val[0]

        self.geometry.nom_addGlobalDV(dvName="twist", value=np.ones(1) * twist0, func=twist)

        # add the design variables to the dvs component's output
        self.dvs.add_output("twist", val=np.ones(1) * twist0)
        self.dvs.add_output("patchV", val=np.array([240.0, 0.0]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("twist", "geometry.twist")
        self.connect("patchV", "cruise.patchV")

        # define the design variables to the top level
        self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("patchV", lower=-50.0, upper=50.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.3)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["cruise.aero_post.CD", "cruise.aero_post.CL"],
    wrt=["twist", "patchV"],
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
    derivDict["CD"] = {}
    derivDict["CD"]["twist-Adjoint"] = results[("cruise.aero_post.CD", "twist")]["J_fwd"][0]
    derivDict["CD"]["twist-FD"] = results[("cruise.aero_post.CD", "twist")]["J_fd"][0]
    derivDict["CD"]["patchV-Adjoint"] = results[("cruise.aero_post.CD", "patchV")]["J_fwd"][0]
    derivDict["CD"]["patchV-FD"] = results[("cruise.aero_post.CD", "patchV")]["J_fd"][0]
    derivDict["CL"] = {}
    derivDict["CL"]["twist-Adjoint"] = results[("cruise.aero_post.CL", "twist")]["J_fwd"][0]
    derivDict["CL"]["twist-FD"] = results[("cruise.aero_post.CL", "twist")]["J_fd"][0]
    derivDict["CL"]["patchV-Adjoint"] = results[("cruise.aero_post.CL", "patchV")]["J_fwd"][0]
    derivDict["CL"]["patchV-FD"] = results[("cruise.aero_post.CL", "patchV")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
