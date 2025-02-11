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

os.chdir("./reg_test_files-main/UBendDuct")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor* *.bin")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 1.0
p0 = 0.0
nuTilda0 = 1.0e-3

daOptions = {
    "designSurfaces": ["ubend"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "useConstrainHbyA": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": False,
        "transport:nu": 1.0e-3,
    },
    "function": {
        "TP1": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["inlet"],
            "scale": 1.0 / (0.5 * U0 * U0),
        },
        "TP2": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["outlet"],
            "scale": 1.0 / (0.5 * U0 * U0),
        },
        "HFX": {
            "type": "wallHeatFlux",
            "source": "patchToFace",
            "patches": ["ubend"],
            "scale": 1.0,
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/UBendDuctFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

        self.add_subsystem("PL", om.ExecComp("val=TP1-TP2"))

    def configure(self):

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # geometry setup

        # Select all points
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[10, 1, 0].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS, axis="z")

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.zeros(1))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.connect("cruise.aero_post.TP1", "PL.TP1")
        self.connect("cruise.aero_post.TP2", "PL.TP2")
        self.add_objective("PL.val", scaler=1.0)
        self.add_constraint("cruise.aero_post.HFX", lower=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["PL.val", "cruise.aero_post.HFX"],
    wrt=["shape"],
    compact_print=True,
    step=1e-4,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["PL"] = prob.get_val("PL.val")
    funcDict["HFX"] = prob.get_val("cruise.aero_post.HFX")
    derivDict = {}
    derivDict["PL"] = {}
    derivDict["PL"]["shape-Adjoint"] = results[("PL.val", "shape")]["J_fwd"][0]
    derivDict["PL"]["shape-FD"] = results[("PL.val", "shape")]["J_fd"][0]
    derivDict["HFX"] = {}
    derivDict["HFX"]["shape-Adjoint"] = results[("cruise.aero_post.HFX", "shape")]["J_fwd"][0]
    derivDict["HFX"]["shape-FD"] = results[("cruise.aero_post.HFX", "shape")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
