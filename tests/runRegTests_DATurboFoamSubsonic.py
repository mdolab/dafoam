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

os.chdir("./reg_test_files-main/CompressorFluid")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor* *.bin")
    os.system("cp -r 0.compressible 0")
    os.system("cp -r 0/U.subsonic 0/U")
    os.system("cp -r 0/p.subsonic 0/p")
    os.system("cp -r constant/thermophysicalProperties.e constant/thermophysicalProperties")
    os.system("cp -r constant/MRFProperties.subsonic constant/MRFProperties")
    os.system("cp -r system/fvSolution.subsonic system/fvSolution")
    os.system("cp -r system/fvSchemes.subsonic system/fvSchemes")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

daOptions = {
    "designSurfaces": ["blade"],
    "solverName": "DATurboFoam",
    "primalMinResTol": 1.0e-11,
    "primalMinResTolDiff": 1e4,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [0.0, 0.0, 100.0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [300.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [101325.0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [1e-4]},
        "useWallFunction": True,
    },
    "function": {
        "TPR": {
            "type": "totalPressureRatio",
            "source": "patchToFace",
            "patches": ["inlet", "outlet"],
            "inletPatches": ["inlet"],
            "outletPatches": ["outlet"],
            "scale": 1.0,
        },
        "CMZ": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["blade"],
            "axis": [0.0, 0.0, 1.0],
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0 / (0.5 * 10.0 * 10.0 * 1.0 * 1.0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-11, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": 100.0, "p": 100000.0, "nuTilda": 1e-3, "phi": 1.0, "T": 300.0},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
    },
    "decomposeParDict": {"preservePatches": ["per1", "per2"]},
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/localFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # Select all points
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[1, 1, 1].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS, axis="y")

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0.0]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.TPR", scaler=1.0)
        self.add_constraint("cruise.aero_post.CMZ", equals=0.3)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["cruise.aero_post.TPR", "cruise.aero_post.CMZ"],
    wrt=["shape"],
    compact_print=True,
    step=1e-4,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["TPR"] = prob.get_val("cruise.aero_post.TPR")
    funcDict["CMZ"] = prob.get_val("cruise.aero_post.CMZ")
    derivDict = {}
    derivDict["TPR"] = {}
    derivDict["TPR"]["shape-Adjoint"] = results[("cruise.aero_post.TPR", "shape")]["J_fwd"][0]
    derivDict["TPR"]["shape-FD"] = results[("cruise.aero_post.TPR", "shape")]["J_fd"][0]
    derivDict["CMZ"] = {}
    derivDict["CMZ"]["shape-Adjoint"] = results[("cruise.aero_post.CMZ", "shape")]["J_fwd"][0]
    derivDict["CMZ"]["shape-FD"] = results[("cruise.aero_post.CMZ", "shape")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
