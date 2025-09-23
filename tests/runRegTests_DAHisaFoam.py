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
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.hisa/* 0/")
    os.system("cp -r system.hisa/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    # replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 100.0
p0 = 101325.0
T0 = 300.0

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DAHisaFoam",
    "printDAOptions": False,
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "primalInitCondition": {"U": [U0, 0.0, 0.0], "p": p0, "T": T0},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet", "outlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inlet", "outlet"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inlet", "outlet"], "value": [T0]},
        "useWallFunction": False,
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
    },
    "adjEqnOption": {"hisaForceJSTFlux": 0, "gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": 101325, "T": 300.0, "nuTilda": 1e-3},
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
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)


# ******* test JST *******
prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["cruise.aero_post.CD"],
    wrt=["shape"],
    compact_print=True,
    step=1e-2,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD_JST"] = prob.get_val("cruise.aero_post.CD")
    derivDict = {}
    derivDict["CD_JST"] = {}
    derivDict["CD_JST"]["shape-Adjoint"] = results[("cruise.aero_post.CD", "shape")]["J_fwd"][0]
    derivDict["CD_JST"]["shape-FD"] = results[("cruise.aero_post.CD", "shape")]["J_fd"][0]


# ******* test AUSMPlusUp *******
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.hisa/* 0/")
    os.system("cp -r system.hisa/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")
    replace_text_in_file("system/fvSchemes", "fluxScheme           JST;", "fluxScheme           AUSMPlusUp;")


daOptions["adjEqnOption"]["hisaForceJSTFlux"] = 0

prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["cruise.aero_post.CD"],
    wrt=["shape"],
    compact_print=True,
    step=1e-2,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict["CD_AUSM"] = prob.get_val("cruise.aero_post.CD")
    derivDict["CD_AUSM"] = {}
    derivDict["CD_AUSM"]["shape-Adjoint"] = results[("cruise.aero_post.CD", "shape")]["J_fwd"][0]
    derivDict["CD_AUSM"]["shape-FD"] = results[("cruise.aero_post.CD", "shape")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
