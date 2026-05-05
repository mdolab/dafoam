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

os.chdir("./reg_test_files-main/PlateHoleV4")
if gcomm.rank == 0:
    os.system("rm -rf processor*")

daOptions = {
    "designSurfaces": ["hole"],
    "solverName": "DASolidDisplacementFoam",
    "primalMinResTol": 1e-10,
    "primalMinResTolDiff": 1e10,
    "maxCorrectBCCalls": 20,
    "function": {
        "VMS": {
            "type": "vonMisesStressKS",
            "source": "allCells",
            "scale": 1.0,
            "coeffKS": 2.0e-3,
        },
        "M": {
            "type": "variableVolSum",
            "source": "allCells",
            "varName": "solid:rho",
            "varType": "scalar",
            "index": 0,
            "scale": 1.0,
        },
    },
    "normalizeStates": {"D": 1.0e-7},
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "gmresAbsTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/plateFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # geometry setup

        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        shapes.append({pts[1, 0, 0]: dir_y, pts[1, 0, 1]: dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.zeros(1))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.VMS", scaler=1.0)
        self.add_constraint("cruise.aero_post.M", scaler=1.0, equals=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["cruise.aero_post.VMS", "cruise.aero_post.M"],
    wrt=["shape"],
    compact_print=True,
    step=1e-3,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["VMS"] = prob.get_val("cruise.aero_post.VMS")
    funcDict["M"] = prob.get_val("cruise.aero_post.M")
    derivDict = {}
    derivDict["VMS"] = {}
    derivDict["VMS"]["shape-Adjoint"] = results[("cruise.aero_post.VMS", "shape")]["J_fwd"][0]
    derivDict["VMS"]["shape-FD"] = results[("cruise.aero_post.VMS", "shape")]["J_fd"][0]
    derivDict["M"] = {}
    derivDict["M"]["shape-Adjoint"] = results[("cruise.aero_post.M", "shape")]["J_fwd"][0]
    derivDict["M"]["shape-FD"] = results[("cruise.aero_post.M", "shape")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
