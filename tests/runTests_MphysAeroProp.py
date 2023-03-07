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

os.chdir("./input/CurvedCubeSnappyHexMesh")

if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.compressible 0")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 50.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
rho0 = p0 / T0 / 287.0
A0 = 1.0

daOptions = {
    "designSurfaces": ["wallsbump"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-10,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [T0]},
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
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wallsbump"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 1.0, 0.0],
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
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
        "p": p0,
        "T": T0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-6},
    "designVar": {
        "shape": {"designVarType": "FFD"},
        "fvSource": {"designVarType": "Field", "fieldName": "fvSource", "fieldType": "vector"},
    },
    "wingProp": {"nForceSections": 10, "axis": [1.0, 0.0, 0.0], "actEps": 0.2, "rotDir": "right"},
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic", prop_coupling="Wing")
        dafoam_builder.initialize(self.comm)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # create the mesh and cruise scenario because we only have one analysis point
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/bumpFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):

        self.cruise.aero_post.mphys_add_funcs()

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # geometry setup

        def fvSource(val, DASolver):
            for idxI, v in enumerate(val):
                cellI = idxI // 3
                compI = idxI % 3
                DASolver.setFieldValue4LocalCellI(b"fvSource", v, cellI, compI)

        self.cruise.coupling.solver.add_dv_func("fvSource", fvSource)
        # no need to give fvSource to aero_post because we don't need its derivs
        # self.cruise.aero_post.add_dv_func("fvSource", fvSource)

        # Select all points
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS)

        # add dvs to ivc and connect
        self.dvs.add_output("shape", val=np.array([0] * nShapes))

        axial_force = np.array([0.1, 0.2, 0.3, 0.4, 0.48, 0.54, 0.60, 0.62, 0.63, 0.4])
        tangential_force = np.array([0.1, 0.2, 0.3, 0.4, 0.48, 0.54, 0.60, 0.62, 0.63, 0.4])
        radial_distance = np.array([0.1, 0.135, 0.205, 0.275, 0.345, 0.415, 0.485, 0.555, 0.625, 0.695, 0.765, 0.8])
        prop_center = np.array([0.075, 0.025, 0.025])
        integral_force = np.array([2, 1])

        self.dvs.add_output("axial_force", val=axial_force)
        self.dvs.add_output("tangential_force", val=tangential_force)
        self.dvs.add_output("radial_distance", val=radial_distance)
        self.dvs.add_output("prop_center", val=prop_center)
        self.dvs.add_output("integral_force", val=integral_force)

        self.connect("shape", "geometry.shape")
        self.connect("axial_force", "cruise.axial_force")
        self.connect("tangential_force", "cruise.tangential_force")
        self.connect("radial_distance", "cruise.radial_distance")
        self.connect("prop_center", "cruise.prop_center")
        self.connect("integral_force", "cruise.integral_force")

        # define the design variables
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        # self.add_design_var("prop_center", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.3, scaler=1.0)


prob = om.Problem(reports=None)
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "IPOPT"
prob.driver.opt_settings = {
    "tol": 1.0e-7,
    "max_iter": 50,
    "output_file": "opt_IPOPT.out",
    "constr_viol_tol": 1.0e-7,
    "mu_strategy": "adaptive",
    "limited_memory_max_history": 26,
    "nlp_scaling_method": "gradient-based",
    "alpha_for_y": "full",
    "recalc_y": "yes",
    "print_level": 5,
    "acceptable_tol": 1.0e-7,
}
prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]


prob.add_recorder(om.SqliteRecorder("cases.sql"))
prob.recording_options["includes"] = []
prob.recording_options["record_objectives"] = True
prob.recording_options["record_constraints"] = True

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

optFuncs = OptFuncs(daOptions, prob)

prob.run_model()

if gcomm.rank == 0:
    derivDict = {}
    CD = prob.get_val("cruise.aero_post.CD")[0]
    CL = prob.get_val("cruise.aero_post.CL")[0]
    derivDict["CD"] = CD
    derivDict["CL"] = CL
    reg_write_dict(derivDict, 1e-4, 1e-6)
