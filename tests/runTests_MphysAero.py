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

os.chdir("./input/NACA0012")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.safv3 constant/turbulenceProperties")

# aero setup
U0 = 10.0
p0 = 0.0
A0 = 0.1
aoa0 = 3.0
LRef = 1.0

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-10,
    "primalMinResTolDiff": 1e4,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "useWallFunction": True,
        "transport:nu": 1.5e-5,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "aoa",
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
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
    },
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusSmooth",
            "center": [-0.55, 0.0, 0.05],
            "direction": [1.0, 0.0, 0.0],
            "innerRadius": 0.01,
            "outerRadius": 0.4,
            "rotDir": "right",
            "scale": 100.0,
            "POD": 0.0,
            "eps": 0.1,  # eps should be of cell size
            "expM": 1.0,
            "expN": 0.5,
            "adjustThrust": 0,
            "targetThrust": 1.0,
        },
    },
    "adjEqnOption": {
        "gmresRelTol": 1.0e-8,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
    },
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "adjPartDerivFDStep": {"State": 1e-6},
    "designVar": {
        "shape": {"designVarType": "FFD"},
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
        "actuator_radius": {"designVarType": "ACTD", "actuatorName": "disk1", "comps": [4]},
        "actuator_center": {"designVarType": "ACTD", "actuatorName": "disk1", "comps": [0, 1, 2]},
        "uin": {"designVarType": "BC", "patches": ["inout"], "variable": "U", "comp": 0},
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

        # Create reference axis
        def aoa(val, DASolver):
            aoa = val[0] * np.pi / 180.0
            U = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
            # we need to update the U value only
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()

        # pass this aoa function to the cruise group
        self.cruise.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise.aero_post.add_dv_func("aoa", aoa)

        def actuator_radius(val, DASolver):
            actR2 = float(val[0])
            # only change design variables
            DASolver.setOption("fvSource", {"disk1": {"outerRadius": actR2}})
            DASolver.updateDAOption()

        self.cruise.coupling.solver.add_dv_func("actuator_radius", actuator_radius)
        self.cruise.aero_post.add_dv_func("actuator_radius", actuator_radius)

        def actuator_center(val, DASolver):
            actX = float(val[0])
            actY = float(val[1])
            actZ = float(val[2])
            # only change design variables
            DASolver.setOption("fvSource", {"disk1": {"center": [actX, actY, actZ]}})
            DASolver.updateDAOption()

        self.cruise.coupling.solver.add_dv_func("actuator_center", actuator_center)
        self.cruise.aero_post.add_dv_func("actuator_center", actuator_center)

        def uin(val, DASolver):
            U = [float(val[0]), 0.0, 0.0]
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()
        
        self.cruise.coupling.solver.add_dv_func("uin", uin)
        self.cruise.aero_post.add_dv_func("uin", uin)

        # Select all points
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS)

        # add dvs to ivc and connect
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("aoa", val=np.array([aoa0]))
        self.dvs.add_output("uin", val=np.array([U0]))
        self.dvs.add_output("actuator_radius", val=np.array([0.4]))
        self.dvs.add_output("actuator_center", val=np.array([-0.55, 0, 0.05]))

        self.connect("shape", "geometry.shape")
        self.connect("aoa", "cruise.aoa")
        self.connect("uin", "cruise.uin")
        self.connect("actuator_radius", "cruise.actuator_radius")
        self.connect("actuator_center", "cruise.actuator_center")

        # define the design variables
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        self.add_design_var("aoa", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("uin", lower=9.0, upper=11.0, scaler=1.0)
        self.add_design_var("actuator_radius", lower=-1000.0, upper=1000.0, scaler=1.0)
        self.add_design_var("actuator_center", lower=-1000.0, upper=1000.0, scaler=1.0)

        # add constraints and the objective
        self.connect("cruise.aero_post.CD", "LoD.CD")
        self.connect("cruise.aero_post.CL", "LoD.CL")
        self.add_objective("LoD.val", scaler=1.0)


prob = om.Problem()
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
totals = prob.compute_totals()

if gcomm.rank == 0:
    derivDict = {}
    derivDict["LoD.val"] = {}
    derivDict["LoD.val"]["shape"] = totals[("LoD.val", "dvs.shape")][0]
    derivDict["LoD.val"]["aoa"] = totals[("LoD.val", "dvs.aoa")][0]
    derivDict["LoD.val"]["uin"] = totals[("LoD.val", "dvs.uin")][0]
    derivDict["LoD.val"]["actuator_radius"] = totals[("LoD.val", "dvs.actuator_radius")][0]
    derivDict["LoD.val"]["actuator_center"] = totals[("LoD.val", "dvs.actuator_center")][0]
    reg_write_dict(derivDict, 1e-4, 1e-6)
