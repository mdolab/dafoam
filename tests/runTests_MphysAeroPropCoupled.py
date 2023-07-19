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

os.chdir("./input/WingProp")

if gcomm.rank == 0:
    os.system("rm -rf */processor*")

# aero setup
daOptionsWing = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-10,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [100.0, 0.0, 0.0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "drag": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "lift": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 1.0, 0.0],
                "scale": 1.0,
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
        "U": 100,
        "p": 101325,
        "T": 300,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "designVar": {
        "twist": {"designVarType": "FFD"},
        "fvSource": {"designVarType": "Field", "fieldName": "fvSource", "fieldType": "vector"},
    },
    "wingProp": {
        "prop1": {
            "active": True,
            "nForceSections": 5,
            "axis": [0.0, 0.0, 1.0],
            "rotationCenter": [0.0, 0.0, 0.0],
            "actEps": 0.2,
            "rotDir": "right",
            "interpScheme": "gauss",
            "bladeName": "blade",
        },
        "prop2": {
            "active": True,
            "nForceSections": 5,
            "axis": [0.0, 0.0, 1.0],
            "rotationCenter": [0.0, 0.0, 0.0],
            "actEps": 0.2,
            "rotDir": "left",
            "interpScheme": "gauss",
            "bladeName": "blade",
        },
    },
}

meshOptionsWing = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": False,
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}

daOptionsProp = {
    "solverName": "DARhoSimpleFoam",
    "designSurfaces": ["blade"],
    "primalMinResTol": 1e-12,
    "primalBC": {
        "MRF": -500.0,
    },
    "objFunc": {
        "thrust": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["blade"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 0.0, 1.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "power": {
            "part1": {
                "type": "power",
                "source": "patchToFace",
                "patches": ["blade"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "scale": -2.0 / 1000.0,  # kW
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"U": 10, "p": 50.0, "nuTilda": 1e-3, "phi": 1.0},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "shapey": {"designVarType": "FFD"},
    },
    "decomposeParDict": {"preservePatches": ["per1", "per2"]},
    "wingProp": {
        "prop": {
            "active": True,
            "nForceSections": 5,
            "axis": [0.0, 0.0, 1.0],
            "rotationCenter": [0.0, 0.0, 0.0],
            "actEps": 0.2,
            "rotDir": "right",
            "interpScheme": "gauss",
            "bladeName": "blade",
        },
    },
}

meshOptionsProp = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder_wing = DAFoamBuilder(
            daOptionsWing, meshOptionsWing, scenario="aerodynamic", prop_coupling="Wing", run_directory="Wing"
        )
        dafoam_builder_wing.initialize(self.comm)

        dafoam_builder_prop = DAFoamBuilder(
            daOptionsProp, meshOptionsProp, scenario="aerodynamic", prop_coupling="Prop", run_directory="Prop"
        )
        dafoam_builder_prop.initialize(self.comm)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # create the mesh and cruise scenario because we only have one analysis point
        self.add_subsystem("mesh_wing", dafoam_builder_wing.get_mesh_coordinate_subsystem())
        self.add_subsystem("mesh_prop", dafoam_builder_prop.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry_wing", OM_DVGEOCOMP(file="Wing/FFD/wingFFD.xyz", type="ffd"))
        self.add_subsystem("geometry_prop", OM_DVGEOCOMP(file="Prop/FFD/localFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise_prop", ScenarioAerodynamic(aero_builder=dafoam_builder_prop))
        self.mphys_add_scenario("cruise_wing", ScenarioAerodynamic(aero_builder=dafoam_builder_wing))

        self.connect("mesh_wing.x_aero0", "geometry_wing.x_aero_in")
        self.connect("geometry_wing.x_aero0", "cruise_wing.x_aero")
        self.connect("mesh_prop.x_aero0", "geometry_prop.x_aero_in")
        self.connect("geometry_prop.x_aero0", "cruise_prop.x_aero")

        # add an exec comp to average two drags
        self.add_subsystem("force_balance", om.ExecComp("value=2*thrust-drag"))

    def configure(self):

        self.cruise_wing.aero_post.mphys_add_funcs()
        self.cruise_prop.aero_post.mphys_add_funcs()

        # create geometric DV setup
        points_wing = self.mesh_wing.mphys_get_surface_mesh()
        points_prop = self.mesh_prop.mphys_get_surface_mesh()

        # add pointset
        self.geometry_wing.nom_add_discipline_coords("aero", points_wing)
        self.geometry_prop.nom_add_discipline_coords("aero", points_prop)

        # geometry setup

        # WING
        def fvSource(val, DASolver):
            for idxI, v in enumerate(val):
                cellI = idxI // 3
                compI = idxI % 3
                DASolver.setFieldValue4LocalCellI(b"fvSource", v, cellI, compI)
                # DASolver.updateBoundaryConditions(b"fvSource", b"vector")

        self.cruise_wing.coupling.solver.add_dv_func("fvSource", fvSource)
        # no need to give fvSource to aero_post because we don't need its derivs
        # self.cruise.aero_post.add_dv_func("fvSource", fvSource)

        nRefAxPtsWing = self.geometry_wing.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Select all points
        def twist_wing(val, geo):
            for i in range(nRefAxPtsWing):
                geo.rot_z["wingAxis"].coef[i] = -val[0]

        self.geometry_wing.nom_addGlobalDV(dvName="twist_wing", value=np.array([2]), func=twist_wing)

        # PROP
        nShapesProp = self.geometry_prop.nom_addLocalDV(dvName="shape_prop")

        # add dvs to ivc and connect
        self.dvs.add_output("twist_wing", val=np.array([2]))
        self.dvs.add_output("shape_prop", val=np.array([0] * nShapesProp))
        self.dvs.add_output("prop1_center", val=np.array([-0.2, 0.2, 0.05]))
        self.dvs.add_output("prop2_center", val=np.array([-0.2, -0.2, 0.05]))

        for i in [1, 2]:
            self.connect("cruise_prop.axial_force", "cruise_wing.prop%d_axial_force" % i)
            self.connect("cruise_prop.tangential_force", "cruise_wing.prop%d_tangential_force" % i)
            self.connect("cruise_prop.radial_location", "cruise_wing.prop%d_radial_location" % i)
            self.connect("cruise_prop.integral_force", "cruise_wing.prop%d_integral_force" % i)
            self.connect("prop%d_center" % i, "cruise_wing.prop%d_prop_center" % i)

        self.connect("twist_wing", "geometry_wing.twist_wing")
        self.connect("shape_prop", "geometry_prop.shape_prop")

        # define the design variables
        self.add_design_var("twist_wing", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("shape_prop", lower=-0.1, upper=0.1, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise_prop.aero_post.power", scaler=1.0)
        self.add_constraint("cruise_wing.aero_post.lift", equals=130, scaler=1.0)

        self.connect("cruise_wing.aero_post.drag", "force_balance.drag")
        self.connect("cruise_prop.aero_post.thrust", "force_balance.thrust")
        self.add_constraint("force_balance.value", equals=0, scaler=1.0)


prob = om.Problem(reports=None)
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_wing_prop.html")

prob.run_model()
totals = prob.compute_totals()

if gcomm.rank == 0:
    objFuncDict = {}
    objFuncDict["power"] = prob.get_val("cruise_prop.aero_post.power")
    objFuncDict["lift"] = prob.get_val("cruise_wing.aero_post.lift")
    objFuncDict["drag"] = prob.get_val("cruise_wing.aero_post.drag")
    objFuncDict["thrust"] = prob.get_val("cruise_prop.aero_post.thrust")
    derivDict = {}
    derivDict["force_balance"] = {}
    derivDict["force_balance"]["prop_shape"] = totals[("force_balance.value", "dvs.shape_prop")][0]
    derivDict["force_balance"]["wing_twist"] = totals[("force_balance.value", "dvs.twist_wing")][0]
    derivDict["lift"] = {}
    derivDict["lift"]["prop_shape"] = totals[("cruise_wing.aero_post.functionals.lift", "dvs.shape_prop")][0]
    derivDict["lift"]["wing_twist"] = totals[("cruise_wing.aero_post.functionals.lift", "dvs.twist_wing")][0]
    derivDict["power"] = {}
    derivDict["power"]["prop_shape"] = totals[("cruise_prop.aero_post.functionals.power", "dvs.shape_prop")][0]
    derivDict["power"]["wing_twist"] = totals[("cruise_prop.aero_post.functionals.power", "dvs.twist_wing")][0]
    reg_write_dict(objFuncDict, 1e-8, 1e-10)
    reg_write_dict(derivDict, 1e-4, 1e-6)
