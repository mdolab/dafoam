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
from funtofem.mphys import MeldThermalBuilder
from pygeo import geo_utils
from mphys.scenario_aerothermal import ScenarioAeroThermal
from pygeo.mphys import OM_DVGEOCOMP

gcomm = MPI.COMM_WORLD

os.chdir("./input/ChannelConjugateHeat")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")

# aero setup
U0 = 10.0
p0 = 101325.0
nuTilda0 = 1.0e-4
T0 = 300

daOptionsAero = {
    "designSurfaces": ["bot"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-7,
    "discipline": "aero",
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": False,
    },
    "objFunc": {
        "PL": {
            "part1": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["inlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            },
            "part2": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["outlet"],
                "scale": -1.0,
                "addToAdjoint": True,
            },
        },
    },
    "couplingInfo": {
        "aerothermal": {
            "active": True,
            "couplingSurfaceGroups": {
                "wallGroup": ["bot"],
            },
        }
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "nuTilda": nuTilda0 * 10.0,
        "T": T0,
        "phi": 1.0,
    },
    "designVar": {
        "shape": {"designVarType": "FFD"},
    },
}

daOptionsThermal = {
    "designSurfaces": ["top"],
    "solverName": "DALaplacianFoam",
    "primalMinResTol": 1.0e-8,
    "discipline": "thermal",
    "objFunc": {
        "HF": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["bot"],
                "scale": -0.001,
                "addToAdjoint": True,
            }
        },
    },
    "couplingInfo": {
        "aerothermal": {
            "active": True,
            "couplingSurfaceGroups": {
                "wallGroup": ["top"],
            },
        }
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "T": 300.0,
    },
    "designVar": {
        "shape": {"designVarType": "FFD"},
    },
}

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}


class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers for both cases (they share the same mesh option)
        dafoam_builder_aero = DAFoamBuilder(daOptionsAero, meshOptions, scenario="aerothermal", run_directory="aero")
        dafoam_builder_aero.initialize(self.comm)

        dafoam_builder_thermal = DAFoamBuilder(daOptionsThermal, meshOptions, scenario="aerothermal", run_directory="thermal")
        dafoam_builder_thermal.initialize(self.comm)

        thermalxfer_builder = MeldThermalBuilder(dafoam_builder_aero, dafoam_builder_thermal)
        thermalxfer_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh_aero", dafoam_builder_aero.get_mesh_coordinate_subsystem())
        self.add_subsystem("mesh_thermal", dafoam_builder_thermal.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD). Note that the aero and thermal use the exact same FFD file
        self.add_subsystem("geometry_aero", OM_DVGEOCOMP(file="aero/FFD/bumpFFD.xyz", type="ffd"))
        self.add_subsystem("geometry_thermal", OM_DVGEOCOMP(file="aero/FFD/bumpFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario(
            "scenario",
            ScenarioAeroThermal(
                aero_builder=dafoam_builder_aero,
                thermal_builder=dafoam_builder_thermal,
                thermalxfer_builder=thermalxfer_builder,
            ),
            om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-6, atol=1e-3),
            om.LinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-6, atol=1e-3),
        )

        # need to manually connect the x_aero0 between the mesh and geometry components
        self.connect("mesh_aero.x_aero0", "geometry_aero.x_aero_in")
        self.connect("geometry_aero.x_aero0", "scenario.x_aero")

        self.connect("mesh_thermal.x_thermal0", "geometry_thermal.x_thermal_in")
        self.connect("geometry_thermal.x_thermal0", "scenario.x_thermal")

    def configure(self):

        super().configure()

        # add the objective function to the cruise scenario
        self.scenario.aero_post.mphys_add_funcs()
        self.scenario.thermal_post.mphys_add_funcs()

        # get the surface coordinates from the mesh component
        points_aero = self.mesh_aero.mphys_get_surface_mesh()
        points_thermal = self.mesh_thermal.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry_aero.nom_add_discipline_coords("aero", points_aero)
        self.geometry_thermal.nom_add_discipline_coords("thermal", points_thermal)

        #self.scenario.coupling._mphys_promote_coupling_variables()

        # select the FFD points to move
        pts = self.geometry_aero.DVGeo.getLocalIndex(0)
        indexList = pts[3:5, :, 1].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry_aero.nom_addLocalDV(dvName="shape", axis="y", pointSelect=PS)
        nShapes = self.geometry_thermal.nom_addLocalDV(dvName="shape", axis="y", pointSelect=PS)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        # manually connect the dvs output to the geometry and cruise
        # sa and sst cases share the same shape
        self.connect("shape", "geometry_aero.shape")
        self.connect("shape", "geometry_thermal.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-0.01, upper=0.01, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("scenario.aero_post.PL", scaler=1.0)
        self.add_constraint("scenario.thermal_post.HF", lower=0.1, scaler=1.0)


prob = om.Problem(reports=None)
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.opt_settings = {
    "ACC": 1.0e-5,
    "MAXIT": 2,
    "IFILE": "opt_SLSQP.txt",
}
prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]


prob.add_recorder(om.SqliteRecorder("cases.sql"))
prob.recording_options["includes"] = []
prob.recording_options["record_objectives"] = True
prob.recording_options["record_constraints"] = True

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aerostruct.html")

optFuncs = OptFuncs([daOptionsAero, daOptionsThermal], prob)

prob.run_model()
totals = prob.compute_totals()

if gcomm.rank == 0:
    derivDict = {}
    derivDict["PL"] = {}
    derivDict["PL"]["shape"] = totals[("scenario.aero_post.functionals.PL", "dvs.shape")][0]
    derivDict["HF"] = {}
    derivDict["HF"]["shape"] = totals[("scenario.thermal_post.functionals.HF", "dvs.shape")][0]
    reg_write_dict(derivDict, 1e-4, 1e-6)
