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

daOptionsAero = {
    "designSurfaces": [
        "hot_air_inner",
        "hot_air_outer",
        "hot_air_sides",
        "cold_air_outer",
        "cold_air_inner",
        "cold_air_sides",
    ],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-17,
    "primalMinResTolDiff": 1.0e17,
    "discipline": "aero",
    "primalBC": {
        "UHot": {"variable": "U", "patches": ["hot_air_in"], "value": [U0, 0.0, 0.0]},
        "UCold": {"variable": "U", "patches": ["cold_air_in"], "value": [-U0, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "objFunc": {
        "PL": {
            "part1": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["hot_air_in"],
                "scale": 1.0,
                "addToAdjoint": True,
            },
            "part2": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["hot_air_out"],
                "scale": -1.0,
                "addToAdjoint": True,
            },
            "part3": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["cold_air_in"],
                "scale": 1.0,
                "addToAdjoint": True,
            },
            "part4": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["cold_air_out"],
                "scale": -1.0,
                "addToAdjoint": True,
            },
        },
        "TOUT": {
            "part1": {
                "type": "patchMean",
                "source": "patchToFace",
                "patches": ["hot_air_out"],
                "varName": "T",
                "varType": "scalar",
                "component": 0,
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "HFH": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["hot_air_inner"],
                "scale": 1,
                "addToAdjoint": True,
            }
        },
        "HFC": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["cold_air_outer"],
                "scale": 1,
                "addToAdjoint": True,
            }
        },
    },
    "couplingInfo": {
        "aerothermal": {
            "active": True,
            "couplingSurfaceGroups": {
                "wallGroup": ["hot_air_inner", "cold_air_outer"],
            },
        }
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-3,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": 101325,
        "nuTilda": 1e-3,
        "T": 300,
        "phi": 1.0,
    },
    "designVar": {
        "shape": {"designVarType": "FFD"},
    },
}

daOptionsThermal = {
    "designSurfaces": ["channel_outer", "channel_inner", "channel_sides"],
    "solverName": "DAHeatTransferFoam",
    "primalMinResTol": 1.0e-17,
    "primalMinResTolDiff": 1.0e17,
    "discipline": "thermal",
    "objFunc": {
        "HF_INNER": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["channel_inner"],
                "scale": 1,
                "addToAdjoint": True,
            }
        },
        "HF_OUTER": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["channel_outer"],
                "scale": 1,
                "addToAdjoint": True,
            }
        },
    },
    "couplingInfo": {
        "aerothermal": {
            "active": True,
            "couplingSurfaceGroups": {
                "wallGroup": ["channel_outer", "channel_inner"],
            },
        }
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-3,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
    },
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

        dafoam_builder_thermal = DAFoamBuilder(
            daOptionsThermal, meshOptions, scenario="aerothermal", run_directory="thermal"
        )
        dafoam_builder_thermal.initialize(self.comm)

        thermalxfer_builder = MeldThermalBuilder(dafoam_builder_aero, dafoam_builder_thermal, n=1, beta=0.5)
        thermalxfer_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh_aero", dafoam_builder_aero.get_mesh_coordinate_subsystem())
        self.add_subsystem("mesh_thermal", dafoam_builder_thermal.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD). Note that the aero and thermal use the exact same FFD file
        self.add_subsystem("geometry_aero", OM_DVGEOCOMP(file="aero/FFD/channelFFD.xyz", type="ffd"))
        self.add_subsystem("geometry_thermal", OM_DVGEOCOMP(file="aero/FFD/channelFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario(
            "scenario",
            ScenarioAeroThermal(
                aero_builder=dafoam_builder_aero,
                thermal_builder=dafoam_builder_thermal,
                thermalxfer_builder=thermalxfer_builder,
            ),
            om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-10, atol=1e-6),
            om.LinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-4),
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

        # self.scenario.coupling._mphys_promote_coupling_variables()

        # select the FFD points to move
        pts = self.geometry_aero.DVGeo.getLocalIndex(0)
        indexList = pts[3:6, :, :].flatten()
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
        self.add_design_var("shape", lower=-1, upper=1, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("scenario.aero_post.HFH", scaler=1.0)
        self.add_constraint("scenario.aero_post.PL", upper=1000, scaler=1.0)
        self.add_constraint("scenario.thermal_post.HF_INNER", upper=1000, scaler=1.0)


prob = om.Problem(reports=None)
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aerothermal.html")

optFuncs = OptFuncs([daOptionsAero, daOptionsThermal], prob)

prob.run_model()
totals = prob.compute_totals()

if gcomm.rank == 0:
    derivDict = {}
    derivDict["PL"] = {}
    derivDict["PL"]["shape"] = totals[("scenario.aero_post.functionals.PL", "dvs.shape")][0]
    derivDict["HFH"] = {}
    derivDict["HFH"]["shape"] = totals[("scenario.aero_post.functionals.HFH", "dvs.shape")][0]
    derivDict["HF_INNER"] = {}
    derivDict["HF_INNER"]["shape"] = totals[("scenario.thermal_post.functionals.HF_INNER", "dvs.shape")][0]
    reg_write_dict(derivDict, 1e-4, 1e-6)
