#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

import openmdao.api as om
import os
import numpy as np

from mpi4py import MPI
from dafoam import PYDAFOAM
from testFuncs import *
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from funtofem.mphys import MeldThermalBuilder
from pygeo import geo_utils
from mphys.scenario_aerothermal import ScenarioAeroThermal
from pygeo.mphys import OM_DVGEOCOMP

# NOTE: we will test DASimpleFoam and DAHeatTransferFoam for incompressible, compressible, and solid solvers using the daCustom wallDistanceMethod

# ********************
# incompressible tests
# ********************
gcomm = MPI.COMM_WORLD
os.chdir("./reg_test_files-main/ChannelConjugateHeatV4")

if gcomm.rank == 0:
    os.system("rm -rf */processor*")

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
    "solverName": "DASimpleFoam",
    "wallDistanceMethod": "daCustom",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1.0e12,
    "discipline": "aero",
    "primalBC": {
        "UHot": {"variable": "U", "patches": ["hot_air_in"], "value": [U0, 0.0, 0.0]},
        "UCold": {"variable": "U", "patches": ["cold_air_in"], "value": [-U0, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "function": {
        "HFX": {
            "type": "wallHeatFlux",
            "byUnitArea": False,
            "source": "patchToFace",
            "patches": ["hot_air_inner"],
            "scale": 1,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-3,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
        "dynAdjustTol": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": 1e-3,
        "T": 300,
        "phi": 1.0,
    },
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "T_convect": {
            "type": "thermalCouplingInput",
            "patches": ["hot_air_inner", "cold_air_outer"],
            "components": ["solver"],
        },
    },
    "outputInfo": {
        "q_convect": {
            "type": "thermalCouplingOutput",
            "patches": ["hot_air_inner", "cold_air_outer"],
            "components": ["thermalCoupling"],
        },
    },
}

daOptionsThermal = {
    "designSurfaces": ["channel_outer", "channel_inner", "channel_sides"],
    "solverName": "DAHeatTransferFoam",
    "wallDistanceMethod": "daCustom",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1.0e12,
    "discipline": "thermal",
    "function": {
        "HF_INNER": {
            "type": "wallHeatFlux",
            "byUnitArea": False,
            "source": "patchToFace",
            "patches": ["channel_inner"],
            "scale": 1,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-3,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
        "dynAdjustTol": True,
    },
    "normalizeStates": {
        "T": 300.0,
    },
    "inputInfo": {
        "thermal_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "q_conduct": {
            "type": "thermalCouplingInput",
            "patches": ["channel_outer", "channel_inner"],
            "components": ["solver"],
        },
    },
    "outputInfo": {
        "T_conduct": {
            "type": "thermalCouplingOutput",
            "patches": ["channel_outer", "channel_inner"],
            "components": ["thermalCoupling"],
        },
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
            om.NonlinearBlockGS(maxiter=20, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-14),
            om.LinearBlockGS(maxiter=20, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-14),
        )

        # need to manually connect the x_aero0 between the mesh and geometry components
        self.connect("mesh_aero.x_aero0", "geometry_aero.x_aero_in")
        self.connect("geometry_aero.x_aero0", "scenario.x_aero")

        self.connect("mesh_thermal.x_thermal0", "geometry_thermal.x_thermal_in")
        self.connect("geometry_thermal.x_thermal0", "scenario.x_thermal")

    def configure(self):

        super().configure()

        # get the surface coordinates from the mesh component
        points_aero = self.mesh_aero.mphys_get_surface_mesh()
        points_thermal = self.mesh_thermal.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry_aero.nom_add_discipline_coords("aero", points_aero)
        self.geometry_thermal.nom_add_discipline_coords("thermal", points_thermal)

        # geometry setup
        pts = self.geometry_aero.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        shapes.append({pts[9, 0, 0]: dir_y, pts[9, 0, 1]: dir_y})
        self.geometry_aero.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)
        self.geometry_thermal.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0]))
        # manually connect the dvs output to the geometry
        self.connect("shape", "geometry_aero.shape")
        self.connect("shape", "geometry_thermal.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-1, upper=1, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("scenario.aero_post.HFX", scaler=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
prob.run_model()

HFX = prob.get_val("scenario.aero_post.HFX")[0]
print("HFX:", HFX)
if (abs(HFX - 180000.0) / (HFX + 1e-16)) > 1e-6:
    print("DAHeatTransferFoam test failed for DACustomCHT!")
    exit(1)
else:
    print("DAHeatTransferFoam test passed for DACustomCHT!")


# ********************
# compressible tests
# ********************
os.chdir("../ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 100.0

daOptions = {
    "solverName": "DARhoSimpleFoam",
    "wallDistanceMethod": "daCustom",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "T0": {"variable": "T", "patches": ["inlet"], "value": [310.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [101325.0]},
        "useWallFunction": True,
    },
    "function": {
        "HFX": {
            "type": "wallHeatFlux",
            "byUnitArea": False,
            "source": "patchToFace",
            "patches": ["walls"],
            "scale": 1.0,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

funcs = {}
DASolver.evalFunctions(funcs)

diff = abs(8967.626339018574 - funcs["HFX"]) / 8967.626339018574
if diff > 1e-10:
    if gcomm.rank == 0:
        print("DAFunction comp test failed for DACustomCHT")
    exit(1)
else:
    if gcomm.rank == 0:
        print("DAFunction comp test passed for DACustomCHT!")
