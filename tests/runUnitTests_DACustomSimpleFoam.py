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
from funtofem.mphys import MeldThermalBuilder
from pygeo import geo_utils
from mphys.scenario_aerothermal import ScenarioAeroThermal
from pygeo.mphys import OM_DVGEOCOMP

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
            "formulation": "daCustom",
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
            "thermalCouplingMode": "daCustom",
            "patches": ["hot_air_inner", "cold_air_outer"],
            "components": ["solver"],
        },
    },
    "outputInfo": {
        "q_convect": {
            "type": "thermalCouplingOutput",
            "thermalCouplingMode": "daCustom",
            "patches": ["hot_air_inner", "cold_air_outer"],
            "components": ["thermalCoupling"],
        },
    },
}

daOptionsThermal = {
    "designSurfaces": ["channel_outer", "channel_inner", "channel_sides"],
    "solverName": "DAHeatTransferFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1.0e12,
    "discipline": "thermal",
    "function": {
        "HF_INNER": {
            "type": "wallHeatFlux",
            "formulation": "daCustom",
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
            "thermalCouplingMode": "daCustom",
            "patches": ["channel_outer", "channel_inner"],
            "components": ["solver"],
        },
    },
    "outputInfo": {
        "T_conduct": {
            "type": "thermalCouplingOutput",
            "thermalCouplingMode": "daCustom",
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
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# verify the total derivatives against the finite-difference
prob.run_model()
totals = prob.compute_totals()

HFX = prob.get_val("scenario.aero_post.HFX")[0]
print("HFX:", HFX)
if (abs(HFX - 180000.0) / (HFX + 1e-16)) > 1e-6:
    print("DACustomHeatTransferFoam test failed!")
    exit(1)
else:
    print("DACustomHeatTransferFoam test passed!")
