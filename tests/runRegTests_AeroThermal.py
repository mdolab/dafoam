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
        "PL1": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["hot_air_in", "cold_air_in"],
            "scale": 1.0,
        },
        "PL2": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["hot_air_out", "cold_air_out"],
            "scale": 1.0,
        },
        "HFH": {
            "type": "wallHeatFlux",
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
            "components": ["solver", "function"],
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
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1.0e12,
    "discipline": "thermal",
    "function": {
        "HF_INNER": {
            "type": "wallHeatFlux",
            "source": "patchToFace",
            "patches": ["channel_inner"],
            "scale": 1,
        },
        "TVolSum": {
            "type": "variableVolSum",
            "source": "allCells",
            "varName": "T",
            "varType": "scalar",
            "index": 0,
            "isSquare": 0,
            "multiplyVol": 1,
            "divByTotalVol": 0,
            "scale": 1.0,
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
            # NOTE. this should include "function" as well. However, the total is worse
            # with "function"...
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
            om.NonlinearBlockGS(maxiter=20, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-14),
            om.LinearBlockGS(maxiter=20, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-14),
        )

        # need to manually connect the x_aero0 between the mesh and geometry components
        self.connect("mesh_aero.x_aero0", "geometry_aero.x_aero_in")
        self.connect("geometry_aero.x_aero0", "scenario.x_aero")

        self.connect("mesh_thermal.x_thermal0", "geometry_thermal.x_thermal_in")
        self.connect("geometry_thermal.x_thermal0", "scenario.x_thermal")

        self.add_subsystem("PL", om.ExecComp("val=PL1-PL2"))

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
        self.connect("scenario.aero_post.PL1", "PL.PL1")
        self.connect("scenario.aero_post.PL2", "PL.PL2")
        self.add_objective("scenario.aero_post.HFH", scaler=1.0)
        self.add_constraint("PL.val", upper=1000, scaler=1.0)
        self.add_constraint("scenario.thermal_post.HF_INNER", upper=1000, scaler=1.0)


prob = om.Problem(reports=None)
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aerothermal.html")

prob.run_model()
results = prob.check_totals(
    of=["PL.val", "scenario.aero_post.HFH", "scenario.thermal_post.HF_INNER", "scenario.thermal_post.TVolSum"],
    wrt=["shape"],
    compact_print=True,
    step=1e-3,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["HFH"] = prob.get_val("scenario.aero_post.HFH")
    funcDict["PL"] = prob.get_val("PL.val")
    funcDict["HF_INNER"] = prob.get_val("scenario.thermal_post.HF_INNER")
    derivDict = {}
    derivDict["PL"] = {}
    derivDict["PL"]["shape-Adjoint"] = results[("PL.val", "shape")]["J_fwd"][0]
    derivDict["PL"]["shape-FD"] = results[("PL.val", "shape")]["J_fd"][0]
    derivDict["HFH"] = {}
    derivDict["HFH"]["shape-Adjoint"] = results[("scenario.aero_post.HFH", "shape")]["J_fwd"][0]
    derivDict["HFH"]["shape-FD"] = results[("scenario.aero_post.HFH", "shape")]["J_fd"][0]
    derivDict["HF_INNER"] = {}
    derivDict["HF_INNER"]["shape-Adjoint"] = results[("scenario.thermal_post.HF_INNER", "shape")]["J_fwd"][0]
    derivDict["HF_INNER"]["shape-FD"] = results[("scenario.thermal_post.HF_INNER", "shape")]["J_fd"][0]
    derivDict["TVolSum"] = {}
    derivDict["TVolSum"]["shape-Adjoint"] = results[("scenario.thermal_post.TVolSum", "shape")]["J_fwd"][0]
    derivDict["TVolSum"]["shape-FD"] = results[("scenario.thermal_post.TVolSum", "shape")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-8, 1e-10)
    reg_write_dict(derivDict, 1e-4, 1e-10)
