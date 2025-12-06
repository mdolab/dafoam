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

os.chdir("./reg_test_files-main/ChannelConjugateHeatV4/thermal")
if gcomm.rank == 0:
    os.system("rm -rf processor*")
    os.system("rm 0.00* -fr")
    # set the kappa = 200 + 20*T
    os.system("sed -i 's/k  200;/kCoeffs (200 20);/g' constant/solidProperties")
    # set channerl_outer boundary type to wallHeatFluxTransfer
    # os.system("cp 0/T 0/T.old")
    # sed_changeBoundary = sed_cmd = (
    # f"sed -i '/^    channel_outer$/{{N;N;N;N;N;N; s/^    channel_outer\\n    {{\\n        type            mixed;\\n        refValue        uniform 300;\\n        refGradient     uniform 0;\\n        valueFraction   uniform 1;\\n    }}$/    channel_outer\\n    {{\\n        type            wallHeatFluxTransfer;\\n        Ta               300;\\n        h     uniform 20000;\\n        value   uniform 0;\\n    }}/;}}' {'0/T'}"
    # )
    # os.system(sed_changeBoundary)


daOptions = {
    "designSurfaces": ["channel_outer", "channel_inner", "channel_sides"],
    "solverName": "DAHeatTransferFoam",
    "discipline": "aero",
    "function": {
        "HFX": {
            "type": "wallHeatFlux",
            "source": "patchToFace",
            "patches": ["channel_inner"],
            "scale": 1,
        },
    },
    "primalMinResTol": 1e-12,
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-10,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
        "dynAdjustTol": True,
    },
    "normalizeStates": {
        "T": 300.0,
    },
    "inputInfo": {"aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]}},
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

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        self.add_subsystem("geometry", OM_DVGEOCOMP(file="../aero/FFD/channelFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # Select all points
        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        shapes.append({pts[10, 0, 0]: dir_y, pts[10, 0, 1]: dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0.0]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.HFX", scaler=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# verify the total derivatives against the finite-difference
prob.run_model()
totals = prob.check_totals(
    of=["cruise.aero_post.HFX"],
    wrt=["shape"],
    compact_print=True,
    step=1e-3,
    form="central",
    step_calc="abs",
)

if MPI.COMM_WORLD.rank == 0:
    # restore the kappa
    os.system("sed -i 's/kCoeffs (200 20);/k  200;/g' constant/solidProperties")
    # restore the boundary
    # os.system("mv 0/T.old 0/T")
if abs(totals[("cruise.aero_post.HFX", "shape")]["rel error"][0]) < 1e-5:
    print("DAHeatTransferFoam test passed!")
else:
    print("DAHeatTransferFoam test failed!")
