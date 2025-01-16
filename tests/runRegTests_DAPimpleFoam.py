#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
import os
import numpy as np
from testFuncs import *

import openmdao.api as om
from openmdao.api import Group
from mphys.multipoint import Multipoint
from dafoam.mphys.mphys_dafoam import DAFoamBuilderUnsteady
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/Ramp")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")
    replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0

daOptions = {
    "designSurfaces": ["bot"],
    "solverName": "DAPimpleFoam",
    #"useAD": {"mode": "forward", "seedIndex": 0, "dvName": "shape"},
    "primalBC": {
        # "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "functionTimeOperator": "average",
        "readZeroFields": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "timeOp": "average",
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0,
            "timeOp": "average",
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural", "dynAdjustTol": False},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "solverInput": {
        "aero_vol_coords": {"type": "volCoord"},
        "patchV": {"type": "patchVelocity", "patches": ["inlet"], "flowAxis": "x", "normalAxis": "y"},
    },
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.05], [0.0, 0.0, 1.0]]],
}


class Top(Group):
    def setup(self):

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/FFD.xyz", type="ffd"), promotes=["*"])

        self.add_subsystem(
            "cruise",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=meshOptions),
            promotes=["*"],
        )

        self.connect("x_aero0", "x_aero")

    def configure(self):

        # create geometric DV setup
        points = self.cruise.get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # add the dv_geo object to the builder solver. This will be used to write deformed FFDs
        self.cruise.solver.add_dvgeo(self.geometry.DVGeo)

        # geometry setup
        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        shapes.append({pts[2, 0, 0]: dir_y, pts[2, 0, 1]: dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # add the design variables to the dvs component's output
        self.dvs.add_output("patchV", val=np.array([10.0, 0.0]))
        self.dvs.add_output("shape", val=np.zeros(1))
        self.dvs.add_output("x_aero_in", val=points, distributed=True)

        # define the design variables to the top level
        self.add_design_var("patchV", indices=[0], lower=-50.0, upper=50.0, scaler=1.0)
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("CD", scaler=1.0)
        # self.add_constraint("CL", equals=0.3)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["CD"],
    wrt=["patchV", "shape"],
    compact_print=True,
    step=1e-2,
    form="central",
    step_calc="abs",
)
if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("CD")
    derivDict = {}
    derivDict["CD"] = {}
    derivDict["CD"]["patchV"] = results[("CD", "patchV")]["J_fwd"][0]
    derivDict["CD"]["shape"] = results[("CD", "shape")]["J_fwd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-8, 1e-12)
