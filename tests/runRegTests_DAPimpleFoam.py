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
from pygeo import geo_utils

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible.unsteady/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    # replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 10.0

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DAPimpleFoam",
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "shape"},
    "primalBC": {
        # "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "readZeroFields": True,
        "additionalOutput": ["U", "p", "phi"],
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "timeOp": "average",
            "timeOpStartIndex": 4,
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0,
            "timeOp": "max",
            "timeOpMaxMode": "KS",
            "timeOpMaxKSCoeff": 0.25,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inlet"],
            "flowAxis": "x",
            "normalAxis": "y",
            "components": ["solver", "function"],
        },
    },
    "unsteadyCompOutput": {
        "CD": ["CD"],
        "CL": ["CL"],
    },
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
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
        indexList = pts[1, 0, 1].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS)

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


# NOTE: the patchV deriv is accurate with FD but not with forward AD. The forward AD changed the primal value
# and the reason is still unknown..

prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

prob.run_model()
results = prob.check_totals(
    of=["cruise.solver.CD", "cruise.solver.CL"],
    wrt=["shape", "patchV"],
    compact_print=True,
    step=1e-2,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("cruise.solver.CD")
    funcDict["CL"] = prob.get_val("cruise.solver.CL")
    derivDict = {}
    derivDict["CD"] = {}
    derivDict["CD"]["shape-Adjoint"] = results[("cruise.solver.CD", "shape")]["J_fwd"][0]
    derivDict["CD"]["shape-FD"] = results[("cruise.solver.CD", "shape")]["J_fd"][0]
    derivDict["CD"]["patchV-Adjoint"] = results[("cruise.solver.CD", "patchV")]["J_fwd"][0]
    derivDict["CD"]["patchV-FD"] = results[("cruise.solver.CD", "patchV")]["J_fd"][0]
    derivDict["CL"] = {}
    derivDict["CL"]["shape-Adjoint"] = results[("cruise.solver.CL", "shape")]["J_fwd"][0]
    derivDict["CL"]["shape-FD"] = results[("cruise.solver.CL", "shape")]["J_fd"][0]
    derivDict["CL"]["patchV-Adjoint"] = results[("cruise.solver.CL", "patchV")]["J_fwd"][0]
    derivDict["CL"]["patchV-FD"] = results[("cruise.solver.CL", "patchV")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-10, 1e-12)
    reg_write_dict(derivDict, 1e-6, 1e-8)
