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
from dafoam.mphys import DAFoamBuilder, DAFoamLinearConstraint, DAFoamVSPVolume
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/NACA0012V4")
if gcomm.rank == 0:
    os.system("rm -rf 0 system processor* *.bin")
    os.system("cp -r 0.incompressible 0")
    os.system("cp -r system.incompressible system")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")

# aero setup
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
A0 = 0.1

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-10,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 1.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
        "transport:nu": 1.5e-5,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="airfoil.vsp3", type="vsp"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

        # thickness constraint
        varA = []
        varB = []
        for j in range(7):
            varA.append(f"UpperCoeff_{j}")
            varB.append(f"LowerCoeff_{j}")
        self.add_subsystem(
            "thickness",
            DAFoamLinearConstraint(varA=varA, coeffA=1.0, varB=varB, coeffB=-1.0, size=1, output_name="thickness_val"),
            promotes=["*"],
        )

        # LE C1 continuity constraint
        self.add_subsystem(
            "le_c1",
            DAFoamLinearConstraint(
                varA=["UpperCoeff_0"], coeffA=1.0, varB=["LowerCoeff_0"], coeffB=1.0, size=1, output_name="le_c1_val"
            ),
            promotes=["*"],
        )

        # add volume constraint
        vsp_vars = []
        for i in range(2):
            for j in range(7):
                vsp_vars.append(f"NACA:UpperCoeff_{i}:Au_{j}")
                vsp_vars.append(f"NACA:LowerCoeff_{i}:Al_{j}")
        self.add_subsystem(
            "volume",
            DAFoamVSPVolume(
                vsp_file="airfoil.vsp3",
                vsp_vars=vsp_vars,
                slice_dir="x",
                n_slices=10,
                output_name="volume_val",
                step=1e-3,
                relativeStep=False,
            ),
        )

    def configure(self):

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        for i in range(2):
            for j in range(7):
                self.geometry.nom_addVSPVariable("NACA", f"UpperCoeff_{i}", f"Au_{j}", scaledStep=False, dh=1e-3)
                self.geometry.nom_addVSPVariable("NACA", f"LowerCoeff_{i}", f"Al_{j}", scaledStep=False, dh=1e-3)

        # add the design variables to the dvs component's output
        # these are for NACA0012
        CST = np.array([0.17299, 0.15121, 0.16626, 0.13844, 0.14289, 0.13999, 0.14070])
        for j in range(7):
            self.dvs.add_output(f"UpperCoeff_{j}", val=CST[j])
            self.connect(f"UpperCoeff_{j}", f"geometry.NACA:UpperCoeff_0:Au_{j}")
            self.connect(f"UpperCoeff_{j}", f"geometry.NACA:UpperCoeff_1:Au_{j}")
            self.connect(f"UpperCoeff_{j}", f"volume.NACA:UpperCoeff_0:Au_{j}")
            self.connect(f"UpperCoeff_{j}", f"volume.NACA:UpperCoeff_1:Au_{j}")
            self.dvs.add_output(f"LowerCoeff_{j}", val=-CST[j])
            self.connect(f"LowerCoeff_{j}", f"geometry.NACA:LowerCoeff_0:Al_{j}")
            self.connect(f"LowerCoeff_{j}", f"geometry.NACA:LowerCoeff_1:Al_{j}")
            self.connect(f"LowerCoeff_{j}", f"volume.NACA:LowerCoeff_0:Al_{j}")
            self.connect(f"LowerCoeff_{j}", f"volume.NACA:LowerCoeff_1:Al_{j}")

        # define the design variables to the top level
        for j in range(7):
            self.add_design_var(f"UpperCoeff_{j}", lower=-1.0, upper=1.0, scaler=1.0)
            self.add_design_var(f"LowerCoeff_{j}", lower=-1.0, upper=1.0, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.5, scaler=1.0)

        # volume constraint
        self.add_constraint("volume.volume_val", lower=1.0, scaler=1.0)

        # 50% thickness
        for j in range(1, 7):
            self.add_constraint(f"thickness_val_{j}", lower=0.5 * 2.0 * CST[j], scaler=1.0, linear=True)
        # LE radius does not change
        self.add_constraint(f"thickness_val_0", lower=2.0 * CST[0], scaler=1.0, linear=True)
        # LE C1 continuous
        self.add_constraint(f"le_c1_val_0", equals=0.0, scaler=1.0, linear=True)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# verify the total derivatives against the finite-difference
prob.run_model()
results = prob.check_totals(
    of=["cruise.aero_post.CD", "cruise.aero_post.CL", "volume.volume_val", "thickness_val_0"],
    wrt=["UpperCoeff_0", "LowerCoeff_2"],
    compact_print=True,
    step=1e-3,
    form="central",
    step_calc="abs",
)

if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("cruise.aero_post.CD")
    funcDict["CL"] = prob.get_val("cruise.aero_post.CL")
    funcDict["volume"] = prob.get_val("volume.volume_val")
    funcDict["thickness_0"] = prob.get_val("thickness_val_0")
    derivDict = {}
    derivDict["CD"] = {}
    derivDict["CD"]["UpperCoeff_0-Adjoint"] = results[("cruise.aero_post.CD", "UpperCoeff_0")]["J_fwd"][0]
    derivDict["CD"]["UpperCoeff_0-FD"] = results[("cruise.aero_post.CD", "UpperCoeff_0")]["J_fd"][0]
    derivDict["CD"]["LowerCoeff_2-Adjoint"] = results[("cruise.aero_post.CD", "LowerCoeff_2")]["J_fwd"][0]
    derivDict["CD"]["LowerCoeff_2-FD"] = results[("cruise.aero_post.CD", "LowerCoeff_2")]["J_fd"][0]

    derivDict["CL"] = {}
    derivDict["CL"]["UpperCoeff_0-Adjoint"] = results[("cruise.aero_post.CL", "UpperCoeff_0")]["J_fwd"][0]
    derivDict["CL"]["UpperCoeff_0-FD"] = results[("cruise.aero_post.CL", "UpperCoeff_0")]["J_fd"][0]
    derivDict["CL"]["LowerCoeff_2-Adjoint"] = results[("cruise.aero_post.CL", "LowerCoeff_2")]["J_fwd"][0]
    derivDict["CL"]["LowerCoeff_2-FD"] = results[("cruise.aero_post.CL", "LowerCoeff_2")]["J_fd"][0]

    derivDict["volume"] = {}
    derivDict["volume"]["UpperCoeff_0-Adjoint"] = results[("volume.volume_val", "UpperCoeff_0")]["J_fwd"][0]
    derivDict["volume"]["UpperCoeff_0-FD"] = results[("volume.volume_val", "UpperCoeff_0")]["J_fd"][0]
    derivDict["volume"]["LowerCoeff_2-Adjoint"] = results[("volume.volume_val", "LowerCoeff_2")]["J_fwd"][0]
    derivDict["volume"]["LowerCoeff_2-FD"] = results[("volume.volume_val", "LowerCoeff_2")]["J_fd"][0]

    derivDict["thickness_0"] = {}
    derivDict["thickness_0"]["UpperCoeff_0-Adjoint"] = results[("thickness_val_0", "UpperCoeff_0")]["J_fwd"][0]
    derivDict["thickness_0"]["UpperCoeff_0-FD"] = results[("thickness_val_0", "UpperCoeff_0")]["J_fd"][0]
    derivDict["thickness_0"]["LowerCoeff_2-Adjoint"] = results[("thickness_val_0", "LowerCoeff_2")]["J_fwd"][0]
    derivDict["thickness_0"]["LowerCoeff_2-FD"] = results[("thickness_val_0", "LowerCoeff_2")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-8, 1e-10)
    reg_write_dict(derivDict, 1e-6, 1e-08)
