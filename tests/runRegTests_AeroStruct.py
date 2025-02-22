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
from tacs.mphys import TacsBuilder
from funtofem.mphys import MeldBuilder
from mphys.scenario_aerostructural import ScenarioAeroStructural
from pygeo.mphys import OM_DVGEOCOMP
from tacs import elements, constitutive, functions

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/Wing")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.subsonic 0")

# aero setup
U0 = 200.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
CL_target = 0.3
aoa0 = 2.0
rho0 = p0 / T0 / 287.0
A0 = 45.5

# struct setup
rho = 2780.0
E = 73.1e9
nu = 0.33
kcorr = 5.0 / 6.0
ys = 324.0e6
t = 0.003
tMin = 0.002
tMax = 0.05

daOptions = {
    "designSurfaces": ["wing", "wing_te"],
    "solverName": "DARhoSimpleFoam",
    # "couplingInfo": {
    #     "aerostructural": {
    #         "active": True,
    #         # the groupling surface group can be different
    #         # from the design surfaces
    #         "couplingSurfaceGroups": {
    #             "wingGroup": ["wing", "wing_te"],
    #         },
    #         "propMovement": False,
    #     },
    # },
    "primalMinResTol": 1.0e-10,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": False,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing", "wing_te"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
            "addToAdjoint": True,
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing", "wing_te"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
            "addToAdjoint": True,
        },
    },
    "adjEqnOption": {
        "gmresRelTol": 1.0e-2,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
        "useNonZeroInitGuess": True,
        "dynAdjustTol": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "checkMeshThreshold": {
        "maxSkewness": 8.0,
    },
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
    },
    "outputInfo": {
        "f_aero": {
            "type": "forceCouplingOutput",
            "patches": ["wing", "wing_te"],
            "components": ["forceCoupling"],
            "pRef": p0,
        },
    },
}

meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": False,
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
}


class Top(Multipoint):
    def setup(self):
        # DAFoam setup
        aero_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerostructural")
        aero_builder.initialize(self.comm)

        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        # TACS Setup
        def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
            prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
            con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum, tlb=tMin, tub=tMax)
            transform = None
            elem = elements.Quad4Shell(transform, con)
            return elem

        def problem_setup(scenario_name, fea_assembler, problem):
            problem.addFunction("mass", functions.StructuralMass)
            problem.addFunction("ks_vmfailure", functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)
            g = np.array([0.0, -9.81, 0.0])
            problem.addInertialLoad(g)

        tacs_options = {
            "element_callback": element_callback,
            "problem_setup": problem_setup,
            "mesh_file": "wingboxProp.bdf",
        }

        struct_builder = TacsBuilder(tacs_options)
        struct_builder.initialize(self.comm)

        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Transfer scheme options
        ################################################################################
        xfer_builder = MeldBuilder(aero_builder, struct_builder, isym=2, check_partials=True)
        xfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="./FFD/parentFFD.xyz", type="ffd"))

        # add the coupling solvers
        nonlinear_solver = om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-7)
        linear_solver = om.LinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-6, atol=1e-12)
        self.mphys_add_scenario(
            "cruise",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        for discipline in ["aero"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0" % discipline)
        for discipline in ["struct"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0" % discipline)

        # add the structural thickness DVs
        ndv_struct = struct_builder.get_ndv()
        dvs.add_output("dv_struct", np.array(ndv_struct * [0.01]))
        self.connect("dv_struct", "cruise.dv_struct")

        self.connect("mesh_aero.x_aero0", "geometry.x_aero_in")
        self.connect("mesh_struct.x_struct0", "geometry.x_struct_in")

    def configure(self):
        super().configure()

        # create geometric DV setup
        points = self.mesh_aero.mphys_get_surface_mesh()

        # create constraint DV setup
        tri_points = self.mesh_aero.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # geometry setup
        self.geometry.nom_addChild("./FFD/wingFFD.xyz")
        # Create reference axis
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k", childIdx=0)

        # Set up global design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wingAxis"].coef[i] = -val[i - 1]

        self.geometry.nom_addGlobalDV(dvName="twist", value=np.array([0] * (nRefAxPts - 1)), func=twist, childIdx=0)

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)
        self.geometry.nom_add_discipline_coords("struct")

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([aoa0] * (nRefAxPts - 1)))
        self.connect("twist", "geometry.twist")

        # define the design variables
        self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0, indices=[1])

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        # self.add_constraint("cruise.aero_post.CL", equals=CL_target, scaler=1.0)
        # stress constraint
        self.add_constraint("cruise.ks_vmfailure", lower=0.0, upper=0.41, scaler=1.0)


prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aerostruct.html")

prob.run_model()
results = prob.check_totals(
    # of=["PL.val", "scenario.aero_post.HFH", "scenario.thermal_post.HF_INNER"],
    # wrt=["shape"],
    compact_print=True,
    step=1e-3,
    form="central",
    step_calc="abs",
)


if gcomm.rank == 0:
    funcDict = {}
    funcDict["CD"] = prob.get_val("cruise.aero_post.CD")
    funcDict["CL"] = prob.get_val("cruise.aero_post.CL")
    funcDict["VM"] = prob.get_val("cruise.ks_vmfailure")
    derivDict = {}
    derivDict["CD"] = {}
    derivDict["CD"]["shape-Adjoint"] = results[("cruise.aero_post.functionals.CD", "dvs.twist")]["J_fwd"][0]
    derivDict["CD"]["shape-FD"] = results[("cruise.aero_post.functionals.CD", "dvs.twist")]["J_fd"][0]
    # derivDict["CL"] = {}
    # derivDict["CL"]["shape-Adjoint"] = results[("cruise.aero_post.functionals.CL", "dvs.twist")]["J_fwd"][0]
    # derivDict["CL"]["shape-FD"] = results[("cruise.aero_post.functionals.CL", "dvs.twist")]["J_fd"][0]
    derivDict["VM"] = {}
    derivDict["VM"]["shape-Adjoint"] = results[("cruise.struct_post.eval_funcs.ks_vmfailure", "dvs.twist")]["J_fwd"][0]
    derivDict["VM"]["shape-FD"] = results[("cruise.struct_post.eval_funcs.ks_vmfailure", "dvs.twist")]["J_fd"][0]
    reg_write_dict(funcDict, 1e-8, 1e-10)
    reg_write_dict(derivDict, 1e-4, 1e-10)
