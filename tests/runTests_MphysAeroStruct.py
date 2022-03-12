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
from dafoam.mphys_dafoam import DAFoamBuilder, checkDesignVarSetup
from tacs.mphys.mphys_tacs import TacsBuilder
from mphys.solver_builders.mphys_meld import MeldBuilder
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.solver_builders.mphys_dvgeo import OM_DVGEOCOMP
from tacs import elements, constitutive, functions

gcomm = MPI.COMM_WORLD

os.chdir("./input/Wing")
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
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleFoam",
    "fsi": {
        "pRef": p0,
    },
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": False,
    },
    # variable bounds for compressible flow conditions
    "primalVarBounds": {
        "UMax": 1000.0,
        "UMin": -1000.0,
        "pMax": 500000.0,
        "pMin": 20000.0,
        "eMax": 500000.0,
        "eMin": 100000.0,
        "rhoMax": 5.0,
        "rhoMin": 0.2,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {
        "gmresRelTol": 1.0e-2,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
        "useNonZeroInitGuess": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "checkMeshThreshold": {
        "maxAspectRatio": 5000.0,
        "maxNonOrth": 70.0,
        "maxSkewness": 8.0,
        "maxIncorrectlyOrientedFaces": 0,
    },
    "designVar": {
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
        "twist": {"designVarType": "FFD"},
        "shape": {"designVarType": "FFD"},
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
            g = np.array([0.0, 0.0, -9.81])
            problem.addInertialLoad(g)

        tacs_options = {
            "element_callback": element_callback,
            "problem_setup": problem_setup,
            "mesh_file": "wingbox.bdf",
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(ffd_file="./FFD/wingFFD.xyz"))

        # add the coupling solvers
        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-8)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-6, atol=1e-6)
        self.mphys_add_scenario(
            "cruise",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        for discipline in ["aero", "struct"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0" % discipline)

        # add the structural thickness DVs
        ndv_struct = struct_builder.get_ndv()
        dvs.add_output("dv_struct", np.array(ndv_struct * [0.01]))
        self.connect("dv_struct", "cruise.dv_struct")

        self.connect("mesh_aero.x_aero0", "geometry.x_aero_in")
        self.connect("mesh_struct.x_struct0", "geometry.x_struct_in")

    def configure(self):
        super().configure()

        self.cruise.aero_post.mphys_add_funcs(["CD", "CL"])

        # create geometric DV setup
        points = self.mesh_aero.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)
        self.geometry.nom_add_discipline_coords("struct")

        # create constraint DV setup
        tri_points = self.mesh_aero.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # geometry setup

        # Create reference axis
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Set up global design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wingAxis"].coef[i] = -val[i - 1]

        def aoa(val, DASolver):
            aoa = val[0] * np.pi / 180.0
            U = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()

        self.cruise.coupling.aero.solver.add_dv_func("aoa", aoa)
        self.cruise.aero_post.add_dv_func("aoa", aoa)

        self.geometry.nom_addGeoDVGlobal(dvName="twist", value=np.array([0] * (nRefAxPts - 1)), func=twist)
        nShapes = self.geometry.nom_addGeoDVLocal(dvName="shape")

        # Set up constraints
        leList = [[0.1, 0, 0.01], [7.5, 0, 13.9]]
        teList = [[4.9, 0, 0.01], [8.9, 0, 13.9]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_add_LETEConstraint("lecon", 0, "iLow")
        self.geometry.nom_add_LETEConstraint("tecon", 0, "iHigh")

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([0] * (nRefAxPts - 1)))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("aoa", val=np.array([aoa0]))
        self.connect("twist", "geometry.twist")
        self.connect("shape", "geometry.shape")
        self.connect("aoa", "cruise.aoa")

        # define the design variables
        self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        self.add_design_var("aoa", lower=0.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)


prob = om.Problem()
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

# check if the design variable dict is properly set
checkDesignVarSetup(daOptions, prob.model.get_design_vars())

prob.run_driver()

prob.record("final")
prob.cleanup()

cr = om.CaseReader("cases.sql")

# get list of cases recorded on problem
problem_cases = cr.list_cases("problem")
cr.list_source_vars("problem")
case = cr.get_case("final")

finalObj = case.get_objectives()
finalCon = case.get_constraints()

finalCL = {}
for key in finalCon.keys():
    if "CL" in key:
        finalCL[key] = finalCon[key]

if gcomm.rank == 0:
    reg_write_dict(finalObj, 1e-6, 1e-8)
    reg_write_dict(finalCL, 1e-6, 1e-8)
