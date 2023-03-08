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
    "designSurfaces": ["wing", "wing_te"],
    "solverName": "DARhoSimpleFoam",
    "couplingInfo": {
        "aerostructural": {
            "active": True,
            "pRef": p0,
            "propMovement": True,
            "fvSource": {
                "disk1": {
                    "nNodes": 4,
                    "radialLoc": 0.1,
                },
            },
            # the groupling surface group can be different 
            # from the design surfaces
            "couplingSurfaceGroups": {
                "wingGroup": ["wing", "wing_te"],
            },
        },
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
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusSmooth",
            "center": [7.0, 0.0, 14.0],
            "direction": [1.0, 0.0, 0.0],
            "innerRadius": 0.1,
            "outerRadius": 1.0,
            "rotDir": "left",
            "scale": 1.0,
            "POD": 0.0,
            "eps": 0.05,
            "expM": 1.0,
            "expN": 0.5,
            "adjustThrust": 1,
            "targetThrust": 2000.0,
        },
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing", "wing_te"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing", "wing_te"],
                "directionMode": "fixedDirection",
                "direction": [0.0, 1.0, 0.0],
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
        "twist": {"designVarType": "FFD"},
        "shape": {"designVarType": "FFD"},
        "actuator_disk1": {"designVarType": "ACTD", "actuatorName": "disk1"},
    },
    #"adjPCLag": 1,
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

        for discipline in ["aero"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0_masked" % discipline)
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

        self.cruise.aero_post.mphys_add_funcs()

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
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", childIdx=0)

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)
        self.geometry.nom_add_discipline_coords("struct")

        def actuator(val, DASolver):
            actX = float(val[0])
            actY = float(val[1])
            actZ = float(val[2])
            actR1 = float(val[3])
            actR2 = float(val[4])
            actScale = float(val[5])
            actPOD = float(val[6])
            actExpM = float(val[7])
            actExpN = float(val[8])
            T = float(val[9])
            DASolver.setOption(
                "fvSource",
                {
                    "disk1": {
                        "center": [actX, actY, actZ],
                        "innerRadius": actR1,
                        "outerRadius": actR2,
                        "scale": actScale,
                        "POD": actPOD,
                        "expM": actExpM,
                        "expN": actExpN,
                        "targetThrust": T,
                    },
                },
            )
            DASolver.updateDAOption()

        self.cruise.coupling.aero.solver.add_dv_func("actuator_disk1", actuator)
        self.cruise.aero_post.add_dv_func("actuator_disk1", actuator)

        # Set up constraints
        leList = [[0.1, 0, 0.01], [7.5, 0, 13.9]]
        teList = [[4.9, 0, 0.01], [8.9, 0, 13.9]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_add_LETEConstraint("lecon", 0, "iLow", childIdx=0)
        self.geometry.nom_add_LETEConstraint("tecon", 0, "iHigh", childIdx=0)

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([aoa0] * (nRefAxPts - 1)))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("actuator", val=np.array([7.0, 0.0, 14.0, 0.1, 1.0, 1.0, 0.0, 1.0, 0.5, 2000.0]))
        self.connect("twist", "geometry.twist")
        self.connect("shape", "geometry.shape")
        self.connect("actuator", "cruise.dv_actuator_disk1", src_indices=[3, 4, 5, 6, 7, 8, 9])
        self.connect("actuator", "cruise.x_prop0_disk1", src_indices=[0, 1, 2])

        # define the design variables
        self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)
        # stress constraint
        self.add_constraint("cruise.ks_vmfailure", lower=0.0, upper=0.41, scaler=1.0)


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

optFuncs = OptFuncs(daOptions, prob)

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
finalVM = {}
for key in finalCon.keys():
    if "CL" in key:
        finalCL[key] = finalCon[key]
    if "ks_vmfailure" in key:
        finalVM[key] = finalCon[key]

if gcomm.rank == 0:
    reg_write_dict(finalObj, 1e-6, 1e-8)
    reg_write_dict(finalCL, 1e-6, 1e-8)
    reg_write_dict(finalVM, 1e-6, 1e-8)
