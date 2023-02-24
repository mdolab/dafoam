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

os.chdir("./input/Wing")
if gcomm.rank == 0:
    os.system("rm -rf 0 processor*")
    os.system("cp -r 0.incompressible 0")

# aero setup
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
CL_target = 0.2
pitch0 = 2.0
rho0 = 1.0
A0 = 45.5

daOptions = {
    "designSurfaces": ["wing", "wing_te"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-10,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "fvSource": {
        "disk1": {
            "type": "actuatorDisk",
            "source": "cylinderAnnulusSmooth",
            "center": [-1.0, 0.0, 5.0],
            "direction": [1.0, 0.0, 0.0],
            "innerRadius": 0.5,
            "outerRadius": 5.0,
            "rotDir": "right",
            "scale": 1.0,
            "POD": 0.0,
            "eps": 3.0,  # eps should be of cell size
            "expM": 1.0,
            "expN": 0.5,
            "adjustThrust": 1,
            "targetThrust": 100.0,
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
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["wing", "wing_te"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * 1.0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {
        "gmresRelTol": 1.0e-8,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-6},
    "checkMeshThreshold": {
        "maxAspectRatio": 5000.0,
        "maxNonOrth": 70.0,
        "maxSkewness": 8.0,
        "maxIncorrectlyOrientedFaces": 0,
    },
    "designVar": {
        "twist": {"designVarType": "FFD"},
        "shape": {"designVarType": "FFD"},
        "actuator": {"designVarType": "ACTD", "actuatorName": "disk1"},
        "uin": {"designVarType": "BC", "patches": ["inout"], "variable": "U", "comp": 0},
    },
    "adjPCLag": 1,
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        super().configure()

        self.cruise.aero_post.mphys_add_funcs()

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # create constraint DV setup
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # geometry setup

        # Create reference axis
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Set up global design variables
        def twist(val, geo):
            for i in range(nRefAxPts):
                geo.rot_z["wingAxis"].coef[i] = -val[i]

        def uin(val, DASolver):
            U = [float(val[0]), 0.0, 0.0]
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()

        self.cruise.coupling.solver.add_dv_func("uin", uin)
        self.cruise.aero_post.add_dv_func("uin", uin)

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
                        "targetThrust": T
                    },
                },
            )
            DASolver.updateDAOption()

        self.cruise.coupling.solver.add_dv_func("actuator", actuator)
        self.cruise.aero_post.add_dv_func("actuator", actuator)

        self.geometry.nom_addGlobalDV(dvName="twist", value=np.array([pitch0] * nRefAxPts), func=twist)

        # Select all points
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS)

        leList = [[0.1, 0, 0.01], [7.5, 0, 13.9]]
        teList = [[4.9, 0, 0.01], [8.9, 0, 13.9]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_add_LETEConstraint("lecon", 0, "iLow")
        self.geometry.nom_add_LETEConstraint("tecon", 0, "iHigh")
        # self.geometry.nom_addCurvatureConstraint1D(
        #    "curvature", start=[3, 0, 0], end=[8, 0, 12], nPts=20, axis=[0, 1, 0], curvatureType="mean", scaled=False
        # )
        self.geometry.nom_addLERadiusConstraints(
            "radius", leList=[[0.1, 0, 0], [7.0, 0, 13]], nSpan=10, axis=[0, 1, 0], chordDir=[1, 0, 0]
        )

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([pitch0] * nRefAxPts))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("uin", val=np.array([U0]))
        self.dvs.add_output("actuator", val=np.array([-1, 0.0, 5, 0.5, 5.0, 1.0, 1.0, 1.0, 0.5, 100.0]))
        self.connect("twist", "geometry.twist")
        self.connect("shape", "geometry.shape")
        self.connect("uin", "cruise.uin")
        self.connect("actuator", "cruise.actuator")

        # define the design variables
        self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        self.add_design_var("uin", lower=9.0, upper=11.0, scaler=1.0)
        self.add_design_var("actuator", lower=-1000.0, upper=1000.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=0.3, scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)
        # self.add_constraint("geometry.curvature", lower=0.0, upper=0.02, scaler=1.0)
        self.add_constraint("geometry.radius", lower=1.0, scaler=1.0)


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

optFuncs = OptFuncs([daOptions], prob)

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

# test the find feasible design function
optFuncs.findFeasibleDesign(
    ["cruise.aero_post.CL", "cruise.aero_post.CMZ"], ["twist", "twist"], designVarsComp=[0, 1], targets=[0.35, 1.5]
)
CL = prob.get_val("cruise.aero_post.CL")[0]
CMZ = prob.get_val("cruise.aero_post.CMZ")[0]
error = np.array([CL - 0.35, CMZ - 1.5])
error_norm = np.linalg.norm(error)
if error_norm > 2e-4:
    print("findFeasibleDesign Failed!!!")
    exit(1)
