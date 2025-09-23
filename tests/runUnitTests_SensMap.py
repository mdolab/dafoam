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
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils
import pyofm

gcomm = MPI.COMM_WORLD

# force the optimizer to print all elements for the numpy arrays
np.set_printoptions(threshold=100000000)

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf 0/* 0.00* *.bin processor*")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible/* system/")
    os.system("cp -r constant/turbulenceProperties.sa constant/turbulenceProperties")
    # replace_text_in_file("system/fvSchemes", "meshWave;", "meshWaveFrozen;")

# aero setup
U0 = 240.0
p0 = 101325.0
T0 = 300.0
A0 = 0.1
twist0 = 3.0
LRef = 1.0
nuTilda0 = 4.5e-5

# aero setup
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
nCells = 343

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "writeAdjointFields": True,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": False,
        "transport:nu": 1.5e-5,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 0.1,
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "alpha": {
            "type": "field",
            "fieldName": "alphaPorosity",
            "fieldType": "scalar",
            "distributed": False,
            "components": ["solver", "function"],
        },
    },
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

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # create the mesh and cruise scenario because we only have one analysis point
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/FFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # add the shape dv
        nShapes = self.geometry.nom_addLocalDV(dvName="shape")

        # Design variable values. For the baseline design, they
        # are all zeros. If you need to plot the sensMap for an
        # intermediate design during optimization, you can record
        # the design variable for that opt iteration and manually
        # assign the design variable here.
        shape0 = np.zeros(nShapes)
        alpha0 = np.zeros(nCells)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=shape0)
        self.dvs.add_output("alpha", val=alpha0)
        # manually connect the dvs output to the geometry and cruise
        self.connect("shape", "geometry.shape")
        self.connect("alpha", "cruise.alpha")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("alpha", lower=-10.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)


prob = om.Problem()
prob.model = Top()

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# optFuncs = OptFuncs(daOptions, prob)

# compute the totals of the obj wrt the surface coordinates x_aero0
# and a field variable alpha.
prob.run_model()
results = prob.compute_totals(
    of=["cruise.aero_post.CD"],
    wrt=["geometry.x_aero0", "dvs.alpha"],
    get_remote=True,
)
# extract the sensitivity
totalsXs = results[("cruise.aero_post.CD", "geometry.x_aero0")][0]
totalsAlpha = results[("cruise.aero_post.CD", "dvs.alpha")][0]

# plot the sens map for the surface coordinates
# first get the size of the surface
DASolver = prob.model.cruise.coupling.solver.DASolver
Xs = DASolver.getSurfaceCoordinates(DASolver.designSurfacesGroup).flatten()
localSize = len(Xs)
# the totalXs is in global format, so we need to assign totalXs to each
# processor, if running in parallel
offset = gcomm.exscan(localSize)
if gcomm.rank == 0:
    offset = 0
totalsLocal = np.zeros(localSize)
iStart = offset
iEnd = offset + localSize
for globalI in range(iStart, iEnd):
    localI = globalI - iStart
    totalsLocal[localI] = totalsXs[globalI]
# call the writeSensMap API. Note here totalsLocal should be the sens owned by each proc
# the 0.0001 is the folder to write the sens
DASolver.solver.writeSensMapSurface("dCD_dXs", totalsLocal, Xs, localSize, 0.0001)

# plot the sens map for the alpha field
localSize = prob.model.cruise.coupling.solver.DASolver.solver.getNLocalCells()
# the totalXs is in global format, so we need to assign totalXs to each
# processor, if running in parallel
offset = gcomm.exscan(localSize)
if gcomm.rank == 0:
    offset = 0
totalsLocal = np.zeros(localSize)
iStart = offset
iEnd = offset + localSize
for globalI in range(iStart, iEnd):
    localI = globalI - iStart
    totalsLocal[localI] = totalsAlpha[globalI]
# call the writeSensMap API. Note here totalsLocal should be the sens owned by each proc
# the 0.0001 is the folder to write the sens
DASolver.solver.writeSensMapField("dCD_dAlpha", totalsLocal, "scalar", 0.0001)
