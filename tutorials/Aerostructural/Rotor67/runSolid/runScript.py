#!/usr/bin/env python
"""
DAFoam run script for the Rotor 67 case
"""

# =================================================================================================
# Imports
# =================================================================================================
import os,time
import argparse
import sys
import numpy as np
from mpi4py import MPI
from baseclasses import *
from dafoam import *
from pygeo import *
from pyspline import *
from idwarp import *
from pyoptsparse import Optimization, OPT


# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--output", help='Output directory', type=str,default='../optOutputSolid/')
parser.add_argument("--opt", help="optimizer to use", type=str, default='snopt')
parser.add_argument("--task", help="type of run to do", type=str, default='runflow')
parser.add_argument("--runcounter", help="run counter", type=int, default=0)
parser.add_argument('--optVars',type=str,help='Vars for the optimizer',default="['shape']")
args = parser.parse_args()
exec('optVars=%s'%args.optVars)
task = args.task
outputDirectory = args.output
gcomm = MPI.COMM_WORLD
runcounter = args.runcounter

# Set the parameters for optimization
aeroOptions = {
    # output options
    'casename':                 'Rotor67_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,


    # design surfaces and cost functions 
    'designsurfacefamily':     'designSurfaces', 
    'designsurfaces':          ['bladeps','bladess','bladetip','bladefillet','hub','hubper1','hubper2','hubbot'], 
    'objfuncs':                ['VMS'],
    'objfuncgeoinfo':          [['allCells']],
    'userdefinedvolumeinfo':   {'userDefinedVolume0':{'type':'box',
                                                      'centerX':0.0,
                                                      'centerY':0.0,
                                                      'centerZ':0.0,
                                                      'sizeX':10.0,
                                                      'sizeY':10.0,
                                                      'sizeZ':10.0,
                                                      'stateName':'D',
                                                      'scale':1.0,
                                                      'component':2}},
    'referencevalues':         {'KSCoeff':8.0e-7,'magURef':1.0,'ARef':1.0,'LRef':1.0,'pRef':0.0,'rhoRef':1.0},
    'liftdir':                 [0.0,0.0,1.0],
    'dragdir':                 [1.0,0.0,0.0],
    'rotrad':                  [0.0,0.0,-840.0],
    'derivuininfo':            {'stateName':'D','component':2,'type':'rotRad','patchNames':['top']},
    # flow setup
    'adjointsolver':           'solidDisplacementDAFoam',
    'rasmodel':                'dummyTurbulenceModel',
    'flowcondition':           'Incompressible',
    'maxflowiters':            500, 
    'writeinterval':           500,
    'setflowbcs':              True,
    'flowbcs':                 {},
    'mechanicalproperties':     {'rho':2700.0,
                                 'nu':0.33,
                                 'E':0.689e11},
    'gradschemes':              {'default':'leastSquares'},
    'fvsolvers':                 {'D':{'solver':'GAMG',
                                       'tolerance':'1e-20',
                                       'relTol':'0.9',
                                       'smoother':'GaussSeidel',
                                       'maxIter':'1',
                                       'nCellsInCoarsestLevel':'20'}},
    # adjoint setup
    'adjgmresmaxiters':        500,
    'adjgmresrestart':         500,
    'adjgmresreltol':          1e-6,
    'adjdvtypes':              ['FFD'], 
    'maxtoljac':               1e200,
    'maxtolpc':                1e200,
    'epsderiv':                1.0e-5, 
    'epsderivuin':             1.0e-3,
    'epsderivffd':             1.0e-4, 
    'adjpcfilllevel':          1, 
    'adjjacmatordering':       'state',
    'adjjacmatreordering':     'rcm',
    'statescaling':            {'DScaling':1e-7},
    
    
    ########## misc setup ##########
    'mpispawnrun':             True,
    'restartopt':              False,
    'meshmaxnonortho':         80.0,
    'meshmaxskewness':         10.0,
    'tractionbcmaxiter':       100,

}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    'gridFile':                os.getcwd(),
    'fileType':                'openfoam',
    # point and normal for the symmetry plane
    'symmetryPlanes':          [[[0.,0., 0.],[0., 0., 0.]]], 
}

# =================================================================================================
# DVGeo
# =================================================================================================
DVGeo = DVGeometry('./FFD/globalFFD.xyz')
# Setup curves for ref_axis
x1 = [0.105,0.135,0.16,0.19,0.22,0.255]
y1 = [0.01, 0.01, 0.01, 0.01, 0.01,0.01]
z1 = [0.05, 0.05, 0.05, 0.05, 0.05,0.05]
c1 = pySpline.Curve(x=x1, y=y1, z=z1, k=2)
DVGeo.addRefAxis('bodyAxis', c1)

DVGeoChild = DVGeometry('./FFD/localFFD.xyz',child=True)
# Setup curves for ref_axis
x2 = [0.105,0.135,0.16,0.19,0.22,0.255]
y2 = [0.01, 0.01, 0.01, 0.01, 0.01,0.01]
z2 = [0.05, 0.05, 0.05, 0.05, 0.05,0.05] 
c2 = pySpline.Curve(x=x2, y=y2, z=z2, k=2)
DVGeoChild.addRefAxis('bodyAxisLocal', c2,volumes=[0])

# FFD shape
pts=DVGeoChild.getLocalIndex(0) 
indexListX=pts[2:-1,:,:].flatten()  # select the top layer FFD starts with i=1
PS=geo_utils.PointSelect('list',indexListX)
DVGeoChild.addGeoDVLocal('shapex', lower=-0.01, upper=0.01, axis='x', scale=1.0,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapey', lower=-0.01, upper=0.01, axis='y', scale=1.0,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapez', lower=-0.01, upper=0.01, axis='z', scale=1.0,pointSelect=PS)

DVGeo.addChild(DVGeoChild)

# =================================================================================================
# DAFoam
# =================================================================================================
CFDSolver = PYDAFOAM(options=aeroOptions, comm=gcomm)
CFDSolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions,comm=gcomm)
CFDSolver.addFamilyGroup(CFDSolver.getOption('designsurfacefamily'),CFDSolver.getOption('designsurfaces'))
if MPI.COMM_WORLD.rank == 0:
    CFDSolver.printFamilyList()
CFDSolver.setMesh(mesh)
CFDSolver.computeAdjointColoring()
evalFuncs = CFDSolver.getOption('objfuncs')
xDVs = DVGeo.getValues()
nDVs = 0
for key in xDVs.keys():
    nDVs += len( xDVs[key] )
CFDSolver.setOption('nffdpoints',nDVs)


# =================================================================================================
# DVCon
# =================================================================================================
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
[p0, v1, v2] = CFDSolver.getTriangulatedMeshSurface(groupName=CFDSolver.getOption('designsurfacefamily'))
surf = [p0, v1, v2]
DVCon.setSurface(surf)


# ================================================================================================
# optFuncs
# =================================================================================================
optFuncs.CFDSolver = CFDSolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm


# =================================================================================================
# Task
# =================================================================================================
if task.lower() == 'runflow':

    # read the design variable values
    f = open('designVariables.dat','r')
    lines = f.readlines()
    f.close()
    newDV = {}
    for line in lines:
	cols = line.split()
        if not cols: # empty
            break
        newDV[cols[0]]=[]
        for val in cols[1:]:
            newDV[cols[0]].append(float(val))
	
    xDV = DVGeo.getValues()

    if gcomm.rank==0:
        print("DVGeo initial ",xDV)    

    for key in xDV:
        length = len(xDV[key])
        for i in range(length):
            xDV[key][i]=newDV[key][i]
    
    if gcomm.rank==0:
        print("DVGeo set: ",xDV)
	
    DVGeo.setDesignVars(xDV)
 
    CFDSolver.flowRunsCounter = runcounter 
    # Evaluate the functions
    CFDSolver()

    f=open('solidMeshFailed.dat','w')
    if CFDSolver.meshQualityFailure == True:
        f.write('1\n')
    else:
        f.write('0\n')
    f.close()

elif task.lower() == 'runadjoint':

    # read the design variable values
    f = open('designVariables.dat','r')
    lines = f.readlines()
    f.close()
    newDV = {}
    for line in lines:
	cols = line.split()
        if not cols: # empty
            break
        newDV[cols[0]]=[]
        for val in cols[1:]:
            newDV[cols[0]].append(float(val))
	
    xDV = DVGeo.getValues()

    if gcomm.rank==0:
        print("DVGeo initial ",xDV)    

    for key in xDV:
        length = len(xDV[key])
        for i in range(length):
            xDV[key][i]=newDV[key][i]
    
    if gcomm.rank==0:
        print("DVGeo set: ",xDV)
	
    DVGeo.setDesignVars(xDV)
    CFDSolver.writeUpdatedVolumePoints()

    CFDSolver.adjointRunsCounter = runcounter
    CFDSolver.solveADjoint()

else:
    print("task arg not found!")
    exit(0)


