#!/usr/bin/env python
"""
DAFoam run script for the Rotor67 structure optimization
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
parser.add_argument("--output", help='Output directory', type=str,default='../optOutput/')
parser.add_argument("--opt", help="optimizer to use", type=str, default='slsqp')
parser.add_argument("--task", help="type of run to do", type=str, default='opt')
parser.add_argument('--optVars',type=str,help='Vars for the optimizer',default="['shape']")
args = parser.parse_args()
exec('optVars=%s'%args.optVars)
task = args.task
outputDirectory = args.output
gcomm = MPI.COMM_WORLD

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
    'rotrad':                  [0.0,0.0,-1680.0],
    'derivuininfo':            {'stateName':'D','component':2,'type':'rotRad','patchNames':['top']},
    # flow setup
    'adjointsolver':           'solidDisplacementDAFoam',
    'rasmodel':                'dummyTurbulenceModel',
    'flowcondition':           'Incompressible',
    'maxflowiters':            300, 
    'writeinterval':           300,
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
    'mpispawnrun':             False,
    'restartopt':              True,
    #'meshmaxnonortho':         70.0,
    #'meshmaxskewness':         10.0,
    'tractionbcmaxiter':       100,

}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    'gridFile':                os.getcwd(),
    'fileType':                'openfoam',
    # point and normal for the symmetry plane
    'symmetryPlanes':          [[[0.,0., 0.],[0., 0., 0.]]], 
}

# options for optimizers
outPrefix = outputDirectory+task+optVars[0]
if args.opt == 'snopt':
    optOptions = {
        'Major feasibility tolerance':  1.0e-6,   # tolerance for constraint
        'Major optimality tolerance':   1.0e-6,   # tolerance for gradient 
        'Minor feasibility tolerance':  1.0e-6,   # tolerance for constraint
        'Verify level':                 -1,
        'Function precision':           1.0e-6,
        'Major iterations limit':       20,
        'Nonderivative linesearch':     None, 
        'Major step limit':             2.0,
        'Penalty parameter':            0.0, # initial penalty parameter
        'Print file':                   os.path.join(outPrefix+'_SNOPT_print.out'),
        'Summary file':                 os.path.join(outPrefix+'_SNOPT_summary.out')
    }
elif args.opt == 'psqp':
    optOptions = {
        'TOLG':                         1.0e-6,   # tolerance for gradient 
        'TOLC':                         1.0e-6,   # tolerance for constraint
        'MIT':                          20,       # max optimization iterations
        'IFILE':                        os.path.join(outPrefix+'_PSQP.out')
    }
elif args.opt == 'slsqp':
    optOptions = {
        'ACC':                          1.0e-5,   # convergence accuracy
        'MAXIT':                        20,       # max optimization iterations
        'IFILE':                        os.path.join(outPrefix+'_SLSQP.out')
    }
elif args.opt == 'ipopt':
    optOptions = {
        'tol':                          1.0e-6,   # convergence accuracy
        'max_iter':                     20,       # max optimization iterations
        'output_file':                  os.path.join(outPrefix+'_IPOPT.out')
    }
else:
    print("opt arg not valid!")
    exit(0)


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
DVGeoChild.addGeoDVLocal('shapex', lower=-0.01, upper=0.01, axis='x', scale=1.0e5,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapey', lower=-0.01, upper=0.01, axis='y', scale=1.0e5,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapez', lower=-0.01, upper=0.01, axis='z', scale=1.0e5,pointSelect=PS)

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
if task.lower()=='opt':
    optProb = Optimization('opt', optFuncs.aeroFuncs, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj('VMS', scale=1.0)
    # Add physical constraints
    #optProb.addCon('CL',lower=0.5,upper=0.5,scale=1)

    if gcomm.rank == 0:
        print optProb

    opt = OPT(args.opt, options=optOptions)
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, sens=optFuncs.aeroFuncsSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print sol

elif task.lower() == 'plotsensmap':

    optFuncs.plotSensMap(runAdjoint=True)

elif task.lower() == 'testsensuin':

    optFuncs.testSensUIn(normStatesList=[True],deltaStateList=[1e-5],deltaUInList=[1e-3])
        
elif task.lower() == 'testsensshape':

    optFuncs.testSensShape(normStatesList=[True],deltaUList=[1e-5],deltaXList=[1e-4])

elif task.lower() == 'xdv2xv':

    optFuncs.xDV2xV()

else:
    print("task arg not found!")
    exit(0)


