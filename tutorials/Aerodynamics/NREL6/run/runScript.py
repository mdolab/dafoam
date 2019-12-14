#!/usr/bin/env python
"""
DAFoam run script for the NREL6 case
"""

# =================================================================================================
# Imports
# =================================================================================================
import os,time,shutil
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
    'casename':                 'NREL6_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,

    # design surfaces and cost functions 
    'designsurfacefamily':     'designSurfaces', 
    'designsurfaces':          ['blade'], 
    'objfuncs':                ['CMX'],
    'objfuncgeoinfo':          [['blade']],
    'referencevalues':         {'magURef':1.0,'ARef':1.0,'LRef':1.0,'pRef':101000.0,'TRef':288.15,'rhoRef':1.0},
    # flow setup
    'adjointsolver':           'turboDAFoam',
    'rasmodel':                'SpalartAllmarasFv3',
    'flowcondition':           'Compressible',
    'maxflowiters':            500, 
    'writeinterval':           500,
    'setflowbcs':              True,
    'flowbcs':                 {'bc0':{'patch':'inlet','variable':'U','value':[7.0,0.0,0.0]}},  

    # adjoint setup
    'adjgmresmaxiters':        1000,
    'adjgmresrestart':         1000,
    'adjgmresreltol':          1.0e-6,
    'stateresettol':           1.0,
    'adjdvtypes':              ['FFD'],
    'epsderiv':                1.0e-6,
    'epsderivffd':             1.0e-4,
    'adjpcfilllevel':          1,
    'adjjacmatordering':       'cell',
    'adjjacmatreordering':     'rcm',
    'normalizestates':         ['U','p','nuTilda','phi','T'],
    'normalizeresiduals':      ['URes','pRes','nuTildaRes','phiRes','TRes'],
    'statescaling':            {'UScaling':10,
                                'pScaling':10000.0,
                                'TScaling':300.0,
                                'nuTildaScaling':1e-3,
                                'phiScaling':1,},
    'mrfproperties':           {'active':'true',
                                'selectionmode':'cellZone',
                                'cellzone':'region0',
                                'nonrotatingpatches':['sides','inlet','outlet'],
                                'axis':[1,0,0],
                                'origin':[0,0,0],
                                'omega':7.5},
    'thermotype':              {'type':'hePsiThermo',
                                'mixture':'pureMixture',
                                'thermo':'hConst',
                                'transport':'const',
                                'equationOfState':'perfectGas',
                                'specie':'specie',
                                'energy':'sensibleEnthalpy'},
    ########## misc setup ##########
    'mpispawnrun':             False,
    'restartopt':              False,
    'meshmaxnonortho':         75.0,
    'meshmaxskewness':         6.0,
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
        'Major iterations limit':       10,
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
FFDFile = './FFD/bodyFittedFFD.xyz'
DVGeo = DVGeometry(FFDFile)
# Setup curves for ref_axis
x = [ 0.0,0.00]
y = [ 0.0,0.0]
z = [ -5.0,5.0]
c1 = pySpline.Curve(x=x, y=y, z=z, k=2)
DVGeo.addRefAxis('bladeAxis', curve = c1,axis='x')
# FFD shape
pts=DVGeo.getLocalIndex(0)
indexList=pts[:,:,:].flatten()  # select the top layer FFD starts with i=1
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapex1', lower=-0.5, upper=0.5, axis='x', scale=1.0,pointSelect=PS)

pts=DVGeo.getLocalIndex(1)
indexList=pts[:,:,:].flatten()  # select the top layer FFD starts with i=1
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapex2', lower=-0.5, upper=0.5, axis='x', scale=1.0,pointSelect=PS)

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
    optProb.addObj('CMX', scale=-1)
    # Add physical constraints

    if gcomm.rank == 0:
        print optProb

    opt = OPT(args.opt, options=optOptions)
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, sens=optFuncs.aeroFuncsSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print sol

elif task.lower() == 'run':

    optFuncs.run()

elif task.lower() == 'plotsensmap':

    optFuncs.plotSensMap(runAdjoint=True)

elif task.lower() == 'testsensuin':

    optFuncs.testSensUIn(normStatesList=[True],deltaStateList=[1e-6],deltaUInList=[1e-3])
        
elif task.lower() == 'testsensshape':

    optFuncs.testSensShape(normStatesList=[True],deltaUList=[1e-6],deltaXList=[1e-4])

elif task.lower() == 'xdv2xv':

    optFuncs.xDV2xV()

else:
    print("task arg not found!")
    exit(0)



