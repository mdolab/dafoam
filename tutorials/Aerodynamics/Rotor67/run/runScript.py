#!/usr/bin/env python
"""
DAFoam run script for the Rotor 67 case
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
    'casename':                 'Rotor67_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,

    # design surfaces and cost functions 
    'designsurfacefamily':     'designSurfaces', 
    'designsurfaces':          ['bladeps','bladess','bladefillet','hub','shroud','inlet','outlet','per1','per2'], 
    'objfuncs':                ['TPR','MFR','CMZ'],
    'objfuncgeoinfo':          [['inlet','outlet'],['outlet'],['bladeps','bladess','bladefillet']],
    'referencevalues':         {'magURef':100.0,'ARef':1.0,'LRef':0.1,'pRef':101325.0,'TRef':288.15,'rhoRef':1.0},
    'derivuininfo':            {'stateName':'p','component':0,'type':'fixedValue','patchNames':['outlet']},
    # flow setup
    'adjointsolver':           'turboDAFoam',
    'rasmodel':                'SpalartAllmarasFv3',
    'flowcondition':           'Compressible',
    'maxflowiters':            5000, 
    'writeinterval':           5000,
    'setflowbcs':              True,
    'flowbcs':                 {'bc0':{'patch':'outlet','variable':'p','value':[130000.0]}},  

    # adjoint setup
    'adjgmresmaxiters':        1500,
    'adjgmresrestart':         1500,
    'adjgmresreltol':          1e-4,
    'reducerescon4jacmat':     True,
    'adjdvtypes':              ['FFD'],
    'epsderiv':                1.0e-6,
    'epsderivffd':             1.0e-4,
    'stateresettol':           1.0e16, 
    'adjpcfilllevel':          1,
    'adjjacmatordering':       'cell',
    'adjjacmatreordering':     'rcm',
    'normalizestates':         ['U','p','e','h','nuTilda','T','phi'],
    'normalizeresiduals':      ['URes','pRes','eRes','nuTildaRes','TRes','phiRes','hRes'],
    'maxresconlv4jacpcmat':    {'URes':2,'pRes':3,'eRes':2,'nuTildaRes':2,'TRes':2,'phiRes':1},
    'statescaling':            {'UScaling':100,
                                'pScaling':100000.0,
                                'TScaling':300.0,
                                'eScaling':10000,
                                'nuTildaScaling':1e-3,
                                'phiScaling':1,},
    'fvrelaxfactors':          {'fields':{'p':0.8,'rho':1.0},
                                'equations':{'U':0.2,
                                             'nuTilda':0.2,
                                             'h':0.2,
                                             'T':0.2,'p':0.8,
                                             'e':0.2}},
    'simplecontrol':           {'rhoLowerBound':'0.2',
                                'rhoUpperBound':'5.0',
                                'pLowerBound':'20000',
                                'pUpperBound':'500000',
                                'ULowerBound':'-800',
                                'UUpperBound':'800',
                                'hLowerBound':'5000',
                                'hUpperBound':'500000',
                                'consistent':'true',
                                'transonic':'true'}, 
    'divschemes':              {'div(phi,U)':'Gauss upwind',
                                'div(phi,e)':'Gauss upwind',
                                'div(phi,h)':'Gauss upwind',
                                'div(phi,nuTilda)':'Gauss upwind',
                                'default':'none',
                                'div(((rho*nuEff)*dev2(T(grad(U)))))': 'Gauss linear',
                                'div(phi,Ekp)': 'Gauss upwind',
                                'div(phi,K)': 'Gauss upwind',
                                'div(phid,p)':'Gauss upwind',
                                'div(pc)':'Gauss upwind',
                                'div((p*(U-URel)))': 'Gauss linear',
                                'div((-devRhoReff.T()&U))':'Gauss linear'},
    'thermotype':               {'type':'hePsiThermo',
                                 'mixture':'pureMixture',
                                 'thermo':'hConst',
                                 'transport':'const',
                                 'equationOfState':'perfectGas',
                                 'specie':'specie',
                                 'energy':'sensibleEnthalpy'},
    'mrfproperties':            {'active':'true',
                                'selectionmode':'cellZone',
                                'cellzone':'region0',
                                'nonrotatingpatches':['per1','per2','inlet','outlet'],
                                'axis':[0,0,1],
                                'origin':[0,0,0],
                                'omega':-1680},
    ########## misc setup ##########
    'mpispawnrun':             False,
    'restartopt':              False,
    'meshmaxnonortho':         75.0,
    'meshmaxskewness':         6.0,
    'trasonicpcoption':        1,
    'preservepatches':         ['per1','per2'],
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
#DVGeoChild.addGeoDVLocal('shapex', lower=-0.005, upper=0.005, axis='x', scale=1.0,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapey', lower=-0.005, upper=0.005, axis='y', scale=1.0,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapez', lower=-0.005, upper=0.005, axis='z', scale=1.0,pointSelect=PS)
#DVGeo.addGeoDVSectionLocal('shapes', secIndex='i', axis=1, lower=-0.01, upper=0.01, scale=1,volList=0,pointSelect=PS)
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
    optProb.addObj('CMZ', scale=1)
    # Add physical constraints
    optProb.addCon('TPR',lower=1.46,upper=1.46,scale=1)
    optProb.addCon('MFR',lower=1.75,upper=1.75,scale=1)

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



