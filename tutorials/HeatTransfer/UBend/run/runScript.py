#!/usr/bin/env python
"""
DAFoam run script for the U bend duct case
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
    'casename':                 'UBendDuct_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,
    'writelinesearch':          False,

    # design surfaces and cost functions 
    'designsurfacefamily':     'designsurfaces', 
    'designsurfaces':          ['ubend','ubendup','sym'], 
    'objfuncs':                ['NUS'],
    'objfuncgeoinfo':          [['ubend']],
    'referencevalues':         {'magURef':8.4,'ARef':1.0,'LRef':0.075,'pRef':0.0,'rhoRef':1.0},

    # flow setup
    'adjointsolver':           'simpleTDAFoam',
    'flowcondition':           'Incompressible',
    'rasmodel':                'SpalartAllmarasFv3',
    'rasmodelparameters':      {'kMin':1e-16,'omegaMin':1e-16,'epsilonMin':1e-16,'nuTildaMin':1e-16}, 
    'maxflowiters':            1000, 
    'writeinterval':           1000,
    'avgobjfuncs':             False,
    'avgobjfuncsstart':        1600,
    'divschemes':           {'default':'none',
                             'div(phi,U)': 'bounded Gauss linearUpwindV grad(U)',
                             'div(phi,nuTilda)': 'bounded Gauss upwind',
                             'div(phi,k)': 'bounded Gauss upwind',
                             'div(phi,omega)': 'bounded Gauss upwind',
                             'div(phi,T)': 'bounded Gauss upwind',
                             'div((nuEff*dev2(T(grad(U)))))':'Gauss linear',
                             'div(pc)':'bounded  Gauss upwind'},
    'laplacianschemes':      {'default':'Gauss linear  corrected'},
    'sngradschemes':         {'default':'corrected '},
    'gradschemes':           {'default':'Gauss linear','grad(U)': 'cellLimited Gauss linear 1.000000',
                              'grad(nuTilda)':'cellLimited Gauss linear 1.000000'},
    'setflowbcs':              False,  
    'inletpatches':            ['inlet'],
    'outletpatches':           ['outlet'],
    'flowbcs':                 {'bc0':{'patch':'inlet','variable':'U','value':[8.4,0.0,0.0]},
                                'bc1':{'patch':'outlet','variable':'p','value':[0.0]},
                                'bc2':{'patch':'inlet','variable':'k','value':[0.265]},
                                'bc3':{'patch':'inlet','variable':'omega','value':[1764.0]},
                                'bc4':{'patch':'inlet','variable':'epsilon','value':[42.0]},
                                'bc5':{'patch':'inlet','variable':'nuTilda','value':[1.5e-4]},
                                'bc6':{'patch':'inlet','variable':'T','value':[293.15]},
                                'useWallFunction':'false'},               
    'transproperties':         {'nu':1.5E-5,
                                'TRef':293.15,
                                'beta':3e-3,
                                'Pr':0.7,
                                'Prt':0.7,'rhoRef':1.0,'CpRef':1005.0}, 
    # adjoint setup
    'adjgmresmaxiters':        1500,
    'adjgmresrestart':         1500,
    'adjgmresreltol':          1e-5,
    'adjdvtypes':              ['FFD'], 
    'epsderiv':                1.0e-6, 
    'epsderivffd':             1.0e-5, 
    'adjpcfilllevel':          2, 
    'adjjacmatordering':       'state',
    'adjjacmatreordering':     'rcm',
    'normalizestates':         ['U','p','p_rgh','phi','k','omega','nuTilda','T','epsilon','G'],
    'normalizeresiduals':      ['URes','pRes','p_rghRes','phiRes','kRes','omegaRes','nuTildaRes','epsilonRes','TRes','GRes'],
    'maxresconlv4jacpcmat':    {'URes':2,'pRes':2,'p_rghRes':2,'phiRes':1,'kRes':2,'omegaRes':2,'epsilonRes':2,'TRes':2,'nuTildaRes':2,'GRes':2},
    'statescaling':            {'UScaling':8.4,
                                'pScaling':32.0,
                                'phiScaling':1.0,
                                'p_rghScaling':32.0,
                                'GScaling':1000.0,
                                'nuTildaScaling':1.5e-3,
                                'kScaling':0.265,
                                'omegaScaling':1764.0,
                                'epsilonScaling':42.0,
                                'TScaling':293.15},
    
    
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
    'symmetryPlanes':          [[[0.,0., 0.],[0., 0., -1.]]], 
}

# options for optimizers
outPrefix = outputDirectory+task+optVars[0]
if args.opt == 'snopt':
    optOptions = {
        'Major feasibility tolerance':  1.0e-7,   # tolerance for constraint
        'Major optimality tolerance':   1.0e-7,   # tolerance for gradient 
        'Minor feasibility tolerance':  1.0e-7,   # tolerance for constraint
        'Verify level':                 -1,
        'Function precision':           1.0e-7,
        'Nonderivative linesearch':     None, 
        'Major step limit':             2.0,
        'Penalty parameter':            0.0, # initial penalty parameter
        'Print file':                   os.path.join(outPrefix+'_SNOPT_print.out'),
        'Summary file':                 os.path.join(outPrefix+'_SNOPT_summary.out')
    }
elif args.opt == 'psqp':
    optOptions = {
        'TOLG':                         1.0e-7,   # tolerance for gradient 
        'TOLC':                         1.0e-7,   # tolerance for constraint
        'MIT':                          25,       # max optimization iterations
        'IFILE':                        os.path.join(outPrefix+'_PSQP.out')
    }
elif args.opt == 'slsqp':
    optOptions = {
        'ACC':                          1.0e-7,   # convergence accuracy
        'MAXIT':                        25,       # max optimization iterations
        'IFILE':                        os.path.join(outPrefix+'_SLSQP.out')
    }
elif args.opt == 'ipopt':
    optOptions = {
        'tol':                          1.0e-7,   # convergence accuracy
        'max_iter':                     25,       # max optimization iterations
        'output_file':                  os.path.join(outPrefix+'_IPOPT.out')
    }
else:
    print("opt arg not valid!")
    exit(0)


# =================================================================================================
# DVGeo
# =================================================================================================
DVGeo = DVGeometry('./FFD/UBendDuctFFDSym.xyz')
x = [0.0,0.6]
y = [0.05,0.05]
z = [0.05,0.05]
c1 = pySpline.Curve(x=x, y=y, z=z, k=2)
DVGeo.addRefAxis('bodyAxis', curve = c1,axis='z')
# Select points
pts=DVGeo.getLocalIndex(0) 
# shapez
indexList=[]
indexList.extend(pts[7:16,:,-1].flatten())
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapez', lower=-0.04, upper=0.0077, axis='z', scale=1.0, pointSelect=PS,config='configz')
# shapeyouter
indexList=[]
indexList.extend(pts[7:16,-1,:].flatten())
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapeyouter', lower=-0.02, upper=0.02, axis='y', scale=1.0, pointSelect=PS,config='configyouter')
# shapeyinner
indexList=[]
indexList.extend(pts[7:16,0,:].flatten())
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapeyinner', lower=-0.04, upper=0.04, axis='y', scale=1.0, pointSelect=PS,config='configyinner')
# shapexinner
indexList=[]
indexList.extend(pts[7:16,0,:].flatten())
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapexinner', lower=-0.04, upper=0.04, axis='x', scale=1.0, pointSelect=PS,config='configxinner')

# shapexouter1
indexList=[]
indexList.extend(pts[9,-1,:].flatten())
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapexouter1', lower=-0.05, upper=0.05, axis='x', scale=1.0, pointSelect=PS,config='configxouter1')

# shapexouter2
indexList=[]
indexList.extend(pts[10,-1,:].flatten())
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapexouter2', lower=-0.05, upper=0.0, axis='x', scale=1.0, pointSelect=PS,config='configxouter2')

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
DVCon.writeSurfaceTecplot('trisurface.dat')


if task.lower()=='opt':
    #Create a linear constraint so that the curvature at the symmetry plane is zero
    indSetA = [] 
    indSetB = []
    for i in xrange(7,16,1):
        indSetA.append(pts[i,0,0])
        indSetB.append(pts[i,0,1])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=0.0,upper=0.0,config='configyinner')

    indSetA = []
    indSetB = []
    for i in xrange(7,16,1):
        indSetA.append(pts[i,-1,0])
        indSetB.append(pts[i,-1,1])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=0.0,upper=0.0,config='configyouter')

    indSetA = []
    indSetB = []
    for i in xrange(7,16,1):
        indSetA.append(pts[i,0,0])
        indSetB.append(pts[i,0,1])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=0.0,upper=0.0,config='configxinner')

    indSetA = []
    indSetB = []
    for i in [9]:
        indSetA.append(pts[i,-1,0])
        indSetB.append(pts[i,-1,1])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=0.0,upper=0.0,config='configxouter1')

    indSetA = []
    indSetB = []
    for i in [10]:
        indSetA.append(pts[i,-1,0])
        indSetB.append(pts[i,-1,1])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=0.0,upper=0.0,config='configxouter2')
    # linear constraint to make sure the inner bend does not intersect
    minInnerBend=0.005
    indSetA = []
    indSetB = []
    for k in range(3):
        indSetA.append(pts[8,0,k])
        indSetB.append(pts[14,0,k])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=-0.03300+minInnerBend,upper=0.03300-minInnerBend,config='configyinner')
    indSetA = []
    indSetB = []
    for k in range(3):
        indSetA.append(pts[9,0,k])
        indSetB.append(pts[13,0,k])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=-0.02853+minInnerBend,upper=0.02853-minInnerBend,config='configyinner')
    indSetA = []
    indSetB = []
    for k in range(3):
        indSetA.append(pts[10,0,k])
        indSetB.append(pts[12,0,k])
    DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=-0.01635+minInnerBend,upper=0.01635-minInnerBend,config='configyinner')

DVCon.writeTecplot('DVConstraints.dat')

# =================================================================================================
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
    optProb.addObj('NUS', scale=-1)
    # Add physical constraints
    #optProb.addCon('CL',lower=0.5,upper=0.5,scale=1)

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

    optFuncs.testSensUIn(normStatesList=[True],deltaUList=[1e-7])
        
elif task.lower() == 'testsensshape':

    optFuncs.testSensShape(normStatesList=[True],deltaUList=[1e-6],deltaXList=[1e-5])

elif task.lower() == 'xdv2xv':

    optFuncs.xDV2xV()

else:
    print("task arg not found!")
    exit(0)



