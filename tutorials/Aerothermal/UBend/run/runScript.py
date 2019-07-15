#!/usr/bin/env python
"""
DAFoam run script for the U bend duct case with radiation and bouyancy
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

    # design surfaces and cost functions 
    'designsurfacefamily':     'designsurfaces', 
    'designsurfaces':          ['ubend','ubendup','sym'], 
    'objfuncs':                ['CPL','NUS'],
    'objfuncgeoinfo':          [['inlet','outlet'],['ubend']],
    'referencevalues':         {'magURef':8.4,'ARef':1.0,'LRef':0.075,'pRef':0.0,'rhoRef':1.0},
    'userdefinedpatchinfo':    {'userDefinedPatch0': {'type':'box',
                                                      'stateName':'U',
                                                      'component':0,
                                                      'scale':1.0,
                                                      'centerX':0.47,
                                                      'centerY':0.057,
                                                      'centerZ':0,
                                                      'sizeX':0.01,
                                                      'sizeY':0.08,
                                                      'sizeZ':0.08}},

    # flow setup
    'adjointsolver':           'buoyantBoussinesqSimpleDAFoam',
    'flowcondition':           'Incompressible',
    'rasmodel':                'SpalartAllmarasFv3',
    'maxflowiters':            500, 
    'writeinterval':           500,
    'avgobjfuncs':             True,
    'avgobjfuncsstart':        400,
    'setflowbcs':              False,             
    'transproperties':         {'nu':1.5E-5,
                                'TRef':293.15,
                                'beta':3e-3,
                                'Pr':0.7,
                                'Prt':0.7,'g':[-9.81,0,0],'rhoRef':1.0,'CpRef':1005.0}, 
    'updatedefaultdicts':       {'radiationproperties':{'radiation':'on','radiationModel':'P1'}}, 
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
    'statescaling':            {'UScaling':8.4,
                                'phiScaling':1.0,
                                'p_rghScaling':32.0,
                                'GScaling':1000.0,
                                'nuTildaScaling':1.5e-3,
                                'TScaling':293.15},
    
    
    ########## misc setup ##########
    'mpispawnrun':             False,
    'restartopt':              False,
    'meshmaxnonortho':         75.0,
    'meshmaxskewness':         6.0,
    'stateresettol':           1e16,
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

# add max curvature constraint for inner bend section to avoid kinks
#DVCon.addCurvatureConstraint('./FFD/innerBendCurv.xyz',curvatureType='KSmean',KSCoeff=1e5,lower=0.0,upper=2.0,addToPyOpt=True,scaled=True)

# add max curvature constraint for inner bend section to avoid kinks
#DVCon.addCurvatureConstraint('./FFD/outerBendCurv.xyz',curvatureType='KSmean',KSCoeff=4e5,lower=0.0,upper=2.0,addToPyOpt=True,scaled=True)

# add max curvature constraint for inner bend section to avoid kinks
#DVCon.addCurvatureConstraint('./FFD/topBendCurv.xyz',curvatureType='KSmean',KSCoeff=2e4,lower=0.0,upper=5e-4,addToPyOpt=True,scaled=False)



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
weight1 = 0.2/1.200
weight2 = -0.8/70.0
obj1='CPL'
obj2='NUS'

def objCon(xDV):

    funcs={}
    funcs,fail = optFuncs.aeroFuncs(xDV)

    funcsMP = {}
    for key in funcs:
        funcsMP[key]=funcs[key]
    funcsMP['obj'] = weight1*funcs[obj1]+weight2*funcs[obj2]

    if gcomm.rank == 0:
        print ('Objective Functions MultiPoint: ',funcsMP)
    return funcsMP,fail

def objConSens(xDV,funcs):

    funcsSens={}
    funcsSens,fail = optFuncs.aeroFuncsSens(xDV,funcs)

    funcsSensMP = {}
    for key in funcsSens:
        funcsSensMP[key]=funcsSens[key]

    funcsSensMP['obj'] = {}
    for key in xDV.keys():
        nDV = len(xDV[key])
        funcsSensMP['obj'][key] = np.zeros(nDV)
        for i in range(nDV):
            funcsSensMP['obj'][key][i] = weight1*funcsSens[obj1][key][i]+weight2*funcsSens[obj2][key][i]
    return funcsSensMP,fail

if task.lower()=='opt':

    optProb = Optimization('opt', objCon, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj('obj', scale=1)
    # Add physical constraints
    #optProb.addCon('CL',lower=0.5,upper=0.5,scale=1)

    if gcomm.rank == 0:
        print optProb

    opt = OPT(args.opt, options=optOptions)
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, sens=objConSens, storeHistory=histFile)
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



