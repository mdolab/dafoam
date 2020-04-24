#!/usr/bin/env python
"""
DAFoam run script for the JBC case
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


# =================================================================================================
# Input
# =================================================================================================
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
    'casename':                 'JBC_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,

    # design surfaces and cost functions 
    'designsurfacefamily':     'designSurfaces', 
    'designsurfaces':          ['hull'], 
    'objfuncs':                ['CD','VARV'],
    'objfuncgeoinfo':          [['hull'],['userDefinedVolume0']],
    'userdefinedvolumeinfo':   {'userDefinedVolume0':{'type':'annulus',
                                                      'stateName':'U',
                                                      'component':0,
                                                      'scale':1.0,
                                                      'centerX':6.886,
                                                      'centerY':0.0,
                                                      'centerZ':-0.282898,
                                                      'width':0.005,
                                                      'radiusInner':0.0207,
                                                      'radiusOuter':0.1015,
                                                      'axis':'x'},
                                'userDefinedVolume1':{'type':'annulus',
                                                      'stateName':'U',
                                                      'component':0,
                                                      'scale':1.0,
                                                      'centerX':6.912,
                                                      'centerY':0.0,
                                                      'centerZ':-0.282898,
                                                      'width':0.015, # change 0.05 to 0.015 for large mesh
                                                      'radiusInner':0.0207,
                                                      'radiusOuter':0.1015,
                                                      'axis':'x'}},
    'referencevalues':         {'magURef':1.179,'ARef':12.2206,'LRef':7.0,'pRef':0.0,'rhoRef':1.0},
    'dragdir':                 [1.0,0.0,0.0],


    # flow setup
    'adjointsolver':           'simpleDAFoam',
    'rasmodel':                'SpalartAllmarasFv3',
    'flowcondition':           'Incompressible', 
    'maxflowiters':            1000, 
    'writeinterval':           1000,
    'setflowbcs':              False,      
    'transproperties':         {'nu':1.107E-6,
                                'TRef':293.15,
                                'beta':3e-3,
                                'Pr':0.7,
                                'Prt':0.85}, 
    
    # actuator disk
    'actuatoractive':          0,
    'actuatorvolumenames':     ['userDefinedVolume1'],
    'actuatorthrustcoeff':     [0.00226],
    'actuatorpoverd':          [0.75],
    'actuatorrotationdir':     ['right'],
    'actuatoradjustthrust':    0,
 
    # adjoint setup
    'adjgmresmaxiters':        1500,
    'adjgmresrestart':         1500,
    'adjdvtypes':              ['FFD'], 
    'correctwalldist':         True,
    'epsderiv':                1.0e-7, 
    'epsderivffd':             1.0e-4, 
    'adjpcfilllevel':          1, 
    'adjjacmatordering':       'cell',
    'adjjacmatreordering':     'rcm',
    'statescaling':            {'UScaling':1.0,
                                'pScaling':0.5,
                                'phiScaling':1.0,
                                'nuTildaScaling':1.0e-4},
    
    
    ########## misc setup ##########
    'mpispawnrun':             False,
    'restartopt':              False,
    'meshmaxskewness':         10.0,
    'meshmaxnonortho':         80.0,
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    'gridFile':                os.getcwd(),
    'fileType':                'openfoam',
    # point and normal for the symmetry plane
    'symmetryPlanes':          [[[0.,0., 0.],[0., 0., 1.]]], 
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
        'ACC':                          1.0e-6,   # convergence accuracy
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
FFDFile = './FFD/JBCFFD_32.xyz'
DVGeo = DVGeometry(FFDFile)
x = [0.0,7.0]
y = [0.,0.]
z = [-0.1,-0.1]
c1 = pySpline.Curve(x=x, y=y, z=z, k=2)
DVGeo.addRefAxis('bodyAxis', curve = c1,axis='z')

# Select points
iVol=0
pts=DVGeo.getLocalIndex(iVol)
# shapez
indexList=[]
indexList.extend(pts[8:12,0,0:4].flatten())
indexList.extend(pts[8:12,-1,0:4].flatten())
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapey', lower=-0.5, upper=0.5, axis='y', scale=1.0, pointSelect=PS)

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
#DVCon.writeSurfaceTecplot('trisurface.dat') 

#Create reflection constraint
pts=DVGeo.getLocalIndex(0) 
indSetA = [] 
indSetB = []
for i in xrange(8,12,1): 
    for k in xrange(0,4,1): 
        indSetA.append(pts[i,0,k])
        indSetB.append(pts[i,-1,k])
DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=1.0,lower=0.0,upper=0.0)


#Create a volume constraint
# Volume constraints
leList = [[4.90000000,	0.00000000,	-0.41149880],
          [4.90000000,	0.00000000,	-0.40347270],
          [4.90000000,	0.00000000,	-0.38803330],
          [4.90000000,	0.00000000,	-0.36534750],
          [4.90000000,	0.00000000,	-0.33601030],
          [4.90000000,	0.00000000,	-0.31016020],
          [4.90000000,	0.00000000,	-0.28327050],
          [4.90000000,	0.00000000,	-0.26248810],
          [4.90000000,	0.00000000,	-0.24076410],
          [4.90000000,	0.00000000,	-0.20933480],
          [4.90000000,	0.00000000,	-0.17458840],
          [4.90000000,	0.00000000,	-0.14233480],
          [4.90000000,	0.00000000,	-0.11692880],
          [4.90000000,	0.00000000,	-0.09984235],
          [4.90000000,	0.00000000,	-0.08874606],
          [4.90000000,	0.00000000,	-0.07969946],
          [4.90000000,	0.00000000,	-0.06954966],
          [4.90000000,	0.00000000,	-0.05864429],
          [4.90000000,	0.00000000,	-0.04829308],
          [4.90000000,	0.00000000,	-0.03831457],
          [4.90000000,	0.00000000,	-0.02430242],
          [4.90000000,	0.00000000,	-0.00100000]]
teList = [[6.70332700,	0.00000000,	-0.41149880],
          [6.73692400,	0.00000000,	-0.40347270],
          [6.76842800,	0.00000000,	-0.38803330],
          [6.79426000,	0.00000000,	-0.36534750],
          [6.81342600,	0.00000000,	-0.33601030],
          [6.83648300,	0.00000000,	-0.31016020],
          [6.85897100,	0.00000000,	-0.28327050],
          [6.83593600,	0.00000000,	-0.26248810],
          [6.80929800,	0.00000000,	-0.24076410],
          [6.79395800,	0.00000000,	-0.20933480],
          [6.79438900,	0.00000000,	-0.17458840],
          [6.80874100,	0.00000000,	-0.14233480],
          [6.83265000,	0.00000000,	-0.11692880],
          [6.86250800,	0.00000000,	-0.09984235],
          [6.89566400,	0.00000000,	-0.08874606],
          [6.92987100,	0.00000000,	-0.07969946],
          [6.96333200,	0.00000000,	-0.06954966],
          [6.99621200,	0.00000000,	-0.05864429],
          [7.02921500,	0.00000000,	-0.04829308],
          [7.06253200,	0.00000000,	-0.03831457],
          [7.09456600,	0.00000000,	-0.02430242],
          [7.12000000,	0.00000000,	-0.00100000]]
DVCon.addVolumeConstraint(leList,teList,nSpan=25,nChord=50,lower=1.0,upper=1.0)


# Thickness constraint for lateral thickness
leList = [[5.01,0.0000,-0.001],
          [5.01,0.0000,-0.410]]
teList = [[6.2,0.0000,-0.001],
          [6.2,0.0000,-0.410]]
DVCon.addThicknessConstraints2D(leList, teList,nSpan=8,nChord=5,lower=1e-3,upper=1.1251,scaled=False)


# Thickness constraint for propeller shaft
leList = [[6.8,0.0000,-0.302],
          [6.8,0.0000,-0.265]]
teList = [[6.865,0.0000,-0.302],
          [6.865,0.0000,-0.265]]
DVCon.addThicknessConstraints2D(leList, teList,nSpan=5,nChord=5,lower=1.0,upper=10.0)

# Curvature constraints
DVCon.addCurvatureConstraint('./FFD/hullCurv.xyz',curvatureType='KSmean',lower=0.0,upper=1.21,addToPyOpt=True,scaled=True)

#DVCon.writeTecplot('constraints.dat')   


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
weightCD = 0.9/0.001773
weightVAR = 0.1/0.001169

def objCon(xDV):

    funcs={}
    funcs,fail = optFuncs.aeroFuncs(xDV)

    funcsMP = {}
    for key in funcs:
        funcsMP[key]=funcs[key]
    funcsMP['obj'] = weightCD*funcs['CD']+weightVAR*funcs['VARV']

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
            funcsSensMP['obj'][key][i] = weightCD*funcsSens['CD'][key][i]+weightVAR*funcsSens['VARV'][key][i]
    return funcsSensMP,fail

if task.lower()=='opt':
    optProb = Optimization('opt', objCon, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj('obj', scale=1)
    # Add physical constraints

    if gcomm.rank == 0:
        print(optProb)

    opt = OPT(args.opt, options=optOptions)
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, sens=objConSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

elif task.lower() == 'run':

    optFuncs.run()

elif task.lower() == 'testsensuin':

    optFuncs.testSensUIn(normStatesList=[True],deltaUList=[1e-7],deltaUInList=[1e-7])
        
elif task.lower() == 'testsensshape':

    optFuncs.testSensShape(normStatesList=[True],deltaUList=[1e-7],deltaXList=[1e-4])

elif task.lower() == 'xdv2xv':

    optFuncs.xDV2xV()

else:
    print("task arg not found!")
    exit(0)


