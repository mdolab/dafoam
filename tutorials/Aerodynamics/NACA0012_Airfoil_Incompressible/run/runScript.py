#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed
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

pRef       = 0.0
rhoRef     = 1.0
UmagIn     = 35.0
LRef       = 1.0
ARef       = 1.0*0.1
CofR       = [0.25,0,0]

CL_star    = 0.375
alpha0     = 3.579107

def calcUAndDir(UIn,alpha1):
    dragDir = [ np.cos(alpha1*np.pi/180),np.sin(alpha1*np.pi/180),0]
    liftDir = [-np.sin(alpha1*np.pi/180),np.cos(alpha1*np.pi/180),0]
    inletU = [float(UIn*np.cos(alpha1*np.pi/180)),float(UIn*np.sin(alpha1*np.pi/180)),0]
    return inletU, dragDir, liftDir

inletu0, dragdir0, liftdir0 = calcUAndDir(UmagIn,alpha0)

# Set the parameters for optimization
aeroOptions = {
    # output options
    'casename':                 'NACA0012_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,

    # design surfaces and cost functions 
    'designsurfacefamily':     'designSurfaces', 
    'designsurfaces':          ['wing','wingte'], 
    'objfuncs':                ['CD','CL'],
    'objfuncgeoinfo':          [['wing','wingte'],['wing','wingte']],
    'referencevalues':         {'magURef':UmagIn,'ARef':ARef,'LRef':LRef,'pRef':pRef,'rhoRef':rhoRef},
    'liftdir':                 liftdir0,
    'dragdir':                 dragdir0,
    'cofr':                    CofR,

    # flow setup
    'adjointsolver':           'simpleDAFoam',
    'rasmodel':                'SpalartAllmarasFv3',
    'flowcondition':           'Incompressible',
    'maxflowiters':            800, 
    'writeinterval':           800,
    'setflowbcs':              True,
    'inletpatches':            ['inout'],
    'outletpatches':           ['inout'],
    'flowbcs':                 {'bc0':{'patch':'inout','variable':'U','value':inletu0},
                                'useWallFunction':'true'},

    # adjoint setup
    'adjgmresmaxiters':        1000,
    'adjgmresrestart':         1000,
    'adjgmresreltol':          1e-6,
    'adjdvtypes':              ['FFD'], 
    'epsderiv':                1.0e-6, 
    'epsderivffd':             1.0e-3, 
    'adjpcfilllevel':          1, 
    'adjjacmatordering':       'cell',
    'adjjacmatreordering':     'natural',
    'statescaling':            {'UScaling':UmagIn,
                                'pScaling':UmagIn*UmagIn/2,
                                'nuTildaScaling':1e-4,
                                'phiScaling':1},
    
    ########## misc setup ##########
    'mpispawnrun':             False,
    'restartopt':              False,
    'meshmaxnonortho':         70.0,
    'meshmaxskewness':         10.0,
    'meshmaxaspectratio':      2000.0, 

}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    'gridFile':                os.getcwd(),
    'fileType':                'openfoam',
    # point and normal for the symmetry plane
    'symmetryPlanes':          [[[0.,0., 0.],[0., 0., 1.]],[[0.,0., 0.1],[0., 0., 1.]]], 
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
FFDFile = './FFD/wingFFD.xyz'
DVGeo = DVGeometry(FFDFile)

# ref axis
x = [0.25,0.25]
y = [0.00,0.00]
z = [0.00,0.10]
c1 = pySpline.Curve(x=x, y=y, z=z, k=2)
DVGeo.addRefAxis('bodyAxis', curve = c1,axis='z')

def alpha(val, geo=None):
    inletu, dragdir, liftdir = calcUAndDir(UmagIn,np.real(val))

    flowbcs=CFDSolver.getOption('flowbcs')
    for key in flowbcs.keys():
        if key == 'useWallFunction':
            continue
        if flowbcs[key]['variable'] == 'U':
            flowbcs[key]['value'] = inletu
    CFDSolver.setOption('setflowbcs',True)
    CFDSolver.setOption('flowbcs',flowbcs)
    CFDSolver.setOption('dragdir',dragdir)
    CFDSolver.setOption('liftdir',liftdir)


# select points
pts=DVGeo.getLocalIndex(0) 
indexList=pts[:,:,:].flatten()
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal('shapey',lower=-1.0, upper=1.0,axis='y',scale=1.0,pointSelect=PS)
DVGeo.addGeoDVGlobal('alpha', alpha0,alpha,lower=0, upper=10., scale=1.0)

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

leList = [[1e-4,0.0,1e-4],[1e-4,0.0,0.1-1e-4]]
teList = [[0.998-1e-4,0.0,1e-4],[0.998-1e-4,0.0,0.1-1e-4]]

DVCon.addVolumeConstraint(leList, teList, nSpan=2, nChord=50,lower=1.0,upper=3, scaled=True)
DVCon.addThicknessConstraints2D(leList, teList,nSpan=2,nChord=50,lower=0.8, upper=3.0,scaled=True)

#Create a linear constraint so that the curvature at the symmetry plane is zero
pts1=DVGeo.getLocalIndex(0)
indSetA = [] 
indSetB = []
for i in range(10):
    for j in [0,1]:
        indSetA.append(pts1[i,j,1])
        indSetB.append(pts1[i,j,0])
DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=-1.0,lower=0.0,upper=0.0)

#Create a linear constraint so that the leading and trailing edges do not change
pts1=DVGeo.getLocalIndex(0)
indSetA = []
indSetB = []
for i in [0,9]:
    for k in [0]: # do not constrain k=1 because it is linked in the above symmetry constraint
        indSetA.append(pts1[i,0,k])
        indSetB.append(pts1[i,1,k])
DVCon.addLinearConstraintsShape(indSetA,indSetB,factorA=1.0,factorB=1.0,lower=0.0,upper=0.0)

# ================================================================================================
# optFuncs
# =================================================================================================
optFuncs.CFDSolver = CFDSolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# =================================================================================================
# optFuncs
# =================================================================================================
def aeroFuncs(xDV):
    """
    Update the design surface and run the flow solver to get objective function values.
    """

    if gcomm.rank == 0:
        print ('\n')
        print ('+--------------------------------------------------------------------------+')
        print ('|                    Evaluating Objective Function                         |')
        print ('+--------------------------------------------------------------------------+')
        print ('Design Variables: ',xDV)

    a = time.time()

    # Setup an empty dictionary for the evaluated function values
    funcs = {}

    # Set the current design variables in the DV object
    DVGeo.setDesignVars(xDV)
    CFDSolver.setDesignVars(xDV)

    # Evaluate the geometric constraints and add them to the funcs dictionary
    DVCon.evalFunctions(funcs)

    alphaVal = xDV['alpha'].real
    alpha(alphaVal)
    if gcomm.rank == 0:
        print('Inlet Info:')
        print('alpha: ',alphaVal)
        print('dragDir: ',CFDSolver.getOption('dragdir'))
        print('liftDir: ',CFDSolver.getOption('liftdir'))
        print('flowBCs: ',CFDSolver.getOption('flowbcs'))

    # Solve the CFD problem
    CFDSolver()

    # Populate the required values from the CFD problem
    CFDSolver.evalFunctions(funcs,evalFuncs=evalFuncs)

    b = time.time()

    # Print the current solution to the screen
    if gcomm.rank == 0:
        print ('Objective Functions: ',funcs)
        print ('Flow Runtime: ',b-a)

    fail = funcs['fail']

    # flush the output to the screen/file
    sys.stdout.flush()

    return funcs,fail

def aeroFuncsSens(xDV,funcs):
    """
    Run the adjoint solver and get objective function sensitivities.
    """

    if gcomm.rank == 0:
        print ('\n')
        print ('+--------------------------------------------------------------------------+')
        print ('|                Evaluating Objective Function Sensitivity                 |')
        print ('+--------------------------------------------------------------------------+')

    a = time.time()

    # Setup an empty dictionary for the evaluated derivative values
    funcsSens={}

    # Evaluate the geometric constraint derivatives
    DVCon.evalFunctionsSens(funcsSens)

    # Solve the adjoint
    CFDSolver.solveADjoint()

    # Evaluate the CFD derivatives
    CFDSolver.evalFunctionsSens(funcsSens,evalFuncs=evalFuncs)

    # need to add alpha funcsSens, here we use FD
    if gcomm.rank == 0:
        print("Evaluating alpha sens...")
    epsAlpha = 1.0e-2
    alphaVal = xDV['alpha']
    alphaVal += epsAlpha
    alpha(alphaVal)
    # Solve the CFD problem
    CFDSolver()
    # reset perturbation
    alphaVal -= epsAlpha
    alpha(alphaVal)

    funcsP={}
    CFDSolver.evalFunctions(funcsP,evalFuncs=evalFuncs)
    for evalFunc in evalFuncs:
        dFuncsdAlpha = ( funcsP[evalFunc] - funcs[evalFunc] ) / epsAlpha
        funcsSens[evalFunc]['alpha'] = dFuncsdAlpha

    b = time.time()

    # Print the current solution to the screen
    if gcomm.rank == 0:
        print('Objective Functions Perturbed: ',funcsP)
        print('Objective Function Sensitivity: ',funcsSens)
        print('Adjoint Runtime: ',b-a)

    fail = funcsSens['fail']

    # flush the output to the screen/file
    sys.stdout.flush()

    return funcsSens,fail

# =================================================================================================
# Task
# =================================================================================================
if task.lower()=='opt':
    optProb = Optimization('opt', aeroFuncs, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj('CD', scale=1)
    # Add physical constraints
    optProb.addCon('CL',lower=CL_star,upper=CL_star,scale=1)

    if gcomm.rank == 0:
        print optProb

    opt = OPT(args.opt, options=optOptions)
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, sens=aeroFuncsSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print sol

elif task.lower() == 'run':

    optFuncs.run()

elif task.lower() == 'plotsensmap':

    optFuncs.plotSensMap(runAdjoint=True)

elif task.lower() == 'testsensuin':

    optFuncs.testSensUIn(normStatesList=[True],deltaStateList=[1e-5],deltaUInList=[1e-3])
        
elif task.lower() == 'testsensshape':

    optFuncs.testSensShape(normStatesList=[True],deltaUList=[1e-6],deltaXList=[1e-4])

elif task.lower() == 'xdv2xv':

    optFuncs.xDV2xV()

elif task.lower() == 'solvecl':

    CFDSolver.setOption('usecoloring',False)

    xDV0 = DVGeo.getValues()
    alpha0 = xDV0['alpha']

    for i in range(10):
        alpha(alpha0)
        # Solve the CFD problem
        CFDSolver()
        funcs={}
        CFDSolver.evalFunctions(funcs,evalFuncs=evalFuncs)
        CL0 = funcs['CL']
        if gcomm.rank == 0:
            print('alpha: %f, CL: %f'%(alpha0.real,CL0))
        if abs(CL0-CL_star)/CL_star<1e-5:
            if gcomm.rank==0:
                print ("Completed! alpha = %f"%alpha0.real)
            break
        # compute sens
        eps = 1e-2
        alphaVal = alpha0 + eps
        alpha(alphaVal)
        funcsP={}
        CFDSolver()
        CFDSolver.evalFunctions(funcsP,evalFuncs=evalFuncs)
        CLP = funcsP['CL']
        deltaAlpha =  (CL_star-CL0)*eps/(CLP-CL0)
        alpha0 += deltaAlpha

else:
    print("task arg not found!")
    exit(0)


