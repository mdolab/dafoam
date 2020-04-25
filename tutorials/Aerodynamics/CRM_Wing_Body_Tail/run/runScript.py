#!/usr/bin/env python
"""
DAFoam run script for the CRM wing-body-tail (DPW4) case (trimmed)
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

LScale     = 1.0 # scale such that the LRef=1

TRef       = 300.0
pRef       = 101325.0
rhoRef     = pRef/TRef/287.0
UmagIn     = 295.0
LRef       = 275.80*0.0254 * LScale
ARef       = 594720.0*0.0254*0.0254/2.0*LScale*LScale
CofR       = [1325.90*0.0254*LScale, 468.75*0.0254*LScale, 177.95*0.0254*LScale]

CL_star    = 0.5
CMY_star   = 0.0
alpha0     = 2.631143

def calcUAndDir(UIn,alpha1):
    dragDir = [ np.cos(alpha1*np.pi/180),0,np.sin(alpha1*np.pi/180)] 
    liftDir = [-np.sin(alpha1*np.pi/180),0,np.cos(alpha1*np.pi/180)] 
    inletU = [float(UIn*np.cos(alpha1*np.pi/180)),0,float(UIn*np.sin(alpha1*np.pi/180))]
    return inletU, dragDir, liftDir

inletu0, dragdir0, liftdir0 = calcUAndDir(UmagIn,alpha0)

# Set the parameters for optimization
aeroOptions = {
    # output options
    'casename':                 'DPW4_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,

    # design surfaces and cost functions 
    'designsurfacefamily':     'designsurfaces', 
    'designsurfaces':          ['wing','tail','body'], 
    'objfuncs':                ['CD','CL','CMY'],
    'objfuncgeoinfo':          [['wing','tail','body'], 
                                ['wing','tail','body'], 
                                ['wing','tail','body']],
    'referencevalues':         {'magURef':UmagIn,'ARef':ARef,'LRef':LRef,'pRef':pRef,'rhoRef':rhoRef,'TRef':TRef},
    'liftdir':                 liftdir0,
    'dragdir':                 dragdir0,
    'cofr':                    CofR,

    # flow setup
    'adjointsolver':           'rhoSimpleCDAFoam',
    'flowcondition':           'Compressible',
    'rasmodel':                'SpalartAllmarasFv3', 
    'maxflowiters':            1000,
    'writeinterval':           1000,
    'divschemes':              {'default':'none',
                                'div(phi,U)': 'Gauss linearUpwindV grad(U)',
                                'div(phi,nuTilda)': 'Gauss upwind',
                                'div(phi,e)': 'Gauss upwind',
                                'div(phid,p)': 'Gauss limitedLinear 1.0',
                                'div(phi,Ekp)': 'Gauss upwind',
                                'div(((rho*nuEff)*dev2(T(grad(U)))))': 'Gauss linear',
                                'div(pc)':'Gauss upwind'},
    'setflowbcs':              True,  
    'inletpatches':            ['inout'],
    'outletpatches':           ['inout'],
    'flowbcs':                 {'bc0':{'patch':'inout','variable':'U','value':inletu0},
                                'useWallFunction':'true'},    
    'fvrelaxfactors':          {'fields':{'p':1.0},
                                 'equations':{'U':0.8,
                                              'p':1.0,
                                              'nuTilda':0.7,
                                              'T':0.7,
                                              'e':0.7}},
   
    # adjoint setup
    'adjgmresmaxiters':        2000,
    'adjgmresrestart':         2000,
    'adjgmresreltol':          1e-5,
    'adjdvtypes':              ['FFD'], 
    'epsderiv':                1.0e-6, 
    'epsderivffd':             1.0e-3, 
    'adjpcfilllevel':          2, 
    'adjjacmatordering':       'cell',
    'adjjacmatreordering':     'natural',
    'statescaling':            {'UScaling':UmagIn,
                                'pScaling':pRef,
                                'eScaling':TRef*1004.0,
                                'phiScaling':1.0,
                                'nuTildaScaling':1.0e-3,
                                'TScaling':TRef},
    
    'simplecontrol':{'nNonOrthogonalCorrectors':'0',
                     'rhoLowerBound':'0.2',
                     'rhoUpperBound':'10.0',
                     'pLowerBound':'20000',
                     'pUpperBound':'1000000',
                     'ULowerBound':'-600',
                     'UUpperBound':'600',
                     'eLowerBound':'100000',
                     'eUpperBound':'500000',
                     'transonic':'true'},

    ########## misc setup ##########
    'mpispawnrun':             False,
    'restartopt':              False,
    'meshmaxnonortho':         75.0,
    'meshmaxskewness':         6.0,
    'stateresettol':           1e-2,
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    'gridFile':                os.getcwd(),
    'fileType':                'openfoam',
    # point and normal for the symmetry plane
    'symmetryPlanes':          [[[0.,0., 0.],[0., -1., 0.]]], 
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
# setup FFD
FFDFile = './FFD/dpw4FFD.xyz'
DVGeo = DVGeometry(FFDFile)
# We will use the FFD coordinates to create the reference axis
# automatically, this will ensure everything lines up exactly as we
# want.

# First extract the coefficients of the FFD that corresponds to the
# wing. This happens to be vol zero: the 'i' direction is 'x'
# (streamwise), the 'j' direction is out the wing and the 'k'
# direction is 'up'
coef = DVGeo.FFD.vols[0].coef.copy()

# First determine the reference chord lengths:
nTwist = coef.shape[1]
sweep_ref = np.zeros((nTwist+1, 3))
for j in range(nTwist):
    max_x = np.max(coef[:, j, :, 0])
    min_x = np.min(coef[:, j, :, 0])
    sweep_ref[j+1, 0] = min_x + 0.25*(max_x-min_x)
    sweep_ref[j+1, 1] = np.average(coef[:, j, :, 1])
    sweep_ref[j+1, 2] = np.average(coef[:, j, :, 2])

# Now add on the first point which is just the second one, projected
# onto the sym plane
sweep_ref[0, :] = sweep_ref[1, :].copy()
sweep_ref[0, 1] = 0.0

# Create the actual reference axis
c1 = Curve(X=sweep_ref, k=2)
DVGeo.addRefAxis('wing', c1, volumes=[0, 5])

# Now the tail reference axis
x = np.array([2365.0, 2365.0])*.0254
y = np.array([0, 840/2.0])*.0254
z = np.array([255.0, 255.0])*.0254
c2 = Curve(x=x, y=y, z=z, k=2)
DVGeo.addRefAxis('tail', c2, volumes=[25])

def twist(val, geo):
    # Set all the twist values
    for i in range(nTwist):
        geo.rot_y['wing'].coef[i+1] = val[i]

    # Also set the twist of the root to the SOB twist
    geo.rot_y['wing'].coef[0] = val[0]

def tailTwist(val, geo):
    # Set one twist angle for the tail
    geo.rot_y['tail'].coef[:] = val[0]

        
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


# FFD shape
pts=DVGeo.getLocalIndex(0)
indexList=pts[:,:,:].flatten()  # select the top layer FFD starts with i=1
PS=geo_utils.PointSelect('list',indexList)
DVGeo.addGeoDVLocal(optVars[0], lower=-1.0, upper=1.0, axis='z', scale=10.0,pointSelect=PS)
# twist
lower = -10*np.ones(nTwist)
upper =  10*np.ones(nTwist)
lower[0] = 0.0 # root twist does not change
upper[0] = 0.0
DVGeo.addGeoDVGlobal('twist', 0*np.zeros(nTwist), twist,lower=lower, upper=upper, scale=0.1)
DVGeo.addGeoDVGlobal('tail', 0*np.zeros(1), tailTwist,lower=-10, upper=10, scale=0.1)
# AOA
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

# Le/Te constraints
DVCon.addLeTeConstraints(0, 'iHigh')
DVCon.addLeTeConstraints(0, 'iLow')


# (flattened)LE Root, break and tip. These are adifferent from above
leRoot = np.array([25.22*LScale, 3.20*LScale, 0])
leBreak = np.array([31.1358*LScale, 10.8712*LScale, 0.0])
leTip = np.array([45.2307*LScale, 29.38*LScale, 0.0])
rootChord = 11.83165*LScale
breakChord = 7.25*LScale
tipChord = 2.727*LScale

coe1=0.2 # in production run where the mesh is refined, set coe1=0.01
coe2=1.0-coe1
xaxis = np.array([1.0, 0, 0])
leList = [leRoot + coe1*rootChord*xaxis,
          leBreak + coe1*breakChord*xaxis,
          leTip + coe1*tipChord*xaxis]

teList = [leRoot + coe2*rootChord*xaxis,
          leBreak + coe2*breakChord*xaxis,
          leTip + coe2*tipChord*xaxis]

DVCon.addVolumeConstraint(leList, teList, nSpan=25, nChord=30,
                          lower=1.0,upper=3, scaled=True)

# Add the same grid of thickness constraints with minimum bound of 0.25
DVCon.addThicknessConstraints2D(leList, teList, 25, 30,
                                lower=0.2, upper=3.0,
                                scaled=True)
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
        print('Objective Functions: ',funcs)
        print('Flow Runtime: ',b-a)

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
        print('Objective Function Sensitivity: ',funcsSens)
        print('Adjoint Runtime: ',b-a)

    fail = funcsSens['fail']

    # flush the output to the screen/file
    sys.stdout.flush()

    return funcsSens,fail


# =================================================================================================
# Task
# =================================================================================================
optFuncs.CFDSolver = CFDSolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

if task.lower()=='opt':
    optProb = Optimization('opt', aeroFuncs, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj('CD', scale=1)
    # Add physical constraints
    optProb.addCon('CL',lower=CL_star,upper=CL_star,scale=1)
    optProb.addCon('CMY',lower=CMY_star,upper=CMY_star,scale=1)
    
    if gcomm.rank == 0:
        print(optProb)

    opt = OPT(args.opt, options=optOptions)
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, sens=aeroFuncsSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

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
                print("Completed! alpha = %f"%alpha0.real)
            break
        # compute sens
        eps = 1e-3
        alphaVal = alpha0 + eps
        alpha(alphaVal)
        funcsP={}
        CFDSolver()
        CFDSolver.evalFunctions(funcsP,evalFuncs=evalFuncs)
        CLP = funcsP['CL']
        deltaAlpha =  (CL_star-CL0)*eps/(CLP-CL0)
        alpha0 += deltaAlpha

elif task.lower() == 'run':

    optFuncs.run()

elif task.lower() == 'testsensuin':

    optFuncs.testSensUIn(normStatesList=[True],deltaUList=[1e-6])
        
elif task.lower() == 'testsensshape':

    optFuncs.testSensShape(normStatesList=[True],deltaUList=[1e-6],deltaXList=[1e-3])

elif task.lower() == 'xdv2xv':

    optFuncs.xDV2xV()

else:
    print("task arg not found!")
    exit(0)


