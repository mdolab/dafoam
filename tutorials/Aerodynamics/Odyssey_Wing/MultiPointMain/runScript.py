#!/usr/bin/env python
"""
DAFoam run script for the multipoint Odyssey UAV wing case at low speed
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
from multipoint import *

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--output", help='Output directory', type=str,default='../optOutput/')
parser.add_argument("--opt", help="optimizer to use", type=str, default='slsqp')
parser.add_argument("--task", help="type of run to do", type=str, default='opt')
parser.add_argument("--nProcs", help="number of processors to use", type=int, default=1)
parser.add_argument('--optVars',type=str,help='Vars for the optimizer',default="['shape']")
args = parser.parse_args()
exec('optVars=%s'%args.optVars)
task = args.task
outputDirectory = args.output
gcomm = MPI.COMM_WORLD


nProcs     = args.nProcs
nFlowCases = 2
CL_star    = [0.6,0.75]
alphaMP    = [1.768493,3.607844]
MPWeights  = [0.3,0.7]
UmagIn     = 24.8
ARef       = 1.2193524

def calcUAndDir(UIn,alpha1):
    dragDir = [ np.cos(alpha1*np.pi/180), np.sin(alpha1*np.pi/180),0] 
    liftDir = [-np.sin(alpha1*np.pi/180), np.cos(alpha1*np.pi/180),0] 
    inletU = [float(UIn*np.cos(alpha1*np.pi/180)),float(UIn*np.sin(alpha1*np.pi/180)),0.0]
    return inletU, dragDir, liftDir

inletu0, dragdir0, liftdir0 = calcUAndDir(UmagIn,alphaMP[0])


# Set the parameters for optimization
aeroOptions = {
    # output options
    'casename':                 'Odyssey_'+task+'_'+optVars[0],
    'outputdirectory':          outputDirectory,
    'writesolution':            True,

    # multipoit
    'multipointopt':            True,

    # design surfaces and cost functions 
    'designsurfacefamily':     'designsurfaces', 
    'designsurfaces':          ['wing'], 
    'objfuncs':                ['CD','CL'],
    'objfuncgeoinfo':          [['wing'],['wing']],
    'referencevalues':         {'magURef':UmagIn,'ARef':ARef,'LRef':1.0,'pRef':0.0,'rhoRef':1.0},
    'liftdir':                 liftdir0,
    'dragdir':                 dragdir0,

    # flow setup
    'adjointsolver':           'simpleDAFoam',
    'flowcondition':           'Incompressible',
    'rasmodel':                'SpalartAllmarasFv3', 
    'maxflowiters':            500, 
    'writeinterval':           500,
    'setflowbcs':              True,  
    'inletpatches':            ['inout'],
    'outletpatches':           ['inout'],
    'flowbcs':                 {'bc0':{'patch':'inout','variable':'U','value':inletu0},
                                'useWallFunction':'true'},    

    # adjoint setup
    'adjgmresmaxiters':        1500,
    'adjgmresrestart':         1500,
    'adjgmresreltol':          1e-6,
    'stateresettol':           1e-3,
    'adjdvtypes':              ['FFD'], 
    'epsderiv':                1.0e-6, 
    'epsderivffd':             1.0e-3, 
    'adjpcfilllevel':          1, 
    'adjjacmatordering':       'state',
    'adjjacmatreordering':     'nd',
    'statescaling':            {'UScaling':UmagIn,
                                'pScaling':UmagIn*UmagIn/2.0,
                                'phiScaling':1.0,
                                'nuTildaScaling':4.5e-4},
    
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


# ======================================================================
#         Create multipoint communication object
# ======================================================================
MP = multiPointSparse(gcomm)
MP.addProcessorSet(optVars[0]+'_mp', nMembers=1, memberSizes=nProcs) # nMember is always 1
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

# =================================================================================================
# DVGeo
# =================================================================================================
# setup FFD
FFDFile = './FFD/OdysseyFFD.xyz'
DVGeo = DVGeometry(FFDFile)

# how many sections for the global DV such as chord and twist
nTwist = 6

# Setup curves for ref_axis
x = [0.5334/4.0, 0.5334/4.0]
y = [0, 0]
z = [0, 2.30]

tmp = pySpline.Curve(x=x, y=y, z=z, k=2)
X = tmp(np.linspace(0, 1, nTwist))
c1 = pySpline.Curve(X=X, k=2)
DVGeo.addRefAxis('wingAxis', c1)

def twist(val, geo):
    # Set all the twist values
    for i in xrange(nTwist):
        geo.rot_z['wingAxis'].coef[i] = val[i]

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
DVGeo.addGeoDVLocal(optVars[0], lower=-0.5, upper=0.5, axis='y', scale=10.0)
# twist
lower = -10*np.ones(nTwist)
upper =  10*np.ones(nTwist)
lower[0] = 0.0 # root twist does not change
upper[0] = 0.0
DVGeo.addGeoDVGlobal('twist', 0*np.zeros(nTwist), twist,lower=lower, upper=upper, scale=0.1)
# AOA
for i in range(nFlowCases):
    DVGeo.addGeoDVGlobal('fc%d_alpha'%i, alphaMP[i],alpha,lower=0, upper=10.0, scale=1.0)

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
DVCon.addLeTeConstraints(0, 'iLow')
DVCon.addLeTeConstraints(0, 'iHigh')

#Create a volume constraint
# Volume constraints
leList = [[0.5334*0.01, 0, 0.005],     [0.5334*0.01, 0, 2.286]]
teList = [[0.5334*0.99, 0.0013, 0.005],[0.5334*0.99, 0.0013, 2.286]]
DVCon.addVolumeConstraint(leList, teList, 20, 20, lower=1.0)

# Thickness constraints
DVCon.addThicknessConstraints2D(leList, teList, 20, 20, lower=0.5)

if gcomm.rank == 0:
    fileName = os.path.join(args.output, 'constraints.dat')
    DVCon.writeTecplot(fileName)

# =================================================================================================
# optFuncs
# =================================================================================================
# ============================
# Setup optFuncs Object
# ============================
def aeroFuncsMP(xDV):
    """
    Update the design surface and run the flow solver to get objective function values.
    """
    
    if gcomm.rank == 0:
        print ('\n')
        print ('+--------------------------------------------------------------------------+')
        print ('|                    Evaluating Objective Function                         |')
        print ('+--------------------------------------------------------------------------+')
        print ('Design Variables: ',xDV)
        
    fail = False
    
    a = time.time()        
        
    # Setup an empty dictionary for the evaluated function values
    funcsMP = {}

    # Set the current design variables in the DV object
    DVGeo.setDesignVars(xDV)
    CFDSolver.setDesignVars(xDV)
    
    for i in xrange(nFlowCases):       
        if gcomm.rank == 0:
            print ('--Solving Flow Configuratioin %d--'%i)
        
        # funcs dict for each flow configuration    
        funcs = {}
    
        # Evaluate the geometric constraints and add them to the funcs dictionary
        DVCon.evalFunctions(funcs)
        
        alphaVal = xDV['fc%d_alpha'%i].real
        alpha(alphaVal)
        if gcomm.rank == 0:
            print('Inlet Info:')
            print('alpha: ',alphaVal)
            print('dragDir: ',CFDSolver.getOption('dragdir'))
            print('liftDir: ',CFDSolver.getOption('liftdir'))
            print('flowBCs: ',CFDSolver.getOption('flowbcs'))
             
        # Solve the CFD problem
        CFDSolver.setMultiPointFCIndex(i)
        CFDSolver()

        # Populate the required values from the CFD problem
        CFDSolver.evalFunctions(funcs,evalFuncs=evalFuncs)

        # Print the current solution to the screen
        if gcomm.rank == 0:
            print ('Objective Functions: ',funcs)
    
        if funcs['fail'] == True:
            fail = True
        
        # append "fc" (flow case) to the evalFuncs keys
        for key in funcs.keys():
            if 'fail' in key:
                pass
            elif 'DVCon' in key:
                funcsMP[key] = funcs[key]
            else:
                funcsMP['fc%d_'%i+key] = funcs[key]

    funcsMP['fail'] = fail
    
    # Print the current solution to the screen
    if gcomm.rank == 0:
        print ('Objective Functions MultiPoint: ',funcsMP)
    
    b = time.time()
    if gcomm.rank == 0:
        print('Flow runtime: ',b-a)
    
    # flush the output to the screen/file
    sys.stdout.flush()   

    return funcsMP
    
def aeroFuncsSensMP(xDV,funcs):
    """
    Run the adjoint solver and get objective function sensitivities.
    """

    if gcomm.rank == 0:
        print ('\n')
        print ('+--------------------------------------------------------------------------+')
        print ('|                Evaluating Objective Function Sensitivity                 |')
        print ('+--------------------------------------------------------------------------+')

    fail = False

    a = time.time()
   
    # Setup an empty dictionary for the evaluated derivative values
    funcsSensMP={}
    
    for i in xrange(nFlowCases):
    
        if gcomm.rank == 0:
            print ('--Solving Adjoint for Flow Configuration %d--'%i)
        
        funcsSens={}

        # Evaluate the geometric and VG constraint derivatives
        DVCon.evalFunctionsSens(funcsSens)
        
        # solve the adjoint
        CFDSolver.setMultiPointFCIndex(i)
        CFDSolver.solveADjoint()
        
        # Evaluate the CFD derivatives
        CFDSolver.evalFunctionsSens(funcsSens,evalFuncs=evalFuncs)

        # need to add alpha funcsSens, here we use FD
        if gcomm.rank == 0:
            print("Evaluating alpha sens...")
        epsAlpha = 1.0e-2
        alphaVal = xDV['fc%d_alpha'%i]
        alphaVal += epsAlpha
        alpha(alphaVal)
        # Solve the CFD problem
        CFDSolver()
        # reset perturbation
        alphaVal -= epsAlpha
        alpha(alphaVal)
        f1 = open('objFuncs.dat','r')
        lines = f1.readlines()
        f1.close()
        for line in lines:
            cols = line.split()
            dFuncs = float(cols[1])
            dFuncsdAlpha = ( dFuncs - funcs['fc%d_%s'%(i,cols[0])] ) / epsAlpha
            funcsSens[cols[0]]['fc%d_alpha'%i] = dFuncsdAlpha
        
        if funcsSens['fail'] == True:
            fail = True
            
        # append "fc" (flow case) to the evalFuncsSens keys    
        for key in funcsSens.keys():
            if 'fail' in key:
                pass
            elif 'DVCon' in key:
                funcsSensMP[key] = funcsSens[key]
            else:
                funcsSensMP['fc%d_'%i+key] = funcsSens[key] 
    
    funcsSensMP['fail'] = fail
    
    # Print the current solution to the screen
    if gcomm.rank == 0:
        print('Objective Function Sensitivity MultiPoint: ',funcsSensMP)

    b = time.time()
    if gcomm.rank == 0:
        print('Adjoint runtime: ',b-a)
    
    return funcsSensMP

# Assemble the objective and any additional constraints:
def objCon(funcs, printOK):

    funcs['obj'] = 0.0
    for i in xrange(nFlowCases):
        funcs['obj'] += funcs['fc%d_CD'%i]*MPWeights[i]
        funcs['fc%d_CL_con'%i] = funcs['fc%d_CL'%i] - CL_star[i]
    if printOK:
       if gcomm.rank == 0:
           print('MultiPoint Objective Functions:', funcs)
    return funcs

# =================================================================================================
# Task
# =================================================================================================

if task.lower()=='opt':
    optProb = Optimization('opt', MP.obj, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj('obj', scale=1000)
    # Add physical constraints
    for i in xrange(nFlowCases):
        optProb.addCon('fc%d_CL_con'%i, lower=0.0, upper=0.0, scale=1.0)
    
    if gcomm.rank == 0:
        print(optProb)

    # The MP object needs the 'obj' and 'sens' function for each proc set,
    # the optimization problem and what the objcon function is:
    MP.setProcSetObjFunc(optVars[0]+'_mp', aeroFuncsMP)
    MP.setProcSetSensFunc(optVars[0]+'_mp', aeroFuncsSensMP)
    MP.setObjCon(objCon)
    MP.setOptProb(optProb)
    optProb.printSparsity()

    # Make Instance of Optimizer
    opt = OPT(args.opt, options=optOptions)

    # Run Optimization
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, MP.sens, storeHistory=histFile)
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
                print ("Completed! alpha = %f"%alpha0.real)
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


