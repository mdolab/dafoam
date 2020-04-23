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
parser.add_argument("--output", help='Output directory', type=str,default='../optOutputFluid/')
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
    'referencevalues':         {'magURef':100.0,'ARef':1.0,'LRef':0.1,'pRef':101325.0,'TRef':288.15,'rhoRef':1.17},
    'derivuininfo':            {'stateName':'p','component':0,'type':'fixedValue','patchNames':['outlet']},
    # flow setup
    'adjointsolver':           'rhoSimpleDAFoam',
    'rasmodel':                'SpalartAllmarasFv3',
    'flowcondition':           'Compressible',
    'maxflowiters':            1000, 
    'writeinterval':           1000,
    'setflowbcs':              True,
    'flowbcs':                 {'bc0':{'patch':'inlet','variable':'U','value':[0,0,98.689710]},'bc1':{'patch':'outlet','variable':'p','value':[102000.0]}},  

    # adjoint setup
    'adjgmresmaxiters':        1000,
    'adjgmresrestart':         1000,
    'adjgmresreltol':          1e-5,
    'adjdvtypes':              ['FFD'],
    'epsderiv':                1.0e-5,
    'epsderivffd':             1.0e-4,
    'adjpcfilllevel':          1,
    'adjjacmatordering':       'cell',
    'adjjacmatreordering':     'natural',
    'statescaling':            {'UScaling':100,
                                'pScaling':100000.0,
                                'TScaling':300.0,
                                'eScaling':10000,
                                'nuTildaScaling':1e-3,
                                'phiScaling':1,},
    'fvrelaxfactors':          {'fields':{'p':0.3,'rho':0.02},
                                'equations':{'U':0.7,
                                             'nuTilda':0.7,
                                             'e':0.7,
                                             'G':0.7,}},
    'mrfproperties':           {'active':'true',
                                'selectionmode':'cellZone',
                                'cellzone':'region0',
                                'nonrotatingpatches':['per1','per2','inlet','outlet'],
                                'axis':[0,0,1],
                                'origin':[0,0,0],
                                'omega':-840.0},

    'simplecontrol':           {'nNonOrthogonalCorrectors':'0',
                                'rhoLowerBound':'0.2',
                                'rhoUpperBound':'5.0',
                                'pLowerBound':'20000',
                                'pUpperBound':'500000',
                                'ULowerBound':'-400',
                                'UUpperBound':'400',
                                'nuTildaUpperBound':'1.0',
                                'eLowerBound':'100000',
                                'eUpperBound':'500000',
                                'transonic':'false'},

    ########## misc setup ##########
    'mpispawnrun':             False,
    'restartopt':              False,
    'meshmaxnonortho':         75.0,
    'meshmaxskewness':         6.0,
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
DVGeoChild.addGeoDVLocal('shapex', lower=-0.002, upper=0.002, axis='x', scale=1.0,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapey', lower=-0.002, upper=0.002, axis='y', scale=1.0,pointSelect=PS)
DVGeoChild.addGeoDVLocal('shapez', lower=-0.002, upper=0.002, axis='z', scale=1.0,pointSelect=PS)
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

evalConFuncsCounter = [0]
evalConFuncsSensCounter = [0]

def evalConFuncs(funcs):

    finishFile='jobFinished'

    logFileName = "solidLog"

    outputDir = CFDSolver.getOption('outputdirectory')

    if funcs['fail'] == False:
            
        if gcomm.rank==0:
            # Remove the log file if it exists
            if os.path.isfile(logFileName):
                try:
                    os.remove(logFileName)
                except:
                    raise Error('pyDAFoam: status %d: Unable to remove %s'%logFileName)
            # remove finish file
            if os.path.isfile(finishFile):
                try:
                    os.remove(finishFile)
                except:
                    raise Error('pyDAFoam: status %d: Unable to remove %s'%finishFile)
                    
            # create the run files
            fTouch=open('runSolidSolver','w')
            fTouch.close()
        
        gcomm.Barrier()
    
        if gcomm.rank==0:
            print("Evaluating structural constraints...")
                            
        # check if the job finishes
        checkTime = CFDSolver.getOption('filechecktime')
        while not os.path.isfile(finishFile):
            time.sleep(checkTime)
        gcomm.Barrier()

    if gcomm.rank==0:
        print("Simulation Finished!")
        shutil.copyfile(logFileName,outputDir+"/"+logFileName+"_%03d"%evalConFuncsCounter[0])
        shutil.copyfile('objFuncsSolid.dat',outputDir+"/"+"objFuncsSolid_%03d.dat"%evalConFuncsCounter[0])

    evalConFuncsCounter[0]+=1
    
    # read the objective function
    f = open('objFuncsSolid.dat')
    lines = f.readlines()
    f.close()

    for line in lines:
        cols = line.split()
        funcs[cols[0]] = float(cols[1])
    
    f=open('solidMeshFailed.dat','r')
    lines=f.readlines()
    f.close()
    for line in lines:
        cols=line.split()
        if cols[0] == '1':
            funcs['fail']=True

    return

def writeDVs():
    '''
    Write the current DV values to files
    '''
    if DVGeo is None:
        return
    
    dvs = DVGeo.getValues()
    if gcomm.rank == 0:
        f = open('designVariables.dat','w')
        for key in dvs.keys():
    	    f.write('%s '%key)
    	    for val in dvs[key]:
    	        f.write('%.15e '%val)
    	    f.write('\n')
        f.close()
    gcomm.Barrier()    

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
    
    # Solve the CFD problem
    CFDSolver()

    # Populate the required values from the CFD problem
    CFDSolver.evalFunctions(funcs,evalFuncs=evalFuncs)

    # call the structural solver to get contraint function values
    writeDVs()
    evalConFuncs(funcs)
    
    #scale vms 
    #funcs['VMS'] = funcs['VMS']/vmsScale

    b = time.time()
    
    # Print the current solution to the screen
    if gcomm.rank == 0:
        print ('Objective Functions: ',funcs)
        print ('Flow Runtime: ',b-a)
        
    fail = funcs['fail']
        
    # flush the output to the screen/file
    sys.stdout.flush()

    return funcs,fail

def evalConFuncsSens(funcsSens):

    finishFile='jobFinished'

    logFileName = "solidAdjointLog"

    outputDir = CFDSolver.getOption('outputdirectory')

    if funcsSens['fail'] == False:
            
        if gcomm.rank==0:
            # Remove the log file if it exists
            if os.path.isfile(logFileName):
                try:
                    os.remove(logFileName)
                except:
                    raise Error('pyDAFoam: status %d: Unable to remove %s'%logFileName)
            # remove finish file
            if os.path.isfile(finishFile):
                try:
                    os.remove(finishFile)
                except:
                    raise Error('pyDAFoam: status %d: Unable to remove %s'%finishFile)
                    
            # create the run files
            fTouch=open('runSolidAdjointSolver','w')
            fTouch.close()
        
        gcomm.Barrier()
    
        if gcomm.rank==0:
            print("Evaluating structural constraints derivatives...")
                            
        # check if the job finishes
        checkTime = CFDSolver.getOption('filechecktime')
        while not os.path.isfile(finishFile):
            time.sleep(checkTime)
        gcomm.Barrier()
    
    if gcomm.rank==0:
        print("Simulation Finished!")
        shutil.copyfile(logFileName,outputDir+"/"+logFileName+"_%03d"%evalConFuncsSensCounter[0])
        shutil.copyfile('objFuncsSens_dVMSdFFD.dat',outputDir+"/"+"objFuncsSens_dVMSdFFD_%03d.dat"%evalConFuncsSensCounter[0])

    evalConFuncsSensCounter[0]+=1
    
    # read the objective function
    f = open('objFuncsSens_dVMSdFFD.dat')
    lines = f.readlines()
    f.close()

    nDVs= CFDSolver.getOption('nffdpoints')

    ffdIn=[]
    lineCounter = 0
    for line in lines:
        # don't read the header and last line
        if lineCounter > 2 and lineCounter < nDVs+3:
            cols = line.split()
            ffdIn.append(float(cols[0]))
        lineCounter+=1
    
    # assign funcsSens
    funcsSens["VMS"]={}

    xDVs = DVGeo.getValues()
    # now loop over all the keys in sorted order and read the sens
    ffdCounterI=0
    # NOTE: we need to first sort the keys for all the local and global DVs
    # so that the sequence is consistent with the writeDeltaVolPointMat function
    for key in sorted(xDVs.keys()):
        # get the length of DV for this key
        lenDVs = len(xDVs[key])
        # loop over all the DVs for this key and assign
        funcsSens["VMS"][key] = np.zeros([lenDVs],'d')
        for i in range(lenDVs):
            funcsSens["VMS"][key][i] = ffdIn[ffdCounterI]
            ffdCounterI+=1

    return

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

    # call the structural solver to get contraint function derivatives
    writeDVs()
    evalConFuncsSens(funcsSens)

    # scale VMS sens
    #for key in funcsSens['VMS'].keys():
    #    for idxI, val in enumerate(funcsSens['VMS'][key]):
    #        funcsSens['VMS'][key][idxI] = val/vmsScale
    
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
if task.lower()=='opt':
    optProb = Optimization('opt', aeroFuncs, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj('CMZ', scale=1)
    # Add physical constraints
    optProb.addCon('TPR',lower=1.112,upper=1.112,scale=1)
    optProb.addCon('MFR',lower=1.003,upper=1.003,scale=1)
    optProb.addCon('VMS',lower=0,upper=4.828e7,scale=1e-7)

    if gcomm.rank == 0:
        print(optProb)

    opt = OPT(args.opt, options=optOptions)
    histFile = os.path.join(outputDirectory, '%s_hist.hst'%args.opt)
    sol = opt(optProb, sens=aeroFuncsSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

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



