#!/usr/bin/env python
import sys,os
from mpi4py import MPI
from collections import OrderedDict
import numpy as np
from dafoam import *
from idwarp import *
from regression_helper import *

# ###################################################################
# Test: calcSensMap
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print('+' + '-'*78 + '+')
        print('| Test Name: ' + '%-66s'%testName + '|')
        print('+' + '-'*78 + '+')

printHeader('calcSensMap')
sys.stdout.flush()

# cases
allCases = ['CurvedCubeSnappyHexMesh']
                    
defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               False,

            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['wallsbump'],
            'rasmodel':                 'SpalartAllmarasFv3',
            'divdev2':                  True,
            'flowcondition':            'Compressible',
            'adjointsolver':            'rhoSimpleDAFoam',
            'objfuncs':                 ['CD','CL'],
            'objfuncgeoinfo':           [['wallsbump'],
                                         ['wallsbump']],
            'liftdir':                  [1.0,0.0,0.0],
            'dragdir':                  [0.0,0.0,1.0],
            'cofr':                     [0.0,0.0,0.0],
            'setflowbcs':               True,
            'inletpatches':             ['inlet'],
            'outletpatches':            ['outlet'],
            'flowbcs':                  {'bc0':{'patch':'inlet','variable':'U','value':[50.0,0.0,0.0]},
                                         'bc1':{'patch':'outlet','variable':'p','value':[101325.0]},
                                         'bc2':{'patch':'inlet','variable':'k','value':[0.06]},
                                         'bc3':{'patch':'inlet','variable':'omega','value':[400.0]},
                                         'bc4':{'patch':'inlet','variable':'epsilon','value':[2.16]},
                                         'bc5':{'patch':'inlet','variable':'nuTilda','value':[1.5e-4]},
                                         'bc6':{'patch':'inlet','variable':'T','value':[300.0]},
                                         'useWallFunction':'true'},                                  
            'transproperties':          {'nu':1.5E-5,
                                         'TRef':300.0,
                                         'beta':3e-3,
                                         'Pr':0.7,
                                         'Prt':0.85},
            'thermoproperties':         {'molWeight':28.97,
                                         'Cp':1005.0,
                                         'Hf':0.0,
                                         'mu':1.8e-5,
                                         'Pr':0.7,
                                         'Prt':1.0,
                                         'TRef':300.0},
            'divschemes':               {'div(phi,U)':'bounded Gauss linearUpwindV grad(U)',
                                         'div(phi,e)':'bounded Gauss upwind',
                                         'div(phi,nuTilda)':'bounded Gauss upwind',
                                         'div(phi,k)':'bounded Gauss upwind',
                                         'div(phi,omega)':'bounded Gauss upwind',
                                         'div(phi,epsilon)':'bounded Gauss upwind',
                                         'default':'none',
                                         'div(((rho*nuEff)*dev2(T(grad(U)))))': 'Gauss linear',
                                         'div(phi,Ekp)': 'bounded Gauss upwind',
                                         'div(phid,p)':'Gauss limitedLinear 1.0',
                                         'div(pc)':'bounded Gauss upwind'},
            'simplecontrol':            {'nNonOrthogonalCorrectors':'0',
                                         'rhoLowerBound':'0.2',
                                         'rhoUpperBound':'10.0',
                                         'pLowerBound':'20000',
                                         'pUpperBound':'1000000',
                                         'ULowerBound':'-800',
                                         'UUpperBound':'800',
                                         'eLowerBound':'50000',
                                         'eUpperBound':'500000',
                                         'transonic':'false'},
            'fvrelaxfactors':           {'fields':{'p':0.3,'rho':0.3},
                                         'equations':{'U':0.7,'nuTilda':0.7,'e':0.7,'k':0.7,'omega':0.7,'epsilon':0.7}},
            'nffdpoints':               1,
            'maxflowiters':             150, 
            'writeinterval':            150, 
            'stateresettol':            1e-1,
            'adjgmresmaxiters':         1000,
            'adjgmresrestart':          1000,
            'adjgmresreltol':           1e-8,
            'adjjacmatordering':        'cell',
            'adjjacmatreordering':      'rcm',
            'epsderiv':                 1.0e-6, 
            'epsderivxv':               1.0e-6, 
            'adjpcfilllevel':           1, 
            'normalizeresiduals':       ['URes','pRes','p_rghRes','phiRes','TRes','nuTildaRes','kRes','omegaRes', 'epsilonRes'],
            'normalizestates':          ['U','p','p_rgh','phi','T','nuTilda','k','omega','epsilon'],
            'statescaling':             {'UScaling':50.0,
                                         'pScaling':101325.0,
                                         'p_rghScaling':101325.0,
                                         'phiScaling':1.0,
                                         'nuTildaScaling':1e-3,
                                         'kScaling':0.06,
                                         'omegaScaling':400.0,
                                         'epsilonScaling':2.16,
                                         'TScaling':300.0},
            'referencevalues':          {'magURef':50.0,'ARef':0.1,'LRef':1.0,'pRef':101325.0,'rhoRef':1.0},
            'mpispawnrun':              True,
            'adjdvtypes':               ['Xv'],
           }

def resetCase():

    if MPI.COMM_WORLD.rank == 0:
        os.system('rm -fr 0/* [1-9]* *.txt *.dat *.clr *Log* *000 *Coloring* objFunc* psi* *.sh *.info processor* postProcessing')
        if os.path.exists('constant/polyMesh/points_orig.gz'):
            shutil.copyfile('constant/polyMesh/points_orig.gz','constant/polyMesh/points.gz')
            os.remove('constant/polyMesh/points_orig.gz')
    MPI.COMM_WORLD.Barrier()

if __name__ == '__main__':

    #change directory to the correct test case
    os.chdir('input/CurvedCubeSnappyHexMesh/')

    resetCase()

    if MPI.COMM_WORLD.rank == 0:
        os.system('cp 0.compressible/* 0/')
    MPI.COMM_WORLD.Barrier()
    
    meshOptions = {'gridFile':'./',
                'fileType':'openfoam',
                'symmetryPlanes':[[[0.,0., 0.],[0., 0., 0.]]]}

    # setup CFDSolver
    CFDSolver = PYDAFOAM(options=defOpts)
    mesh = USMesh(options=meshOptions)
    CFDSolver.addFamilyGroup(CFDSolver.getOption('designsurfacefamily'),CFDSolver.getOption('designsurfaces'))
    CFDSolver.setMesh(mesh)

    # compute the adjoint coloring
    CFDSolver.computeAdjointColoring()
       
    evalFuncs=CFDSolver.getOption('objfuncs')
    # run flow
    CFDSolver()
    funcs = {}
    CFDSolver.evalFunctions(funcs, evalFuncs=evalFuncs)
    if MPI.COMM_WORLD.rank == 0:
        print('Eval Functions:')
        reg_write_dict(funcs, 1e-10, 1e-10)

    # solve adjoint
    funcsSens = {}
    CFDSolver.solveADjoint()
    CFDSolver.evalFunctionsSens(funcsSens,evalFuncs = evalFuncs)
    funcsSensAvg={}
    for key in list(funcsSens.keys()):
        if 'fail' in key:
            continue
        funcsSensAvg[key]={}
        for key1 in list(funcsSens[key].keys()):
            funcsSensAvg[key][key1]=0.0
            count=0
            for val in funcsSens[key][key1]:
                funcsSensAvg[key][key1]+=val
                count+=1
            funcsSensAvg[key][key1]/=count

    if MPI.COMM_WORLD.rank == 0: 
        print('Eval Functions Sens:')
        reg_write_dict(funcsSensAvg, 1e-6, 1e-10)
    MPI.COMM_WORLD.Barrier()
    resetCase()

