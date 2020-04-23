#!/usr/bin/env python
import sys
from mpi4py import MPI
from collections import OrderedDict
from setup_Airfoil import runTests
import numpy as np

# ###################################################################
# Test: simpleTDAFoam
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print('+' + '-'*78 + '+')
        print('| Test Name: ' + '%-66s'%testName + '|')
        print('+' + '-'*78 + '+')

printHeader('simpleTDAFoam')
sys.stdout.flush()

# cases
allCases = ['Airfoil']

def calcUAndDir(UIn,alpha1):
    dragDir = [ np.cos(alpha1*np.pi/180),np.sin(alpha1*np.pi/180),0]
    liftDir = [-np.sin(alpha1*np.pi/180),np.cos(alpha1*np.pi/180),0]
    inletU = [float(UIn*np.cos(alpha1*np.pi/180)),float(UIn*np.sin(alpha1*np.pi/180)),0]
    return inletU, dragDir, liftDir

UmagIn=30.0 # incompressible
alpha0=2.5
inletu0, dragdir0, liftdir0 = calcUAndDir(UmagIn,alpha0)

# solver configurations
testInfo=OrderedDict()
testInfo['task1']={'solver':'simpleTDAFoam',
                    'turbModel':'SpalartAllmaras',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'false',
                    'testCases':[allCases[0]]}
testInfo['task2']={'solver':'simpleTDAFoam',
                    'turbModel':'kOmega',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'false',
                    'testCases':[allCases[0]]}

defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               False,

            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['wing'],
            'rasmodel':                 'SpalartAllmaras',
            'divdev2':                  True,
            'flowcondition':            'Incompressible',
            'adjointsolver':            'simpleTDAFoam',
            'objfuncs':                 ['CD','CL','CMX','CMY','CMZ','NUS'],
            'objfuncgeoinfo':           [['wing'],
                                         ['wing'],
                                         ['wing'],
                                         ['wing'],
                                         ['wing'],
                                         ['wing']],
            'liftdir':                  liftdir0,
            'dragdir':                  dragdir0,
            'cofr':                     [0.0,0.0,0.0],
            'setflowbcs':               True,
            'inletpatches':             ['inout'],
            'outletpatches':            ['inout'],
            'flowbcs':                  {'bc0':{'patch':'inout','variable':'U','value':inletu0},
                                         'bc1':{'patch':'inout','variable':'p','value':[0.0]},
                                         'bc2':{'patch':'inout','variable':'k','value':[0.06]},
                                         'bc3':{'patch':'inout','variable':'omega','value':[400.0]},
                                         'bc4':{'patch':'inout','variable':'epsilon','value':[2.16]},
                                         'bc5':{'patch':'inout','variable':'nuTilda','value':[1.5e-4]},
                                         'bc6':{'patch':'inout','variable':'T','value':[300.0]},
                                         'bc7':{'patch':'wing','variable':'T','value':[310.0]},
                                         'useWallFunction':'false'},                                  
            'transproperties':          {'nu':1.5E-5,
                                         'TRef':300.0,
                                         'beta':3e-3,
                                         'Pr':0.7,
                                         'Prt':0.85},
            'nffdpoints':               6,
            'maxflowiters':             400, 
            'writeinterval':            400, 
            'stateresettol':            1e-5,
            'adjgmresmaxiters':         1000,
            'adjgmresrestart':          1000,
            'adjgmresreltol':           1e-8,
            'adjjacmatordering':        'cell',
            'adjjacmatreordering':      'rcm',
            'epsderiv':                 1.0e-6, 
            'epsderivffd':              1.0e-4, 
            'adjpcfilllevel':           1, 
            'normalizeresiduals':       ['URes','pRes','p_rghRes','phiRes','TRes','nuTildaRes','kRes','omegaRes', 'epsilonRes'],
            'normalizestates':          ['U','p','p_rgh','phi','T','nuTilda','k','omega','epsilon'],
            'statescaling':             {'UScaling':UmagIn,
                                         'pScaling':UmagIn*UmagIn/2.0,
                                         'p_rghScaling':UmagIn*UmagIn/2.0,
                                         'phiScaling':1.0,
                                         'nuTildaScaling':1e-3,
                                         'kScaling':0.06,
                                         'omegaScaling':400.0,
                                         'epsilonScaling':2.16,
                                         'TScaling':300.0},
            'referencevalues':          {'magURef':UmagIn,'ARef':0.1,'LRef':1.0,'pRef':0.0,'rhoRef':1.0},
            'mpispawnrun':              True,
            'adjdvtypes':               ['FFD'],
            'runpotentialfoam':         True
           }

if __name__ == '__main__':

    runTests(sys.argv[1],defOpts,testInfo)

