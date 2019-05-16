#!/usr/bin/env python
import sys
from mpi4py import MPI
from collections import OrderedDict
from AirfoilSetup import runTests
import numpy as np

# ###################################################################
# Test: rhoSimpleCDAFoam
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print '+' + '-'*78 + '+'
        print '| Test Name: ' + '%-66s'%testName + '|'
        print '+' + '-'*78 + '+'

printHeader('rhoSimpleCDAFoam')
sys.stdout.flush()

# cases
allCases = ['Airfoil']

def calcUAndDir(UIn,alpha1):
    dragDir = [ np.cos(alpha1*np.pi/180),np.sin(alpha1*np.pi/180),0]
    liftDir = [-np.sin(alpha1*np.pi/180),np.cos(alpha1*np.pi/180),0]
    inletU = [float(UIn*np.cos(alpha1*np.pi/180)),float(UIn*np.sin(alpha1*np.pi/180)),0]
    return inletU, dragDir, liftDir

UmagIn=235.0 # transonic
alpha0=2.5
inletu0, dragdir0, liftdir0 = calcUAndDir(UmagIn,alpha0)

# solver configurations
testInfo=OrderedDict()
testInfo['task1']={'solver':'rhoSimpleCDAFoam',
                    'turbModel':'kOmegaSST',
                    'flowCondition':'Compressible',
                    'useWallFunction':'true',
                    'testCases':[allCases[0]]}
testInfo['task2']={'solver':'rhoSimpleCDAFoam',
                    'turbModel':'kOmega',
                    'flowCondition':'Compressible',
                    'useWallFunction':'true',
                    'testCases':[allCases[0]]}

defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               False,

            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['wing'],
            'rasmodel':                 'kEpsilon',
            'divdev2':                  True,
            'flowcondition':            'Compressible',
            'adjointsolver':            'rhoSimpleCDAFoam',
            'objfuncs':                 ['CD','CL','CMX','CMY','CMZ'],
            'objfuncgeoinfo':           [['wing'],
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
                                         'bc1':{'patch':'inout','variable':'p','value':[101325.0]},
                                         'bc2':{'patch':'inout','variable':'k','value':[0.06]},
                                         'bc3':{'patch':'inout','variable':'omega','value':[400.0]},
                                         'bc4':{'patch':'inout','variable':'epsilon','value':[2.16]},
                                         'bc5':{'patch':'inout','variable':'nuTilda','value':[1.5e-4]},
                                         'bc6':{'patch':'inout','variable':'T','value':[300.0]},
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
            'divschemes':               {'div(phi,U)':'Gauss linearUpwindV grad(U)',
                                         'div(phi,e)':'Gauss upwind',
                                         'div(phi,nuTilda)':'Gauss upwind',
                                         'div(phi,k)':'Gauss upwind',
                                         'div(phi,omega)':'Gauss upwind',
                                         'div(phi,epsilon)':'Gauss upwind',
                                         'default':'none',
                                         'div(((rho*nuEff)*dev2(T(grad(U)))))': 'Gauss linear',
                                         'div(phi,Ekp)': 'Gauss upwind',
                                         'div(phid,p)':'Gauss limitedLinear 1.0',
                                         'div(pc)':'Gauss upwind'},
            'simplecontrol':            {'nNonOrthogonalCorrectors':'0',
                                         'rhoLowerBound':'0.2',
                                         'rhoUpperBound':'10.0',
                                         'pLowerBound':'20000',
                                         'pUpperBound':'1000000',
                                         'ULowerBound':'-800',
                                         'UUpperBound':'800',
                                         'eLowerBound':'50000',
                                         'eUpperBound':'500000',
                                         'transonic':'true'},
            'fvrelaxfactors':           {'fields':{'p':1.0},
                                         'equations':{'p':1.0,'U':0.8,'nuTilda':0.7,'e':0.7,'k':0.7,'omega':0.7,'epsilon':0.7}},
            'nffdpoints':               6,
            'maxflowiters':             500, 
            'writeinterval':            500, 
            'stateresettol':            1e-5,
            'adjgmresmaxiters':         1000,
            'adjgmresrestart':          1000,
            'adjgmresreltol':           1e-8,
            'adjjacmatordering':        'cell',
            'adjjacmatreordering':      'natural',
            'epsderiv':                 1.0e-6, 
            'epsderivffd':              1.0e-4, 
            'adjpcfilllevel':           1, 
            'normalizeresiduals':       ['URes','pRes','p_rghRes','phiRes','TRes','nuTildaRes','kRes','omegaRes', 'epsilonRes'],
            'normalizestates':          ['U','p','p_rgh','phi','T','nuTilda','k','omega','epsilon'],
            'statescaling':             {'UScaling':UmagIn,
                                         'pScaling':101325.0,
                                         'p_rghScaling':101325.0,
                                         'phiScaling':1.0,
                                         'nuTildaScaling':1e-3,
                                         'kScaling':0.06,
                                         'omegaScaling':400.0,
                                         'epsilonScaling':1.0,
                                         'TScaling':300.0},
            'referencevalues':          {'magURef':UmagIn,'ARef':0.1,'LRef':1.0,'pRef':101325.0,'rhoRef':1.0},
            'mpispawnrun':              True,
            'adjdvtypes':               ['FFD'],
           }

if __name__ == '__main__':

    runTests(sys.argv[1],defOpts,testInfo)

