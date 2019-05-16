#!/usr/bin/env python
import sys
from mpi4py import MPI
from collections import OrderedDict
from UBendDuctSetup import runTests
import numpy as np

# ###################################################################
# Test: buoyantBoussinesqSimpleDAFoam
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print '+' + '-'*78 + '+'
        print '| Test Name: ' + '%-66s'%testName + '|'
        print '+' + '-'*78 + '+'

printHeader('buoyantBoussinesqSimpleDAFoam')
sys.stdout.flush()

# cases
allCases = ['UBendDuct']

UmagIn=8.4 # incompressible

# solver configurations
testInfo=OrderedDict()
testInfo['task1']={'solver':'buoyantBoussinesqSimpleDAFoam',
                    'turbModel':'SpalartAllmarasFv3',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'false',
                    'testCases':[allCases[0]]}
                    
defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               False,

            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['ubend'],
            'rasmodel':                 'SpalartAllmarasFv3',
            'divdev2':                  True,
            'flowcondition':            'Incompressible',
            'adjointsolver':            'buoyantBoussinesqSimpleDAFoam',
            'objfuncs':                 ['CPL','NUS'],
            'objfuncgeoinfo':           [['inlet','outlet'],
                                         ['ubend']],
            'setflowbcs':               True,
            'inletpatches':             ['inlet'],
            'outletpatches':            ['outlet'],
            'flowbcs':                  {'bc0':{'patch':'inlet','variable':'U','value':[UmagIn,0,0]},
                                         'bc1':{'patch':'outlet','variable':'p_rgh','value':[0.0]},
                                         'bc2':{'patch':'inlet','variable':'k','value':[0.06]},
                                         'bc3':{'patch':'inlet','variable':'omega','value':[400.0]},
                                         'bc4':{'patch':'inlet','variable':'epsilon','value':[2.16]},
                                         'bc5':{'patch':'inlet','variable':'nuTilda','value':[1.5e-4]},
                                         'bc6':{'patch':'inlet','variable':'T','value':[300.0]},
                                         'useWallFunction':'false'},                                  
            'transproperties':          {'nu':1.5E-5,
                                         'TRef':300.0,
                                         'beta':3e-3,
                                         'Pr':0.7,
                                         'Prt':0.85,'g':[-9.81,0,0],'rhoRef':1.0,'CpRef':1005.0},
            'updatedefaultdicts':       {'radiationproperties':{'radiation':'on','radiationModel':'P1'}},
            'nffdpoints':               3,
            'maxflowiters':             1000, 
            'writeinterval':            1000, 
            'stateresettol':            1e-5,
            'adjgmresmaxiters':         1000,
            'adjgmresrestart':          1000,
            'adjgmresreltol':           1e-8,
            'adjjacmatordering':        'state',
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

