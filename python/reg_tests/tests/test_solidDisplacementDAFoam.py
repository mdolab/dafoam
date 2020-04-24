#!/usr/bin/env python
import sys
from mpi4py import MPI
from collections import OrderedDict
from setup_CompressorSolid import runTests
import numpy as np

# ###################################################################
# Test: solidDisplacementDAFoam
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print('+' + '-'*78 + '+')
        print('| Test Name: ' + '%-66s'%testName + '|')
        print('+' + '-'*78 + '+')

printHeader('solidDisplacementDAFoam')
sys.stdout.flush()

# cases
allCases = ['CompressorSolid']

# solver configurations
testInfo=OrderedDict()
testInfo['task1']={'solver':'solidDisplacementDAFoam',
                   'testCases':[allCases[0]]}

defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               False,
            
            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['bladeps','bladess','top','bot'],
            'rasmodel':                 'dummyTurbulenceModel',
            'flowcondition':            'Incompressible',
            'adjointsolver':            'solidDisplacementDAFoam',
            'objfuncs':                 ['VMS'],
            'objfuncgeoinfo':           [['allCells']],
            'rotrad':                   [0.0,0.0,0.0],
            'setflowbcs':               True,
            'referencevalues':          {'KSCoeff':5.0e-5,'magURef':1.0,'ARef':1.0,'LRef':1.0,'pRef':0.0,'rhoRef':1.0},
            'flowbcs':                  {'bc0':{'patch':'bladeps','variable':'D','value':[0.0,0.0,0.0],'pressure':[1.0e4]}},       
            'derivuininfo':             {'stateName':'D','component':-1,'type':'tractionDisplacement','patchNames':['bladeps']},
            'transproperties':          {'nu':1.5E-5,
                                         'TRef':300.0,
                                         'beta':3e-3,
                                         'Pr':0.7,
                                         'Prt':0.85,
                                         'DT':4e-5},
            'thermalproperties':        {'C':434.0,
                                         'k':60.5,
                                         'alpha':1.1e-5,
                                         'thermalStress':'false'},
            'mechanicalproperties':     {'rho':7854.0,
                                         'nu':0.0,
                                         'E':2e11},
            'gradschemes':              {'default':'leastSquares'},
            'fvsolvers':                {'"(D|T)"':{'solver':'GAMG',
                                                    'tolerance':'1e-20',
                                                    'relTol':'0.9',
                                                    'smoother':'GaussSeidel',
                                                    'maxIter':'50'}},
            #'sngradschemes':            {'default':'none'},
            'updatedefaultdicts':       {'fvsolvers':{'"(D|T)"':{'solver':'GAMG',
                                                                 'tolerance':'1e-20',
                                                                 'relTol':'0.9',
                                                                 'smoother':'GaussSeidel',
                                                                 'maxIter':'50',
                                                                 'nCellsInCoarsestLevel':'20'}}},
            'maxflowiters':             1000, 
            'writeinterval':            1000, 
            'adjgmresmaxiters':         1000,
            'adjgmresrestart':          1000,
            'adjgmresreltol':           1e-10,
            'maxtoljac':                1e200,
            'maxtolpc':                 1e200,
            'adjjacmatordering':        'state',
            'adjjacmatreordering':      'rcm',
            'epsderiv':                 1.0e-5, 
            'epsderivuin':              1.0e-3, 
            'adjpcfilllevel':           0, 
            'normalizestates':         ['D','U','p','phi','k','omega','nuTilda','epsilon','T'],
            'normalizeresiduals':      ['DRes','URes','pRes','phiRes','kRes','omegaRes','nuTildaRes','epsilonRes','TRes'],
            'maxresconlv4jacpcmat':    {'DRes':2,'URes':2,'pRes':2,'phiRes':1,'nuTildaRes':2,'kRes':2,'omegaRes':2,'epsilonRes':2,'TRes':2},
            'statescaling':            {'DScaling':1e-7,
                                        'UScaling':20.0,
                                        'pScaling':200.0,
                                        'phiScaling':1.0,
                                        'nuTildaScaling':1.5e-4,
                                        'kScaling':0.06,
                                        'epsilonScaling':2.16,
                                        'omegaScaling':400.0,
                                        'TScaling':300.0},
            'mpispawnrun':              True,
            'adjdvtypes':               ['UIn'],
            'meshmaxnonortho':          80.0,
           }

if __name__ == '__main__':

    runTests(sys.argv[1],defOpts,testInfo)

