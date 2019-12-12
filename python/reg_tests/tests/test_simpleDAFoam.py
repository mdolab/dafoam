#!/usr/bin/env python
import sys
from mpi4py import MPI
from collections import OrderedDict
from setup_Airfoil import runTests
from setup_CurvedCube import runTests as runTests1
from setup_CompressorFluid import runTests as runTests2
import numpy as np

# ###################################################################
# Test: simpleDAFoam
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print '+' + '-'*78 + '+'
        print '| Test Name: ' + '%-66s'%testName + '|'
        print '+' + '-'*78 + '+'

printHeader('simpleDAFoam')
sys.stdout.flush()

# cases
allCases = ['Airfoil','CurvedCubeSnappyHexMesh','CompressorFluid']

# Airfoil case
def calcUAndDir(UIn,alpha1):
    dragDir = [ np.cos(alpha1*np.pi/180),np.sin(alpha1*np.pi/180),0]
    liftDir = [-np.sin(alpha1*np.pi/180),np.cos(alpha1*np.pi/180),0]
    inletU = [float(UIn*np.cos(alpha1*np.pi/180)),float(UIn*np.sin(alpha1*np.pi/180)),0]
    return inletU, dragDir, liftDir

UmagIn=30.0 # incompressible
alpha0=2.5
inletu0, dragdir0, liftdir0 = calcUAndDir(UmagIn,alpha0)

# solver configurations
# for the airfoil case
testInfo=OrderedDict()
testInfo['task1']={'solver':'simpleDAFoam',
                    'turbModel':'kOmegaSST',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'false',
                    'testCases':[allCases[0]]}
testInfo['task2']={'solver':'simpleDAFoam',
                    'turbModel':'kEpsilon',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'true',
                    'testCases':[allCases[0]]}
testInfo['task3']={'solver':'simpleDAFoam',
                    'turbModel':'realizableKE',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'false',
                    'testCases':[allCases[0]]}

# for the curvedCube case
testInfo1=OrderedDict()
testInfo1['task1']={'solver':'simpleDAFoam',
                    'turbModel':'SpalartAllmaras',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'false',
                    'testCases':[allCases[1]]}

# for the compressor case
testInfo2=OrderedDict()
testInfo2['task1']={'solver':'simpleDAFoam',
                    'turbModel':'SpalartAllmaras',
                    'flowCondition':'Incompressible',
                    'useWallFunction':'true',
                    'energy':'sensibleInternalEnergy',
                    'testCases':[allCases[2]]}
                    
defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               False,

            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['wing'],
            'rasmodel':                 'SpalartAllmaras',
            'divdev2':                  True,
            'flowcondition':            'Incompressible',
            'adjointsolver':            'simpleDAFoam',
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
                                         'bc1':{'patch':'inout','variable':'p','value':[0.0]},
                                         'bc2':{'patch':'inout','variable':'k','value':[0.06]},
                                         'bc3':{'patch':'inout','variable':'omega','value':[400.0]},
                                         'bc4':{'patch':'inout','variable':'epsilon','value':[2.16]},
                                         'bc5':{'patch':'inout','variable':'nuTilda','value':[1.5e-4]},
                                         'bc6':{'patch':'inout','variable':'T','value':[300.0]},
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
            'adjjacmatreordering':      'nd',
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

    # run the airfoil case
    runTests(sys.argv[1],defOpts,testInfo)

    # now reset some parameters and run the 3D curvedCube case
    alpha0=0.0
    inletu0, dragdir0, liftdir0         = calcUAndDir(UmagIn,alpha0)
    defOpts['liftdir']                  = liftdir0
    defOpts['dragdir']                  = dragdir0
    defOpts['flowbcs']['bc0']['value']  = inletu0
    defOpts['nffdpoints']               = 1
    defOpts['maxflowiters']             = 100 
    defOpts['writeinterval']            = 100 
    defOpts['objfuncs']                 = ['CD','CL','CMX','CMY','CMZ','AVGV','VARV','AVGS']
    defOpts['designsurfaces']           = ['wallsbump']
    defOpts['objfuncgeoinfo']           = [['wallsbump'],
                                           ['wallsbump'],
                                           ['wallsbump'],
                                           ['wallsbump'],
                                           ['wallsbump'],
                                           ['userDefinedVolume0'],
                                           ['userDefinedVolume0'],
                                           ['userDefinedPatch0']]
    defOpts['userdefinedpatchinfo']     = {'userDefinedPatch0':{'type':'patch',
                                           'patchName':'inlet',
                                           'stateName':'p',
                                           'component':0,
                                           'scale':1.0}}
    defOpts['userdefinedvolumeinfo']    = {'userDefinedVolume0':{'type':'annulus',
                                           'stateName':'U',
                                           'component':0,
                                           'scale':1.0,
                                           'centerX':0.5,
                                           'centerY':0.5,
                                           'centerZ':0.5,
                                           'width':0.5,
                                           'radiusInner':0.1,
                                           'radiusOuter':0.3,
                                           'axis':'x'}}
    defOpts['inletpatches']             = ['inlet']
    defOpts['outletpatches']            = ['outlet']
    defOpts['flowbcs']                  = {'bc0':{'patch':'inlet','variable':'U','value':inletu0},
                                           'bc1':{'patch':'outlet','variable':'p','value':[0.0]},
                                           'bc2':{'patch':'inlet','variable':'k','value':[0.06]},
                                           'bc3':{'patch':'inlet','variable':'omega','value':[400.0]},
                                           'bc4':{'patch':'inlet','variable':'epsilon','value':[2.16]},
                                           'bc5':{'patch':'inlet','variable':'nuTilda','value':[1.5e-4]},
                                           'bc6':{'patch':'inlet','variable':'T','value':[300.0]},
                                           'useWallFunction':'false'}
    defOpts['adjdvtypes']               = ['FFD','UIn']
    runTests1(sys.argv[1],defOpts,testInfo1)

    # reset the parameters and run the compressor case
    defOpts['designsurfaces']           =  ['blade','hub','shroud','inlet','outlet','per1','per2']
    defOpts['objfuncs']                 =  ['CMZ']
    defOpts['objfuncgeoinfo']           =  [['blade']]
    defOpts['maxflowiters']             = 40 
    defOpts['writeinterval']            = 40
    defOpts['setflowbcs']               = False
    defOpts['mrfproperties']            = {'active':'true',
                                           'selectionmode':'cellZone',
                                           'cellzone':'region0',
                                           'nonrotatingpatches':['per1','per2','inlet','outlet'],
                                           'axis':[0,0,1],
                                           'origin':[0,0,0],
                                           'omega':-50.0}
    defOpts['thermotype']               = {'type':'hePsiThermo',
                                        'mixture':'pureMixture',
                                        'thermo':'hConst',
                                        'transport':'const',
                                        'equationOfState':'perfectGas',
                                        'specie':'specie',
                                        'energy':'sensibleInternalEnergy'}
    defOpts['preservepatches']          = ['per1','per2']
    defOpts['nffdpoints']               = 8
    defOpts['adjdvtypes']               = ['FFD']
    runTests2(sys.argv[1],defOpts,testInfo2)
    

