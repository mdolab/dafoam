#!/usr/bin/env python
import sys
from mpi4py import MPI
from collections import OrderedDict
from CompressorFluidSetup import runTests
import numpy as np

# ###################################################################
# Test: turboDAFoam
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print '+' + '-'*78 + '+'
        print '| Test Name: ' + '%-66s'%testName + '|'
        print '+' + '-'*78 + '+'

printHeader('turboDAFoam')
sys.stdout.flush()

# cases
allCases = ['CompressorFluid']

# solver configurations
testInfo=OrderedDict()
testInfo['task1']={'solver':'turboDAFoam',
                    'turbModel':'SpalartAllmaras',
                    'flowCondition':'Compressible',
                    'useWallFunction':'true',
                    'testCases':[allCases[0]]}

defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               False,

            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['blade','hub','shroud','per1','per2','inlet','outlet'], 
            'rasmodel':                 'SpalartAllmaras',
            'divdev2':                  True,
            'flowcondition':            'Compressible',
            'adjointsolver':            'turboDAFoam',
            'objfuncs':                 ['TPR','MFR','CMZ','TTR'],
            'objfuncgeoinfo':           [['inlet','outlet'],['outlet'],['blade'],['inlet','outlet']],
            'setflowbcs':              True,
            'flowbcs':                 {'bc0':{'patch':'inlet','variable':'p','value':[93000.0]},
                                        'bc1':{'patch':'outlet','variable':'p','value':[101000.0]},
                                        'useWallFunction':'true'},                        
            'transproperties':         {'nu':1.5E-5,
                                        'TRef':300.0,
                                        'beta':3e-3,
                                        'Pr':0.7,
                                        'Prt':0.85},
            'thermoproperties':        {'molWeight':28.97,
                                        'Cp':1005.0,
                                        'Hf':0.0,
                                        'mu':1.8e-5,
                                        'Pr':0.7,
                                        'Prt':1.0,
                                        'TRef':300.0},
            'divschemes':              {'div(phi,U)':'Gauss upwind',
                                        'div(phi,h)':'Gauss upwind',
                                        'div(phi,nuTilda)':'Gauss upwind',
                                        'default':'none',
                                        'div(((rho*nuEff)*dev2(T(grad(U)))))': 'Gauss linear',
                                        'div(phi,K)': 'Gauss upwind',
                                        'div(phid,p)':'Gauss upwind',
                                        'div((p*(U-URel)))': 'Gauss linear',
                                        'div((-devRhoReff.T()&U))':'Gauss linear',
                                        'div(pc)':'Gauss upwind'},
            'simplecontrol':           {'nNonOrthogonalCorrectors':'0',
                                        'rhoLowerBound':'0.2',
                                        'rhoUpperBound':'10.0',
                                        'pLowerBound':'20000',
                                        'pUpperBound':'1000000',
                                        'ULowerBound':'-800',
                                        'UUpperBound':'800',
                                        'hLowerBound':'50000',
                                        'hUpperBound':'500000',
                                        'transonic':'true'},
            'thermotype':              {'type':'hePsiThermo',
                                        'mixture':'pureMixture',
                                        'thermo':'hConst',
                                        'transport':'const',
                                        'equationOfState':'perfectGas',
                                        'specie':'specie',
                                        'energy':'sensibleEnthalpy'},
            'mrfproperties':           {'active':'true',
                                        'selectionmode':'cellZone',
                                        'cellzone':'region0',
                                        'nonrotatingpatches':['inlet','outlet','per1','per2'],
                                        'axis':[0,0,1],
                                        'origin':[0,0,0],
                                        'omega':-1000},
            'fvrelaxfactors':          {'fields':{'p':0.8,'rho':1.0},
                                        'equations':{'p':0.8,'U':0.2,'nuTilda':0.2,'h':0.2}},
            'nffdpoints':               8,
            'maxflowiters':             3000, 
            'writeinterval':            3000, 
            'stateresettol':            1e-4,
            'adjgmresmaxiters':         1000,
            'adjgmresrestart':          1000,
            'adjgmresreltol':           1e-5,
            'adjjacmatordering':        'cell',
            'adjjacmatreordering':      'natural',
            'epsderiv':                 1.0e-6, 
            'epsderivffd':              1.0e-3, 
            'adjpcfilllevel':           1,
            'adjasmoverlap':            1, 
            'normalizestates':         ['U','p','e','nuTilda','T','phi','k','epsilon'],
            'normalizeresiduals':      ['URes','pRes','eRes','nuTildaRes','TRes','phiRes','kRes','epsilonRes'],
            'maxresconlv4jacpcmat':    {'URes':2,'pRes':3,'eRes':2,'nuTildaRes':2,'TRes':2,'phiRes':2,'kRes':2,'epsilonRes':2},
            'statescaling':            {'UScaling':1,
                                        'pScaling':1000.0,
                                        'TScaling':300.0,
                                        'k':1.0,
                                        'epsilon':1.0,
                                        'nuTildaScaling':1e-3,
                                        'phiScaling':1,},
            'referencevalues':          {'magURef':1.0,'ARef':1.0,'LRef':1.0,'pRef':101325.0,'rhoRef':1.0},
            'mpispawnrun':              True,
            'adjdvtypes':               ['FFD'],
            'preservepatches':          ['per1','per2'],
           }

if __name__ == '__main__':

    runTests(sys.argv[1],defOpts,testInfo)

    # run subsonic
    defOpts['mrfproperties']['omega']    = -500.0
    defOpts['simplecontrol']['transonic']= 'false'
    defOpts['fvrelaxfactors']            = {'fields':{'p':0.3,'rho':0.02},
                                            'equations':{'p':0.3,'U':0.7,'nuTilda':0.7,'h':0.7}}
    defOpts['flowbcs']['bc0']['value'][0]=100000.0
    runTests(sys.argv[1],defOpts,testInfo)

