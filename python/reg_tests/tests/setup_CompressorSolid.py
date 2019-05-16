import sys, os, copy, numpy,shutil
from mpi4py import MPI
from regression_helper import *
from pygeo import *
from idwarp import *
from pyspline import *
from dafoam import *
from collections import OrderedDict

def resetCase():

    if MPI.COMM_WORLD.rank == 0:
        os.system('rm -fr 0/* [1-9]* *.txt *.dat *.clr *Log* *000 *Coloring* objFunc* psi* *.sh *.info processor* postProcessing')
        if os.path.exists('constant/polyMesh/points_orig.gz'):
            shutil.copyfile('constant/polyMesh/points_orig.gz','constant/polyMesh/points.gz')
            os.remove('constant/polyMesh/points_orig.gz')
    MPI.COMM_WORLD.Barrier()

def runTests(mode,defOpts,testInfo):

    for task in testInfo:

        aeroOptions = copy.deepcopy(defOpts)
        aeroOptions['adjointsolver']=testInfo[task]['solver']
        
        if mode=='compare':
            aeroOptions['usecoloring'] = False
            aeroOptions['updatemesh'] = True

        for testCase in testInfo[task]['testCases']:   

            #change directory to the correct test case
            os.chdir('input/'+testCase+'/')
  
            resetCase()
            if MPI.COMM_WORLD.rank == 0:
                os.system('cp 0.incompressible/* 0/')
            MPI.COMM_WORLD.Barrier()

            # setup CFDSolver
            CFDSolver = PYDAFOAM(options=aeroOptions)
        
            # compute the adjoint coloring
            CFDSolver.computeAdjointColoring()
               
            evalFuncs=CFDSolver.getOption('objfuncs')

            # run flow
            CFDSolver()
            funcs = {}
            CFDSolver.evalFunctions(funcs, evalFuncs=evalFuncs)
            if MPI.COMM_WORLD.rank == 0:
                print 'Eval Functions:'
                reg_write_dict(funcs, 1e-10, 1e-10)

            if mode=='test':
                
                # solve adjoint
                funcsSens = {}
                CFDSolver.solveADjoint()
                CFDSolver.evalFunctionsSens(funcsSens,evalFuncs = evalFuncs)
                # we need to remove dObjdUInz and dObjdUIny
                for key1 in funcsSens.keys():
                    if key1 in CFDSolver.getOption('objfuncs'):
                        for key2 in funcsSens[key1].keys():
                            if key2 == "UIn":
                                funcsSens[key1][key2]=numpy.delete( funcsSens[key1][key2],2 )
                                funcsSens[key1][key2]=numpy.delete( funcsSens[key1][key2],1 )
                if MPI.COMM_WORLD.rank == 0: 
                    print 'Eval Functions Sens:'
                    reg_write_dict(funcsSens, 1e-6, 1e-10)
                MPI.COMM_WORLD.Barrier()
    
                resetCase()

            elif mode=='compare':

                epsUIn = 1e-3

                # initialize funcsSensFD
                funcsSensFD = {}
                funcsSensFD['fail'] = 0
                for funcName in evalFuncs:
                    funcsSensFD[funcName] = {}
                    funcsSensFD[funcName]['UIn'] = 0
                
                # ****** UIn sens ******  
                flowBCs=CFDSolver.getOption('flowbcs')
                CFDSolver.setOption('setflowbcs',True)
                UxRef = flowBCs['bc0']['pressure'][0]
                    
                # perturb +epsUIn
                flowBCs['bc0']['pressure'][0] = UxRef+epsUIn
                CFDSolver.setOption('flowbcs',flowBCs)
                funcp = {}
                CFDSolver()
                CFDSolver.evalFunctions(funcp,evalFuncs=evalFuncs)
                if MPI.COMM_WORLD.rank == 0:
                    print flowBCs
                    print(funcp)

                # perturb -epsUin
                flowBCs['bc0']['pressure'][0] = UxRef-epsUIn
                CFDSolver.setOption('flowbcs',flowBCs)
                funcm = {}
                CFDSolver()
                CFDSolver.evalFunctions(funcm,evalFuncs=evalFuncs)
                if MPI.COMM_WORLD.rank == 0:
                    print flowBCs
                    print(funcm)

                # reset perturbation    
                flowBCs['bc0']['pressure'][0] = UxRef
                CFDSolver.setOption('flowbcs',flowBCs)
                
                for funcName in evalFuncs:
                    grad = ( funcp[funcName]-funcm[funcName] ) / 2.0/ epsUIn
                    funcsSensFD[funcName]['UIn'] = grad
                    if MPI.COMM_WORLD.rank == 0:
                        print(funcName,'UIn',grad)
                    sys.stdout.flush()

                if MPI.COMM_WORLD.rank == 0: 
                    print 'Eval Functions Sens:'
                    reg_write_dict(funcsSensFD, 1e-6, 1e-10)
                MPI.COMM_WORLD.Barrier()

                resetCase()

            else: 
                print('Mode not valid')
                exit()

            del CFDSolver

            #change directory back to the root directory
            os.chdir('../../')