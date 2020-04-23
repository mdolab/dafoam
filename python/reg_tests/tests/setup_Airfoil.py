import sys, os, copy, numpy,shutil
from mpi4py import MPI
from regression_helper import *
from pygeo import *
from idwarp import *
from pyspline import *
from dafoam import *
from collections import OrderedDict

meshOptions = {'gridFile':'./',
                'fileType':'openfoam',
                'symmetryPlanes':[[[0.,0., 0.],[0., 0., 1.]]]}

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
        turbModel=testInfo[task]['turbModel']
        aeroOptions['rasmodel'] = turbModel
        aeroOptions['adjointsolver']=testInfo[task]['solver']
        aeroOptions['flowCondition']=testInfo[task]['flowCondition']
        aeroOptions['flowbcs']['useWallFunction']=testInfo[task]['useWallFunction']
        
        if mode=='compare':
            aeroOptions['usecoloring'] = False
            aeroOptions['updatemesh'] = True

        for testCase in testInfo[task]['testCases']:   

            #change directory to the correct test case
            os.chdir('input/'+testCase+'/')
  
            resetCase()
            if MPI.COMM_WORLD.rank == 0:
                if testInfo[task]['flowCondition']=='Incompressible':
                    os.system('cp 0.incompressible/* 0/')
                elif testInfo[task]['flowCondition']=='Compressible':
                    os.system('cp 0.compressible/* 0/')
            MPI.COMM_WORLD.Barrier()

            # =====================
            # Setup Geometry Object
            # =====================
            
            FFDFile = './FFD/wingFFD.xyz'
            DVGeo = DVGeometry(FFDFile)
            
            # ref axis
            x = [0,1]
            y = [0.01,0.01 ]
            z = [0.05,0.05]
            c1 = pySpline.Curve(x=x, y=y, z=z, k=2)
            DVGeo.addRefAxis('bodyAxis', curve = c1,axis='y')
            
            # select points
            pts=DVGeo.getLocalIndex(0) 
            indexList=pts[:,:,0].flatten()
            PS=geo_utils.PointSelect('list',indexList)
            DVGeo.addGeoDVLocal('shapey',lower=-0.1, upper=0.1,axis='y',scale=1.0,pointSelect=PS)
            
            # setup CFDSolver
            CFDSolver = PYDAFOAM(options=aeroOptions)
            CFDSolver.setDVGeo(DVGeo)
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

            if mode=='test':
                
                # solve adjoint
                funcsSens = {}
                CFDSolver.solveADjoint()
                CFDSolver.evalFunctionsSens(funcsSens,evalFuncs = evalFuncs)
                if MPI.COMM_WORLD.rank == 0: 
                    print('Eval Functions Sens:')
                    reg_write_dict(funcsSens, 1e-6, 1e-10)
                MPI.COMM_WORLD.Barrier()
    
                resetCase()

            elif mode=='compare':

                nDVs = CFDSolver.getOption('nffdpoints')
                epsFFD = CFDSolver.getOption('epsderivffd')
                epsUIn = 1.0e-3

                # get the DVs 
                xDV = DVGeo.getValues()

                # initialize funcsSensFD
                funcsSensFD = {}
                funcsSensFD['fail'] = 0
                for funcName in evalFuncs:
                    funcsSensFD[funcName] = {}
                    for key in xDV:
                        DVLen = len(xDV[key])
                        funcsSensFD[funcName][key] = numpy.zeros(DVLen)
                
                # ******** FFD sens *********
                # calculate FD ref values        
                for key in xDV:
                    DVLen = len(xDV[key])
                    for i in range(DVLen):
            
                        funcp={}
                        funcm={}  
                        
                        xDV[key][i] =epsFFD
                        DVGeo.setDesignVars(xDV)
                        CFDSolver.setDesignVars(xDV)
                        CFDSolver()
                        CFDSolver.evalFunctions(funcp,evalFuncs=evalFuncs)
                        if MPI.COMM_WORLD.rank == 0:
                            print (xDV)
                            print(funcp)
                        
                        xDV[key][i] =-epsFFD
                        DVGeo.setDesignVars(xDV)
                        CFDSolver.setDesignVars(xDV)
                        CFDSolver()
                        CFDSolver.evalFunctions(funcm,evalFuncs=evalFuncs)
                        if MPI.COMM_WORLD.rank == 0:
                            print (xDV)
                            print(funcp)
                        
                        # reset perturabation
                        xDV[key][i] =0.0
                        DVGeo.setDesignVars(xDV)
                        CFDSolver.setDesignVars(xDV)
            
                        for funcName in evalFuncs:
                            grad = (funcp[funcName]-funcm[funcName])/ 2.0 / epsFFD
                            if MPI.COMM_WORLD.rank == 0:
                                print(funcName,key,i,grad)
                            sys.stdout.flush()
                            funcsSensFD[funcName][key][i] = grad

                if MPI.COMM_WORLD.rank == 0: 
                    print('Eval Functions Sens:')
                    reg_write_dict(funcsSensFD, 1e-6, 1e-10)
                MPI.COMM_WORLD.Barrier()

                resetCase()

            else: 
                print('Mode not valid')
                exit()

            del CFDSolver
            del DVGeo
            del mesh

            #change directory back to the root directory
            os.chdir('../../')