#!/usr/bin/env python
"""

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Common functions for DAFoam optimization setup.
    
"""


# =============================================================================
# Imports
# =============================================================================
import time,sys,os
import numpy as np
np.set_printoptions(precision=16)

objFuncOffset={}
objFuncScale={}

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

    # check if we need to offset or scale the object functions
    # note: we first offset then scale!
    if not len(objFuncOffset) == 0:
        if gcomm.rank == 0:
            print ('Offseting objective functions: ',objFuncOffset)
        for key in funcs.keys():
            if key in objFuncOffset.keys():
                origVal = funcs[key]
                funcs[key] = origVal+objFuncOffset[key]
    if not len(objFuncScale) == 0:
        if gcomm.rank == 0:
            print ('scaling objective functions: ',objFuncScale)
        for key in funcs.keys():
            if key in objFuncScale.keys():
                origVal = funcs[key]
                funcs[key] = origVal*objFuncScale[key]

    b = time.time()
    
    # Print the current solution to the screen
    if gcomm.rank == 0:
        print ('Objective Functions: ',funcs)
        print ('Flow Runtime: ',b-a)
        
    fail = funcs['fail']
        
    # flush the output to the screen/file
    sys.stdout.flush()

    return funcs,fail
    
def aeroFuncsMP(xDV):
    
    if gcomm.rank == 0:
        print ('\n')
        print ('+--------------------------------------------------------------------------+')
        print ('|                    Evaluating Objective Function                         |')
        print ('+--------------------------------------------------------------------------+')
        print ('Design Variables: ',xDV)
        
    fail = False
    
    a = time.time()        
        
    # Setup an empty dictionary for the evaluated function values
    funcsMP = {}

    # Set the current design variables in the DV object
    DVGeo.setDesignVars(xDV)
    CFDSolver.setDesignVars(xDV)
    
    for i in xrange(nFlowCases):       
        if gcomm.rank == 0:
            print ('--Solving Flow Configuratioin %d--'%i)
        
        # funcs dict for each flow configuration    
        funcs = {}
    
        # Evaluate the geometric constraints and add them to the funcs dictionary
        DVCon.evalFunctions(funcs)
        
        dir1=np.cos(yawAngles[i]*np.pi/180.0)
        dir2=np.sin(yawAngles[i]*np.pi/180.0)
        CFDSolver.setOption('dragdir',[dir1,dir2,0.0])
         
        # Solve the CFD problem
        CFDSolver.setMultiPointFCIndex(i)
        CFDSolver()

        # Populate the required values from the CFD problem
        CFDSolver.evalFunctions(funcs,evalFuncs=evalFuncs)

        # Print the current solution to the screen
        if gcomm.rank == 0:
            print ('Objective Functions: ',funcs)
    
        if funcs['fail'] == True:
            fail = True
        
        # append "fc" (flow case) to the evalFuncs keys
        for key in funcs.keys():
            if 'fail' in key:
                pass
            elif 'DVCon' in key:
                funcsMP[key] = funcs[key]
            else:
                funcsMP['fc%d_'%i+key] = funcs[key]

    funcsMP['fail'] = fail
    
    # Print the current solution to the screen
    if gcomm.rank == 0:
        print ('Objective Functions MultiPoint: ',funcsMP)
    
    b = time.time()
    if gcomm.rank == 0:
        print 'Flow runtime: ',b-a
    
    # flush the output to the screen/file
    sys.stdout.flush()   

    return funcsMP

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

    # check if we need to scale the object functions sens
    # note: we dont need to offset!
    if not len(objFuncScale) == 0:
        if gcomm.rank == 0:
            print ('scaling objective functions sensitivities: ',objFuncScale)
        for key in funcsSens.keys():
            if key in objFuncScale.keys():
                for key1 in funcsSens[key].keys():
                    nDVs = len(funcsSens[key][key1]) 
                    if nDVs > 1:
                        for i in range(nDVs):
                            origVal = funcsSens[key][key1][i]
                            funcsSens[key][key1][i] = origVal*objFuncScale[key]
                    else:
                        origVal = funcsSens[key][key1]
                        funcsSens[key][key1] = origVal*objFuncScale[key]
    
    b = time.time()
    
    # Print the current solution to the screen
    if gcomm.rank == 0:
        print('Objective Function Sensitivity: ',funcsSens)
        print('Adjoint Runtime: ',b-a)
        
    fail = funcsSens['fail']
    
    # flush the output to the screen/file
    sys.stdout.flush()
    
    return funcsSens,fail
    
def aeroFuncsSensMP(xDV,funcs):

    if gcomm.rank == 0:
        print ('\n')
        print ('+--------------------------------------------------------------------------+')
        print ('|                Evaluating Objective Function Sensitivity                 |')
        print ('+--------------------------------------------------------------------------+')
        
    fail = False

    a = time.time()
   
    # Setup an empty dictionary for the evaluated derivative values
    funcsSensMP={}
    
    for i in xrange(nFlowCases):
    
        if gcomm.rank == 0:
            print ('--Solving Adjoint for Flow Configuration %d--'%i)
        
        funcsSens={}

        # Evaluate the geometric and VG constraint derivatives
        DVCon.evalFunctionsSens(funcsSens)
        
        # solve the adjoint
        CFDSolver.setMultiPointFCIndex(i)
        CFDSolver.solveADjoint()
        
        # Evaluate the CFD derivatives
        CFDSolver.evalFunctionsSens(funcsSens,evalFuncs=evalFuncs)
        
        if funcsSens['fail'] == True:
            fail = True
            
        # append "fc" (flow case) to the evalFuncsSens keys    
        for key in funcsSens.keys():
            if 'fail' in key:
                pass
            elif 'DVCon' in key:
                funcsSensMP[key] = funcsSens[key]
            else:
                funcsSensMP['fc%d_'%i+key] = funcsSens[key] 
    
    funcsSensMP['fail'] = fail
    
    # Print the current solution to the screen
    if gcomm.rank == 0:
        print('Objective Function Sensitivity MultiPoint: ',funcsSensMP)

    b = time.time()
    if gcomm.rank == 0:
        print 'Adjoint runtime: ',b-a
    
    return funcsSensMP
    
# Assemble the objective and any additional constraints:
def objCon(funcs, printOK):

    funcs['obj'] = 0.0
    for i in xrange(nFlowCases):
        funcs['obj'] += funcs['fc%d_fx'%i]*MPWeights[i]
        #funcs['fc%d_fy_con'%i] = funcs['fc%d_fy'%i] - CL_star[i]
    if printOK:
       if gcomm.rank == 0:
           print 'MultiPoint Objective Functions:', funcs
    return funcs
        
def run():
    """
    Just run the flow and adjoint
    """
    xDV = DVGeo.getValues()
    
    # Evaluate the functions
    funcs = {}
    funcs,fail = aeroFuncs(xDV)
    
    if gcomm.rank == 0:
        print funcs
        
    # Evaluate the sensitivities
    funcsSens = {}
    funcsSens,fail = aeroFuncsSens(xDV,funcs)
    
    if gcomm.rank == 0:
        print funcsSens

def plotSensMap(runAdjoint=True):
    """
    Run the flow and adjoint, then plot the sensitivity map
    """

    CFDSolver.setOption('adjdvtypes',['Xv'])
    CFDSolver.setOption('epsderivxv',1e-6)

    if runAdjoint==True:
        xDV = DVGeo.getValues()
        
        # Evaluate the functions
        funcs = {}
        funcs,fail = aeroFuncs(xDV)
        
        if gcomm.rank == 0:
            print funcs
            
        # Evaluate the sensitivities
        funcsSens = {}
        funcsSens,fail = aeroFuncsSens(xDV,funcs)
        
        if gcomm.rank == 0:
            print funcsSens
    
    CFDSolver.writeSurfaceSensitivityMap(evalFuncs=evalFuncs,groupName=CFDSolver.getOption('designsurfacefamily'))

def testSensUIn(normStatesList=[True],deltaStateList=[1e-5],deltaUInList=[1e-5],epsFD=1e-3):
    """
    Verify the inlet U sensitivity against finite-difference references
    """

    CFDSolver.setOption('adjdvtypes',['UIn'])

    if gcomm.rank==0:
        fOut = open('./testInletSens.txt','w')
    
    derivUInInfo = CFDSolver.getOption('derivuininfo')
    stateName = derivUInInfo['stateName']
    mode = derivUInInfo['type']
    component = derivUInInfo['component']
    patchNames = derivUInInfo['patchNames']
   
    # gradAdj
    for case in range(len(normStatesList)): 
        normStates = normStatesList[case]
        deltaState=deltaStateList[case]
        deltaUIn=deltaUInList[case]
        
        if normStates:
            CFDSolver.setOption('normalizeresiduals',CFDSolver.getOption('normalizeresiduals'))
            CFDSolver.setOption('normalizestates',CFDSolver.getOption('normalizestates'))
        else:
            CFDSolver.setOption('normalizeresiduals',[])
            CFDSolver.setOption('normalizestates',[])
        CFDSolver.setOption('epsderiv',deltaState)
        CFDSolver.setOption('epsderivuin',deltaUIn)
        
        funcs={}
        funcsSens={}
        CFDSolver()
        CFDSolver.evalFunctions(funcs,evalFuncs=evalFuncs)
        CFDSolver.solveADjoint()
        CFDSolver.evalFunctionsSens(funcsSens,evalFuncs=evalFuncs)

        if gcomm.rank == 0:
            print funcsSens
            fOut.write('Adjoint Results:\n')
            fOut.write('NormStates: '+str(normStates)+' DeltaU: '+str(deltaState)+' DeltaUIn: '+str(deltaUIn)+'\n')
        
        for funcName in sorted(evalFuncs):
            line = funcName+' '+str(funcsSens[funcName]['UIn'])+'\n'
            if gcomm.rank==0:
                fOut.write(line)
                fOut.flush()

    if mode=="rotRad":
        rotRad=CFDSolver.getOption('rotrad')

        rotRad[component] +=epsFD
        CFDSolver.setOption('rotrad', rotRad)
        funcp = {}
        CFDSolver()
        CFDSolver.evalFunctions(funcp,evalFuncs=evalFuncs)
        if gcomm.rank == 0:
            print rotRad
            print(funcp)
        
        rotRad[component] -= 2.0*epsFD
        CFDSolver.setOption('rotrad', rotRad)
        funcm = {}
        CFDSolver()
        CFDSolver.evalFunctions(funcm,evalFuncs=evalFuncs)
        if gcomm.rank == 0:
            print rotRad
            print(funcm)

        gradFD={}
        for funcName in sorted(evalFuncs):
            gradFD[funcName] = ( funcp[funcName]-funcm[funcName] ) / 2.0/epsFD
            
        if gcomm.rank == 0:
            print gradFD
            for funcName in sorted(evalFuncs):
                line = funcName+' '+str(gradFD[funcName])+'\n'
                fOut.write(line)
                fOut.flush()
        if gcomm.rank==0:
            fOut.close()
    elif mode=="tractionDisplacement":
        # gradFD
        flowBCs=CFDSolver.getOption('flowbcs')
        CFDSolver.setOption('setflowbcs',True)

        bcKey=None
        for key in flowBCs.keys():
            if not key == 'useWallFunction':
                if flowBCs[key]['variable']==stateName:
                    bcKey=key
        if bcKey == None:
            print('stateName not found in flowBCs')
            exit()

        if component ==-1: # pressure
            stateRef = flowBCs[bcKey]['pressure'][0]
        else:
            stateRef = flowBCs[bcKey]['value'][component]
        
        if gcomm.rank==0:
            print('Calculating FD Sensitivity',epsFD)
            fOut.write('epsFD: '+str(epsFD)+'\n')
        if component ==-1: # pressure
            flowBCs[bcKey]['pressure'][0] = stateRef+epsFD
        else:
            flowBCs[bcKey]['value'][component] = stateRef+epsFD
        CFDSolver.setOption('flowbcs',flowBCs)
        funcp = {}
        CFDSolver()
        CFDSolver.evalFunctions(funcp,evalFuncs=evalFuncs) 
        if gcomm.rank == 0:
            print flowBCs
            print(funcp)
        if component ==-1: # pressure
            flowBCs[bcKey]['pressure'][0] = stateRef-epsFD
        else:
            flowBCs[bcKey]['value'][component] = stateRef-epsFD
        CFDSolver.setOption('flowbcs',flowBCs)
        funcm = {}
        CFDSolver()
        CFDSolver.evalFunctions(funcm,evalFuncs=evalFuncs) 
        if gcomm.rank == 0:
            print flowBCs
            print(funcm)
        if component == -1:
            flowBCs[bcKey]['pressure'][0] = stateRef
        else:
            flowBCs[bcKey]['value'][component] = stateRef
        CFDSolver.setOption('flowbcs',flowBCs)
        
        gradFD={}
        for funcName in sorted(evalFuncs):
            gradFD[funcName] = ( funcp[funcName]-funcm[funcName] ) / 2.0/epsFD
            
        if gcomm.rank == 0:
            print gradFD
            for funcName in sorted(evalFuncs):
                line = funcName+' '+str(gradFD[funcName])+'\n'
                fOut.write(line)
                fOut.flush()
                
        if gcomm.rank==0:
            fOut.close()
    elif mode in ["fixedValue","fixedGradient"]:          
        # gradFD
        flowBCs=CFDSolver.getOption('flowbcs')
        CFDSolver.setOption('setflowbcs',True)

        bcKey=None
        for key in flowBCs.keys():
            if not key == 'useWallFunction':
                if flowBCs[key]['variable']==stateName:
                    bcKey=key
        if bcKey == None:
            print('stateName not found in flowBCs')
            exit()
            
        stateRef = flowBCs[bcKey]['value'][component]
        
        if gcomm.rank==0:
            print('Calculating FD Sensitivity',epsFD)
            fOut.write('epsFD: '+str(epsFD)+'\n')
    
        flowBCs[bcKey]['value'][component] = stateRef+epsFD
        CFDSolver.setOption('flowbcs',flowBCs)
        funcp = {}
        CFDSolver()
        CFDSolver.evalFunctions(funcp,evalFuncs=evalFuncs) 
        if gcomm.rank == 0:
            print flowBCs
            print(funcp)
        
        flowBCs[bcKey]['value'][component] = stateRef-epsFD
        CFDSolver.setOption('flowbcs',flowBCs)
        funcm = {}
        CFDSolver()
        CFDSolver.evalFunctions(funcm,evalFuncs=evalFuncs) 
        if gcomm.rank == 0:
            print flowBCs
            print(funcm)
        
        flowBCs[bcKey]['value'][component] = stateRef
        CFDSolver.setOption('flowbcs',flowBCs)
        
        gradFD={}
        for funcName in sorted(evalFuncs):
            gradFD[funcName] = ( funcp[funcName]-funcm[funcName] ) / 2.0/epsFD
            
        if gcomm.rank == 0:
            print gradFD
            for funcName in sorted(evalFuncs):
                line = funcName+' '+str(gradFD[funcName])+'\n'
                fOut.write(line)
                fOut.flush()
                
        if gcomm.rank==0:
            fOut.close()
    else:
        print("type not supported!")
        exit()
            
    return

def testSensVis(normStatesList=[True],deltaUList=[1e-5],deltaVisList=[1e-8]):
    """
    Verify the Vis sensitivity against finite-difference references
    """

    xDV = DVGeo.getValues()

    CFDSolver.setOption('adjdvtypes',['Vis'])

    if gcomm.rank==0:
        fOut = open('./testVisSens.txt','w')
    
    # gradAdj
    for normStates in normStatesList: 
        for deltaU in deltaUList:
            for deltaVis in deltaVisList:
                if normStates:
                    CFDSolver.setOption('normalizeresiduals',CFDSolver.getOption('normalizeresiduals'))
                    CFDSolver.setOption('normalizestates',CFDSolver.getOption('normalizestates'))
                else:
                    CFDSolver.setOption('normalizeresiduals',[])
                    CFDSolver.setOption('normalizestates',[])
                CFDSolver.setOption('epsderiv',deltaU)
                CFDSolver.setOption('epsderivvis',deltaVis)
                
                funcs={}
                funcsSens={}
                funcs,fail = aeroFuncs(xDV) 
                funcsSens,fail = aeroFuncsSens(xDV,funcs)
        
                if gcomm.rank == 0:
                    for funcName in sorted(evalFuncs):
                        fOut.write(funcName+' Vis '+str(normStates)+' '+str(deltaU)+' '+str(deltaVis)+'\n')
                        line = str(funcsSens[funcName]['Vis'][0])+'\n'
                        fOut.write(line)
                        fOut.flush()        
    
    # gradFD 
    transDict=CFDSolver.getOption('transproperties')
    nuRef=transDict['nu']
    for deltaVis in deltaVisList:

        if gcomm.rank==0:
            print('-------FD----------',deltaVis)
            fOut.write('DeltaX: '+str(deltaVis)+'\n')
            
        # initialize gradFD
        gradFD = {}

        funcp={}
        funcm={}  
        transDict['nu']=nuRef+deltaVis
        CFDSolver.setOption('transproperties',transDict)
        funcp,fail = aeroFuncs(xDV)
        transDict['nu']=nuRef-deltaVis
        CFDSolver.setOption('transproperties',transDict)
        funcm,fail = aeroFuncs(xDV)
                
        for funcName in sorted(evalFuncs):
            gradFD[funcName]=(funcp[funcName]-funcm[funcName])/(2.0*deltaVis)
        if gcomm.rank==0:
            print(gradFD)
                
        # write FD results
        if gcomm.rank==0:
            for funcName in sorted(evalFuncs):
                fOut.write(funcName+' Vis '+str(normStates)+' '+str(deltaU)+' '+str(deltaVis)+'\n')
                line = str(gradFD[funcName])+'\n'
                fOut.write(line)
                fOut.flush() 
                              
    if gcomm.rank==0:
        fOut.close()
    
    return

def testSensShape(normStatesList=[True],deltaUList=[1e-5],deltaXList=[1e-4],mode='FFD'):
    """
    Verify the FFD sensitivity against finite-difference references
    """
    
    xDV = DVGeo.getValues()
    
    CFDSolver.setOption('adjdvtypes',[mode])

    if gcomm.rank==0:
        fOut = open('./testFFDSens.txt','w')
    
    # gradAdj
    for normStates in normStatesList: 
        for deltaU in deltaUList:
            for deltaX in deltaXList:
                if normStates:
                    CFDSolver.setOption('normalizeresiduals',CFDSolver.getOption('normalizeresiduals'))
                    CFDSolver.setOption('normalizestates',CFDSolver.getOption('normalizestates'))
                else:
                    CFDSolver.setOption('normalizeresiduals',[])
                    CFDSolver.setOption('normalizestates',[])
                CFDSolver.setOption('epsderiv',deltaU)
                CFDSolver.setOption('epsderivffd',deltaX)
                CFDSolver.setOption('epsderivxv',deltaX)
                
                funcs={}
                funcsSens={}
                funcs,fail = aeroFuncs(xDV) 
                funcsSens,fail = aeroFuncsSens(xDV,funcs)
        
                if gcomm.rank == 0:
                    for funcName in sorted(evalFuncs):
                        for shapeVar in sorted(xDV.keys()):
                            fOut.write(funcName+' '+shapeVar+' '
                                    +str(normStates)+' '+str(deltaU)+' '+str(deltaX)+'\n')
                            try:
                                nDVs = len(funcsSens[funcName][shapeVar])
                            except:
                                nDVs = 1
                            for n in range(nDVs):
                                line = str(funcsSens[funcName][shapeVar][n])+'\n'
                                fOut.write(line)
                                fOut.flush()        
    
    # gradFD 
    for deltaX in deltaXList:
        # initialize gradFD
        gradFD = {}
        for funcName in sorted(evalFuncs):
            gradFD[funcName] = {}
            for shapeVar in sorted(xDV.keys()):
                gradFD[funcName][shapeVar] = np.zeros(len(xDV[shapeVar]))
    
        if gcomm.rank==0:
            print('-------FD----------',deltaX)
            fOut.write('DeltaX: '+str(deltaX)+'\n')
            
        for shapeVar in sorted(xDV.keys()):
            try:
                nDVs = len(xDV[shapeVar])
            except:
                nDVs = 1
            for i in range(nDVs):
                funcp={}
                funcm={}  
                xDV[shapeVar][i] +=deltaX
                funcp,fail = aeroFuncs(xDV)
                xDV[shapeVar][i] -=2.0*deltaX
                funcm,fail = aeroFuncs(xDV)
                xDV[shapeVar][i] +=deltaX
                
                for funcName in sorted(evalFuncs):
                    gradFD[funcName][shapeVar][i]=(funcp[funcName]-funcm[funcName])/(2.0*deltaX)
                if gcomm.rank==0:
                    print(gradFD)
                
        # write FD results
        if gcomm.rank==0:
            for funcName in sorted(evalFuncs):
                for shapeVar in sorted(xDV.keys()):
                    fOut.write(funcName+' '+shapeVar+' '
                                +str(normStates)+' '+str(deltaU)+' '+str(deltaX)+'\n')
                    try:
                        nDVs = len(gradFD[funcName][shapeVar])
                    except:
                        nDVs = 1
                    for n in range(nDVs):
                        line = str(gradFD[funcName][shapeVar][n])+'\n'
                        fOut.write(line)
                        fOut.flush() 
                              
    if gcomm.rank==0:
        fOut.close()
    
    return
    
def xDV2xV():
    """
    Read the design variables from files and deform the volume mesh accordingly
    The file name should be DVNames.dat
    """

    # read the design variable values
    f = open('designVariables.dat','r')
    lines = f.readlines()
    f.close()
    newDV = {}
    for line in lines:
	cols = line.split()
        if not cols: # empty
            break
        newDV[cols[0]]=[]
        for val in cols[1:]:
            newDV[cols[0]].append(float(val))

    if gcomm.rank==0:
        print("Values in designVariables.dat: ",newDV)
	
    xDV = DVGeo.getValues()

    if gcomm.rank==0:
        print("DVGeo initial ",xDV)    

    for key in xDV:
        length = len(xDV[key])
        for i in range(length):
            xDV[key][i]=newDV[key][i]
    
    if gcomm.rank==0:
        print("DVGeo set: ",xDV)
	
    DVGeo.setDesignVars(xDV)
    CFDSolver.writeUpdatedVolumePoints()
