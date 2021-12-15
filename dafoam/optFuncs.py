#!/usr/bin/env python
"""

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
    Common functions for DAFoam optimization setup.

"""


# =============================================================================
# Imports
# =============================================================================
import time
import sys
import numpy as np
import warnings
import copy

warnings.filterwarnings("once")
np.set_printoptions(precision=16, suppress=True)


def calcObjFuncValues(xDV):
    """
    Update the design surface and run the primal solver to get objective function values.
    """

    Info("\n")
    Info("+--------------------------------------------------------------------------+")
    Info("|                  Evaluating Objective Functions %03d                      |" % DASolver.nSolvePrimals)
    Info("+--------------------------------------------------------------------------+")
    Info("Design Variables: ")
    Info(xDV)

    a = time.time()

    # Setup an empty dictionary for the evaluated function values
    funcs = {}

    # Set the current design variables in the DV object
    DVGeo.setDesignVars(xDV)
    DASolver.setDesignVars(xDV)

    # Evaluate the geometric constraints and add them to the funcs dictionary
    DVCon.evalFunctions(funcs)

    # Solve the CFD problem
    DASolver()

    # Populate the required values from the CFD problem
    DASolver.evalFunctions(funcs, evalFuncs=evalFuncs)

    b = time.time()

    # Print the current solution to the screen
    Info("Objective Functions: ")
    Info(funcs)
    Info("Flow Runtime: %g" % (b - a))

    fail = funcs["fail"]

    return funcs, fail


def calcObjFuncValuesMP(xDV):
    """
    Update the design surface and run the primal solver to get objective function values.
    This is the multipoint version of calcObjFuncValues
    """

    Info("\n")
    Info("+--------------------------------------------------------------------------+")
    Info("|                  Evaluating Objective Functions %03d                      |" % DASolver.nSolvePrimals)
    Info("+--------------------------------------------------------------------------+")
    Info("Design Variables: ")
    Info(xDV)

    a = time.time()

    fail = False

    # Setup an empty dictionary for the evaluated function values
    funcsMP = {}

    # Set the current design variables in the DV object
    DVGeo.setDesignVars(xDV)
    DASolver.setDesignVars(xDV)

    nMultiPoints = DASolver.getOption("nMultiPoints")

    for i in range(nMultiPoints):

        Info("--Solving Primal for Configuration %d--" % i)

        funcs = {}

        # Evaluate the geometric constraints and add them to the funcs dictionary
        DVCon.evalFunctions(funcs)

        # set the multi point condition provided by users in the
        # runScript.py script. This function should define what
        # conditions to change for each case i
        setMultiPointCondition(xDV, i)

        # Solve the CFD problem
        DASolver()

        # Populate the required values from the CFD problem
        DASolver.evalFunctions(funcs, evalFuncs=evalFuncs)

        # save the state vector for case i and will be used in solveAdjoint
        DASolver.saveMultiPointField(i)

        # if any of the multipoint primal fails, return fail=True
        if funcs["fail"] is True:
            fail = True

        if DASolver.getOption("debug"):
            Info("Objective Functions for Configuration %d: " % i)
            Info(funcs)

        # assign funcs to funcsMP
        setMultiPointObjFuncs(funcs, funcsMP, i)

    funcsMP["fail"] = fail

    Info("Objective Functions MultiPoint: ")
    Info(funcsMP)

    b = time.time()
    Info("Flow Runtime: %g" % (b - a))

    return funcsMP, fail


def calcObjFuncValuesUnsteady(xDV):
    """
    Update the design surface and run the primal solver to get objective function values.
    This is the unsteady adjoint version of calcObjFuncValues
    """

    Info("\n")
    Info("+--------------------------------------------------------------------------+")
    Info("|                  Evaluating Objective Functions %03d                      |" % DASolver.nSolvePrimals)
    Info("+--------------------------------------------------------------------------+")
    Info("Design Variables: ")
    Info(xDV)

    a = time.time()

    # Setup an empty dictionary for the evaluated function values
    funcs = {}

    # Set the current design variables in the DV object
    DVGeo.setDesignVars(xDV)
    DASolver.setDesignVars(xDV)

    # Evaluate the geometric constraints and add them to the funcs dictionary
    DVCon.evalFunctions(funcs)

    # Solve the CFD problem
    DASolver()

    # Set values for the unsteady adjoint objectives. This function needs to be
    # implemented in run scripts
    setObjFuncsUnsteady(DASolver, funcs, evalFuncs)

    # assign state lists to mats
    DASolver.setTimeInstanceVar(mode="list2Mat")

    b = time.time()

    # Print the current solution to the screen
    Info("Objective Functions: ")
    Info(funcs)
    Info("Flow Runtime: %g" % (b - a))

    fail = funcs["fail"]

    return funcs, fail


def calcObjFuncSens(xDV, funcs):
    """
    Run the adjoint solver and get objective function sensitivities.
    """

    Info("\n")
    Info("+--------------------------------------------------------------------------+")
    Info("|              Evaluating Objective Function Sensitivities %03d             |" % DASolver.nSolveAdjoints)
    Info("+--------------------------------------------------------------------------+")

    a = time.time()

    # write the design variable values to file
    DASolver.writeDesignVariable("designVariableHist.txt", xDV)

    # write the deform FFDs
    DASolver.writeDeformedFFDs()

    # Setup an empty dictionary for the evaluated derivative values
    funcsSens = {}

    # Evaluate the geometric constraint derivatives
    DVCon.evalFunctionsSens(funcsSens)

    # Solve the adjoint
    DASolver.solveAdjoint()

    # Evaluate the CFD derivatives
    DASolver.evalFunctionsSens(funcsSens, evalFuncs=evalFuncs)

    b = time.time()

    # Print the current solution to the screen
    with np.printoptions(precision=16, threshold=5, suppress=True):
        Info("Objective Function Sensitivity: ")
        Info(funcsSens)
        Info("Adjoint Runtime: %g s" % (b - a))

    # write the sensitivity values to file
    DASolver.writeTotalDeriv("totalDerivHist.txt", funcsSens, evalFuncs)

    fail = funcsSens["fail"]

    return funcsSens, fail


def calcObjFuncSensMP(xDV, funcs):
    """
    Run the adjoint solver and get objective function sensitivities.
    This is the multipoint version of calcObjFuncSens
    """

    Info("\n")
    Info("+--------------------------------------------------------------------------+")
    Info("|              Evaluating Objective Function Sensitivities %03d             |" % DASolver.nSolveAdjoints)
    Info("+--------------------------------------------------------------------------+")

    fail = False

    a = time.time()

    # write the design variable values to file
    DASolver.writeDesignVariable("designVariableHist.txt", xDV)

    # write the deform FFDs
    DASolver.writeDeformedFFDs()

    # Setup an empty dictionary for the evaluated derivative values
    funcsSensMP = {}

    nMultiPoints = DASolver.getOption("nMultiPoints")

    for i in range(nMultiPoints):

        Info("--Solving Adjoint for Configuration %d--" % i)

        funcsSens = {}

        # Evaluate the geometric constraint derivatives
        DVCon.evalFunctionsSens(funcsSens)

        # set the state vector for case i
        DASolver.setMultiPointField(i)

        # set the multi point condition provided by users in the
        # runScript.py script. This function should define what
        # conditions to change for each case i
        setMultiPointCondition(xDV, i)

        # Solve the adjoint
        DASolver.solveAdjoint()

        # Evaluate the CFD derivatives
        DASolver.evalFunctionsSens(funcsSens, evalFuncs=evalFuncs)

        if funcsSens["fail"] is True:
            fail = True

        if DASolver.getOption("debug"):
            with np.printoptions(precision=16, threshold=5, suppress=True):
                Info("Objective Function Sensitivity: ")
                Info(funcsSens)

        # assign funcs to funcsMP
        setMultiPointObjFuncsSens(xDV, funcs, funcsSens, funcsSensMP, i)

    funcsSensMP["fail"] = fail

    # Print the current solution to the screen
    with np.printoptions(precision=16, threshold=5, suppress=True):
        Info("Objective Function Sensitivity MultiPoiint: ")
        Info(funcsSensMP)

    b = time.time()
    Info("Adjoint Runtime: %g s" % (b - a))

    return funcsSensMP, fail


def calcObjFuncSensUnsteady(xDV, funcs):
    """
    Run the adjoint solver and get objective function sensitivities.
    This is the unsteady adjoint version of calcObjFuncSens
    """

    Info("\n")
    Info("+--------------------------------------------------------------------------+")
    Info("|              Evaluating Objective Function Sensitivities %03d             |" % DASolver.nSolveAdjoints)
    Info("+--------------------------------------------------------------------------+")

    fail = False

    a = time.time()

    # write the design variable values to file
    DASolver.writeDesignVariable("designVariableHist.txt", xDV)

    # write the deform FFDs
    DASolver.writeDeformedFFDs()

    # assign the state mats to lists
    DASolver.setTimeInstanceVar(mode="mat2List")

    # Setup an empty dictionary for the evaluated derivative values
    funcsSensCombined = {}

    funcsSensAllInstances = []

    mode = DASolver.getOption("unsteadyAdjoint")["mode"]
    nTimeInstances = DASolver.getOption("unsteadyAdjoint")["nTimeInstances"]
    if mode == "hybridAdjoint":
        iEnd = -1
    elif mode == "timeAccurateAdjoint":
        iEnd = 0

    # NOTE: calling calcRes here is critical because it will setup the correct
    # old time levels for setTimeInstanceField. Otherwise, the residual for the
    # first adjoint time instance will be incorrect because the residuals have
    # not been computed and the old time levels will be zeros for all variables,
    # this will create issues for the setTimeInstanceField call (nOldTimes)
    DASolver.calcPrimalResidualStatistics("calc")

    # set these vectors zeros
    DASolver.zeroTimeAccurateAdjointVectors()

    for i in range(nTimeInstances - 1, iEnd, -1):

        Info("--Solving Adjoint for Time Instance %d--" % i)

        funcsSens = {}

        # Evaluate the geometric constraint derivatives
        DVCon.evalFunctionsSens(funcsSens)

        # set the state vector for case i
        DASolver.setTimeInstanceField(i)

        # Solve the adjoint
        DASolver.solveAdjoint()

        # Evaluate the CFD derivatives
        DASolver.evalFunctionsSens(funcsSens, evalFuncs=evalFuncs)

        if funcsSens["fail"] is True:
            fail = True

        if DASolver.getOption("debug"):
            with np.printoptions(precision=16, threshold=5, suppress=True):
                Info("Objective Function Sensitivity: ")
                Info(funcsSens)

        funcsSensAllInstances.append(funcsSens)

    setObjFuncsSensUnsteady(DASolver, funcs, funcsSensAllInstances, funcsSensCombined)

    funcsSensCombined["fail"] = fail

    # Print the current solution to the screen
    with np.printoptions(precision=16, threshold=5, suppress=True):
        Info("Objective Function Sensitivity Unsteady Adjoint: ")
        Info(funcsSensCombined)

    b = time.time()
    Info("Adjoint Runtime: %g s" % (b - a))

    return funcsSensCombined, fail


def runPrimal(objFun=calcObjFuncValues):
    """
    Just run the primal
    """

    xDV = DVGeo.getValues()
    funcs = {}
    funcs, fail = objFun(xDV)

    return funcs, fail


def runAdjoint(objFun=calcObjFuncValues, sensFun=calcObjFuncSens, fileName=None):
    """
    Just run the primal and adjoint
    """

    DASolver.runColoring()
    xDV = DVGeo.getValues()
    funcs = {}
    funcs, fail = objFun(xDV)
    funcsSens = {}
    funcsSens, fail = sensFun(xDV, funcs)

    # Optionally, we can write the sensitivity to a file if fileName is provided
    if fileName is not None:
        if gcomm.rank == 0:
            fOut = open(fileName, "w")
            for funcName in evalFuncs:
                for shapeVar in xDV:
                    fOut.write(funcName + " " + shapeVar + "\n")
                    try:
                        nDVs = len(funcsSens[funcName][shapeVar])
                    except Exception:
                        nDVs = 1
                    for n in range(nDVs):
                        line = str(funcsSens[funcName][shapeVar][n]) + "\n"
                        fOut.write(line)
                        fOut.flush()
            fOut.close()

    return funcsSens, fail

def runForwardAD(dvName="None", seedIndex=-1):
    """
    Run the forward mode AD for the primal solver to compute the brute force total
    derivative. This is primarily used in verification of the adjoint accuracy
    """

    if not DASolver.getOption("useAD")["mode"] == "forward":
        Info("runForwardAD only supports useAD->mode=forward!")
        Info("Please set useAD->mode to forward and rerun!")
        exit(1)
    DASolver.setOption("useAD",{"dvName": dvName, "seedIndex": seedIndex})
    DASolver.updateDAOption()
    DASolver()

def solveCL(CL_star, alphaName, liftName, objFun=calcObjFuncValues, eps=1e-2, tol=1e-4, maxit=10):
    """
    Adjust the angle of attack or pitch to match the target lift.
    This is usually needed for wing aerodynamic optimization
    """

    Info("\n")
    Info("+--------------------------------------------------------------------------+")
    Info("|              Running SolveCL to find alpha that matches target CL        |")
    Info("+--------------------------------------------------------------------------+")
    Info("eps: %g  tol: %g  maxit: %g" % (eps, tol, maxit))

    xDVs = DVGeo.getValues()
    alpha = xDVs[alphaName]

    for i in range(maxit):
        # Solve the CFD problem
        xDVs[alphaName] = alpha
        funcs = {}
        funcs, fail = objFun(xDVs)
        CL0 = funcs[liftName]
        Info("alpha: %f, CL: %f" % (alpha.real, CL0))
        if abs(CL0 - CL_star) / CL_star < tol:
            Info("Completed! alpha = %f" % alpha.real)
            return alpha.real
        # compute sens
        alphaVal = alpha + eps
        xDVs[alphaName] = alphaVal
        funcsP = {}
        funcsP, fail = objFun(xDVs)
        CLP = funcsP[liftName]
        deltaAlpha = (CL_star - CL0) * eps / (CLP - CL0)
        alpha += deltaAlpha

    return alpha.real


def calcFDSens(objFun=calcObjFuncValues, fileName=None):
    """
    Compute finite-difference sensitivity
    """

    xDV = DVGeo.getValues()

    # gradFD
    deltaX = DASolver.getOption("adjPartDerivFDStep")["FFD"]
    # initialize gradFD
    gradFD = {}
    for funcName in evalFuncs:
        gradFD[funcName] = {}
        for shapeVar in xDV:
            gradFD[funcName][shapeVar] = np.zeros(len(xDV[shapeVar]))
    if gcomm.rank == 0:
        print("-------FD----------", deltaX, flush=True)

    for shapeVar in xDV:
        try:
            nDVs = len(xDV[shapeVar])
        except Exception:
            nDVs = 1
        for i in range(nDVs):
            funcp = {}
            funcm = {}
            xDV[shapeVar][i] += deltaX
            funcp, fail = objFun(xDV)
            xDV[shapeVar][i] -= 2.0 * deltaX
            funcm, fail = objFun(xDV)
            xDV[shapeVar][i] += deltaX
            for funcName in evalFuncs:
                gradFD[funcName][shapeVar][i] = (funcp[funcName] - funcm[funcName]) / (2.0 * deltaX)
            Info(gradFD)
    # write FD results
    if fileName is not None:
        if gcomm.rank == 0:
            fOut = open(fileName, "w")
            fOut.write("DeltaX: " + str(deltaX) + "\n")
            for funcName in evalFuncs:
                for shapeVar in xDV:
                    fOut.write(funcName + " " + shapeVar + "\n")
                    try:
                        nDVs = len(gradFD[funcName][shapeVar])
                    except Exception:
                        nDVs = 1
                    for n in range(nDVs):
                        line = str(gradFD[funcName][shapeVar][n]) + "\n"
                        fOut.write(line)
                        fOut.flush()
            fOut.close()


class Info(object):
    """
    Print information and flush to screen for parallel cases
    """

    def __init__(self, message):
        if gcomm.rank == 0:
            print(message, flush=True)
        gcomm.Barrier()
