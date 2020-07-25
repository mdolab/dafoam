#!/usr/bin/env python
"""
Run C++ tests
"""

from mpi4py import MPI
from pyTestDAFoamIncompressible import pyTestDAFoamIncompressible
import sys
import os
import petsc4py

petsc4py.init(sys.argv)

comm = MPI.COMM_WORLD


class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """

    def __init__(self, message):
        msg = "\n+" + "-" * 78 + "+" + "\n" + "| pyDAFoam Error: "
        i = 19
        for word in message.split():
            if len(word) + i + 1 > 78:  # Finish line and start new one
                msg += " " * (78 - i) + "|\n| " + word + " "
                i = 1 + len(word) + 1
            else:
                msg += word + " "
                i += len(word) + 1
        msg += " " * (78 - i) + "|\n" + "+" + "-" * 78 + "+" + "\n"
        print(msg)
        Exception.__init__(self)

        return


def checkErrors(testName, errorCode):
    if errorCode != 0:
        raise Error("Tests Failed for %s! Rank %d " % (testName, comm.rank))
    else:
        print("Tests Passed for %s! Rank %d" % (testName, comm.rank))

defOpts = {
    # primal options
    "primalVarBounds": [dict, {}],
    "flowCondition": [str, "Incompressible"],
    "turbulenceModel": [str, "SpalartAllmaras"],
    "primalBC": [dict, {}],
    "fvSource": [dict, {}],
    "printInterval": [int, 100],
    "primalMinResTol": [float, 1.0e-8],
    "primalMinResTolDiff": [float, 1.0e2],
    # adjoint options
    "adjUseColoring": [bool, True],
    "adjPartDerivFDStep": [dict, {"State": 1.0e-5, "FFD": 1.0e-3, "BC": 1.0e-2, "AOA": 1.0e-3,},],
    "adjStateOrdering": [str, "state"],
    "adjEqnOption": [
        dict,
        {
            "globalPCIters": 0,
            "asmOverlap": 1,
            "localPCIters": 1,
            "jacMatReOrdering": "rcm",
            "pcFillLevel": 1,
            "gmresMaxIters": 1000,
            "gmresRestart": 1000,
            "gmresRelTol": 1.0e-6,
            "gmresAbsTol": 1.0e-14,
            "gmresTolDiff": 1.0e2,
        },
    ],
    "normalizeStates": [dict, {}],
    "normalizeResiduals": [list, ["URes", "pRes", "nuTildaRes", "phiRes", "TRes"]],
    "maxResConLv4JacPCMat": [
        dict,
        {
            "pRes": 2,
            "phiRes": 1,
            "URes": 2,
            "TRes": 2,
            "nuTildaRes": 2,
            "kRes": 2,
            "epsilonRes": 2,
            "omegaRes": 2,
            "p_rghRes": 2,
        },
    ],
    "transonicPCOption": [int, -1],
    # optimization options
    "designVar": [dict, {}],
    # system options
    "rootDir": [str, "./"],
    "solverName": [str, "DASimpleFoam"],
    "printAllOptions": [bool, True],
    "objFunc": [dict, {}],
    "debug": [bool, False],
    # surface definition
    "meshSurfaceFamily": [str, "None"],
    "designSurfaceFamily": [str, "None"],
    "designSurfaces": [list, ["body"]],
}

pyDict = {
    "key1": [int, 15],
    "key2": [float, 5.5],
    "key3": [str, "solver1"],
    "key4": [bool, False],
    "key5": [list, [1, 2, 3]],
    "key6": [list, [1.5, 2.3, 3.4]],
    "key7": [list, ["ele1", "ele2", "ele3"]],
    "key8": [list, [False, True, False]],
    "key9": [
        dict,
        {
            "subkey1": [int, 30],
            "subkey2": [float, 3.5],
            "subkey3": [str, "solver2"],
            "subkey4": [bool, True],
            "subkey5": [list, [4, 5, 6]],
            "subkey6": [list, [2.5, 7.7, 8.9]],
            "subkey7": [list, ["ele4", "ele5", "ele6"]],
            "subkey8": [list, [True, False, True]],
        },
    ],
}

pyStrList = ["a", "B", "C1"]

parallelFlag = ""
if comm.size > 1:
    parallelFlag = "-parallel"
solverArg = "TestDAFoamIncompressible -python " + parallelFlag
tests = pyTestDAFoamIncompressible(solverArg.encode())

# Test1: DAUtility
os.chdir("../input/CurvedCubeHexMesh")
testErrors = tests.testDAUtility(pyDict, pyStrList)
checkErrors("DAUtility", testErrors)
os.chdir("../../DAFoamIncompressible")

# Test2: DAOption
os.chdir("../input/CurvedCubeHexMesh")
testErrors = tests.testDAOption(pyDict)
checkErrors("DAOption", testErrors)
os.chdir("../../DAFoamIncompressible")

# Test3: DAModel
os.chdir("../input/CurvedCubeHexMesh")
testErrors = tests.testDAModel(defOpts)
checkErrors("DAModel", testErrors)
os.chdir("../../DAFoamIncompressible")

# Test4: DAStateInfo
os.chdir("../input/CurvedCubeHexMesh")
testErrors = tests.testDAStateInfo(defOpts)
checkErrors("DAStateInfo", testErrors)
os.chdir("../../DAFoamIncompressible")

# Test5: DAObjFunc
os.chdir("../input/CurvedCubeHexMesh")
testErrors = tests.testDAObjFunc(defOpts)
checkErrors("DAObjFunc", testErrors)
os.chdir("../../DAFoamIncompressible")

# Test6: DAField
os.chdir("../input/CurvedCubeHexMesh")
testErrors = tests.testDAField(defOpts)
checkErrors("DAField", testErrors)
os.chdir("../../DAFoamIncompressible")
