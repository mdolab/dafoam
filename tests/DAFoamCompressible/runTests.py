#!/usr/bin/env python
"""
Run C++ tests
"""

from mpi4py import MPI
from pyTestDAFoamCompressible import pyTestDAFoamCompressible
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


parallelFlag = ""
if comm.size > 1:
    parallelFlag = "-parallel"
solverArg = "TestDAFoamCompressible -python " + parallelFlag
tests = pyTestDAFoamCompressible(solverArg.encode())

# Test1: DAStateInfo
os.chdir("../input/CurvedCubeHexMesh")
testDict = {"solverName": [str, "DARhoSimpleFoam"], "turbulenceModel": [str, "SpalartAllmaras"]}
testErrors = tests.testDAStateInfo(testDict)
checkErrors("DAStateInfo", testErrors)
os.chdir("../../DAFoamCompressible")
