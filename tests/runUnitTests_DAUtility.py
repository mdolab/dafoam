#!/usr/bin/env python

import dafoam
import os, subprocess, sys
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
from dafoam.libs.pyUnitTests import pyUnitTests

os.chdir("./reg_test_files-main/ConvergentChannel")
if MPI.COMM_WORLD.rank == 0:
    subprocess.call("decomposePar", stdout=sys.stdout, stderr=subprocess.STDOUT, shell=False)

solverArg = "unitTests -python -parallel"
options = {
    "key0": [int, 2],
    "key1": 1,
    "key2": 2.5,
    "key3": "test",
    "key4": True,
    "key5": [1, 2],
    "key6": [2.1, 3.2],
    "key7": ["test1", "test2"],
    "key8": [False, True],
    "key9": [[1, 2], [3, 4]],
    "key10": [[1.2, 2.1], [3.4, 4.3]],
    "key11": {
        "subKey1": 1,
        "subKey2": 2.5,
        "subKey3": "test",
        "subKey4": False,
        "subKey5": [1, 2],
        "subKey6": [3.5, 4.1],
        "subKey7": ["test1", "test2"],
        "subKey8": {
            "subSubKey1": 1,
            "subSubKey2": 2.5,
            "subSubKey3": "test",
            "subSubKey4": False,
            "subSubKey5": [1, 2],
            "subSubKey6": [3.5, 4.1],
            "subSubKey7": ["test1", "test2"],
        },
    },
}

solver = pyUnitTests()
solver.runDAUtilityTest1(solverArg.encode(), options)
