#!/usr/bin/env python

import dafoam
from pyUnitTests import pyUnitTests

solverArg = "unitTests"
options = {
    "key1": 1,
    "key2": 2.5,
    "key3": "test",
    "key4": True,
    "key5": [1, 2],
    "key6": [2.1, 3.2],
    "key7": ["test1", "test2"],
    "key8": {
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
solver.runDAOptionTest1(solverArg.encode(), options)
