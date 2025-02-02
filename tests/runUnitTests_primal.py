#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os

gcomm = MPI.COMM_WORLD

os.chdir("./reg_test_files-main/ConvergentChannel")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")
    os.system("cp -r constant/turbulenceProperties.ke constant/turbulenceProperties")

# aero setup
U0 = 10.0

daOptions = {"solverName": "DARhoSimpleFoam", "primalMinResTol": 1e-12, "printDAOptions": False}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
states = DASolver.getStates()
norm = np.linalg.norm(states)
norm = gcomm.allreduce(norm, op=MPI.SUM)
if ((3787018.3272404578 - norm) / norm) > 1e-10:
    print("ke test failed!")
    exit(1)
else:
    print("ke test passed!")

if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")
    os.system("cp -r constant/turbulenceProperties.kw constant/turbulenceProperties")

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
states = DASolver.getStates()
norm = np.linalg.norm(states)
norm = gcomm.allreduce(norm, op=MPI.SUM)
if ((3787032.628925756 - norm) / norm) > 1e-10:
    print("kw test failed!")
    exit(1)
else:
    print("kw test passed!")

if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")
    os.system("cp -r constant/turbulenceProperties.sstlm constant/turbulenceProperties")

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
states = DASolver.getStates()
norm = np.linalg.norm(states)
norm = gcomm.allreduce(norm, op=MPI.SUM)
if ((3787217.019831411 - norm) / norm) > 1e-10:
    print("SSTLM test failed!")
    exit(1)
else:
    print("SSTLM test passed!")
