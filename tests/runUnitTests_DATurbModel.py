#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
import sys
import numpy as np
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)


gcomm = MPI.COMM_WORLD
os.chdir("./reg_test_files-main/ConvergentChannel")


def runTurbTests(daOptions, gcomm, turbName, refs):
    if gcomm.rank == 0:
        os.system("cp -r constant/turbulenceProperties.%s constant/turbulenceProperties" % turbName)

    DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
    DASolver()
    states = DASolver.getStates()
    norm = np.linalg.norm(states)
    norm = gcomm.allreduce(norm, op=MPI.SUM)
    print(norm)
    if (abs(refs[0] - norm) / (norm + 1e-16)) > 1e-10:
        print("%s test failed!" % turbName)
        exit(1)
    else:
        print("%s test passed!" % turbName)

    dRdWTPC = PETSc.Mat().create(PETSc.COMM_WORLD)
    nLocalAdjointStates = DASolver.getNLocalAdjointStates()
    dRdWTPC.setSizes(((nLocalAdjointStates, None), (nLocalAdjointStates, None)))
    dRdWTPC.setFromOptions()
    dRdWTPC.setPreallocationNNZ((100, 100))
    dRdWTPC.setUp()
    turbOnly = 1

    dRdWTPC.zeroEntries()
    DASolver.solver.calcPCMatWithFvMatrix(dRdWTPC, turbOnly)
    norm = dRdWTPC.norm()
    print(norm)
    if (abs(refs[1] - norm) / (norm + 1e-16)) > 1e-8:
        print("%s test failed!" % turbName)
        exit(1)
    else:
        print("%s test passed!" % turbName)

    states = DASolver.getStates()
    seed = np.ones_like(states) * 0.001
    product = np.zeros_like(states)
    DASolver.solverAD.calcJacTVecProduct(
        "stateName",
        "stateVar",
        states,
        "residualName",
        "residual",
        seed,
        product,
    )
    norm = np.linalg.norm(product)
    norm = gcomm.allreduce(norm, op=MPI.SUM)
    print(norm)
    if (abs(refs[2] - norm) / (norm + 1e-16)) > 1e-8:
        print("%s test failed!" % turbName)
        exit(1)
    else:
        print("%s test passed!" % turbName)


# *********************
# compressible models
# *********************
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.compressible/* 0/")
    os.system("cp -r system.subsonic/* system/")

daOptions = {
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1e-12,
    "primalMinResTolDiff": 1e12,
    "printDAOptions": False,
    "primalBC": {
        "useWallFunction": False,
    },
}

runTurbTests(daOptions, gcomm, "sa", [3786983.0152606126, 24688.14724830308, 1377093.7901194005])
runTurbTests(daOptions, gcomm, "safv3", [3786983.4415608784, 24723.757496203303, 1377040.8468381139])
runTurbTests(daOptions, gcomm, "sst", [3787030.9471557215, 28962.17463329316, 1390611.8099678971])
runTurbTests(daOptions, gcomm, "kw", [3787032.628925756, 28956.64422546926, 1390393.340066838])
runTurbTests(daOptions, gcomm, "sstlm", [3787217.019831411, 0.0, 1390631.4948573676])
daOptions["primalBC"]["useWallFunction"] = True
runTurbTests(daOptions, gcomm, "ke", [3787224.67554039, 32907.42061875005, 1372907.024896523])

# *********************
# incompressible models
# *********************
if gcomm.rank == 0:
    os.system("rm -rf 0/* processor* *.bin")
    os.system("cp -r 0.incompressible/* 0/")
    os.system("cp -r system.incompressible/* system/")

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1e-12,
    "primalMinResTolDiff": 1e12,
    "printDAOptions": False,
    "primalBC": {
        "useWallFunction": False,
    },
}


runTurbTests(daOptions, gcomm, "sa", [11267.58717789064, 4641.040337039481, 1732.238877108044])
runTurbTests(daOptions, gcomm, "safv3", [11267.58536537872, 4643.017299973399, 1732.1877430108193])
runTurbTests(daOptions, gcomm, "sst", [12710.942795556493, 5056.461333737587, 4260.63747585974])
runTurbTests(daOptions, gcomm, "kw", [12843.518233534207, 5057.952497987695, 4341.259273044216])
runTurbTests(daOptions, gcomm, "sstlm", [21113.42085052999, 0.0, 4392.621225985673])
daOptions["primalBC"]["useWallFunction"] = True
runTurbTests(daOptions, gcomm, "ke", [11275.140206006967, 7536.877768251798, 3357.6905407908644])
