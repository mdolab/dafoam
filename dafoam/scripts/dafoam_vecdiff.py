#!/usr/bin/env python
"""
Compare values in two different vectors and identify the max relative/absolute error
"""
import os, sys
import argparse
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc


def evalVecDiff(vecName1, vecName2, mode, diffTol=1e-30):

    # the relative error is computed as
    # (vec1-vec2)/vec1

    # read  vec 1
    vec1 = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(vecName1, comm=PETSc.COMM_WORLD)
    vec1.load(viewer)

    # read  vec 2
    vec2 = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(vecName2, comm=PETSc.COMM_WORLD)
    vec2.load(viewer)

    # read  vec Diff
    vecDiff = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(vecName1, comm=PETSc.COMM_WORLD)
    vecDiff.load(viewer)

    Istart, Iend = vec1.getOwnershipRange()

    vecDiff.axpy(-1.0, vec2)

    maxDiff = -1.0e30
    maxVal1 = -1.0e30
    maxVal2 = -1.0e30
    maxRowI = -1.0e30
    l2norm = 0.0
    foundDiff = 0
    for i in range(Istart, Iend):
        valDiff = abs(vecDiff[i])
        valRef = abs(vec1[i])
        if mode == "rel":
            valError = valDiff / (valRef + diffTol)
        elif mode == "abs":
            valError = valDiff
        else:
            print("mode not supported! Options are: abs or rel")
        l2norm = l2norm + valDiff ** 2
        if abs(valError) > diffTol:
            if abs(valError) > maxDiff:
                maxDiff = valError
                maxRowI = i
                maxVal1 = vec1.getValue(i)
                maxVal2 = vec2.getValue(i)
                foundDiff = 1

    if foundDiff == 1:
        print("L2Norm: %20.16e" % l2norm)
        print("MaxDiff: %20.16e" % maxDiff)
        print("MaxVal1: %20.16e" % maxVal1)
        print("MaxVal2: %20.16e" % maxVal2)
        print("MaxrowI: %d" % maxRowI)
        return maxDiff
    else:
        print("Two vectors are exactly same with tolerance: %e" % diffTol)
        return 0.0, 0.0


if __name__ == "__main__":
    print("\nUsage: python dafoam_vecreldiff.py vecName1 vecName2 mode")
    print("Example python dafoam_vecreldiff.py dFdW1.bin dFdW2.bin abs\n")
    evalVecDiff(sys.argv[1], sys.argv[2], sys.argv[3])
