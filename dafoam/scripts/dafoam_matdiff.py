#!/usr/bin/env python
"""
Compare values in two different Petsc matrices and identify the max relative/absolute error
"""

import os, sys
import argparse
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc


def evalMatDiff(mat1, mat2, mode, diffTol=1e-30):
    # the relative error is computed as
    # (mat1-mat2)/mat1

    # read  mat 1
    jacMat1 = PETSc.Mat().create(PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(mat1, comm=PETSc.COMM_WORLD)
    jacMat1.load(viewer)

    # read  mat 2
    jacMat2 = PETSc.Mat().create(PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(mat2, comm=PETSc.COMM_WORLD)
    jacMat2.load(viewer)

    Istart, Iend = jacMat1.getOwnershipRange()

    # absolute error
    jacMatDiff = PETSc.Mat().create(PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(mat1, comm=PETSc.COMM_WORLD)
    jacMatDiff.load(viewer)
    jacMatDiff.axpy(-1.0, jacMat2, structure=jacMatDiff.Structure.DIFFERENT_NONZERO_PATTERN)

    maxDiff = -1.0e30
    maxVal1 = -1.0e30
    maxVal2 = -1.0e30
    maxRowI = -1.0e30
    maxColI = -1.0e30
    l2norm = 0.0
    foundDiff = 0
    for i in range(Istart, Iend):
        rowVals = jacMatDiff.getRow(i)
        nCols = len(rowVals[0])
        for j in range(nCols):
            colI = rowVals[0][j]
            valDiff = abs(rowVals[1][j])
            valRef = abs(jacMat1[i, colI])
            if mode == "rel":
                valError = valDiff / (valRef + diffTol)
            elif mode == "abs":
                valError = valDiff
            else:
                print("mode not supported! Options are: abs or rel")
            l2norm = l2norm + valDiff**2
            if abs(valError) > diffTol:
                if abs(valError) > maxDiff:
                    foundDiff = 1
                    maxDiff = valError
                    maxRowI = i
                    maxColI = colI
                    maxVal1 = jacMat1.getValue(i, colI)
                    maxVal2 = jacMat2.getValue(i, colI)

    if foundDiff == 1:
        print("L2Norm: %20.16e" % l2norm)
        print("MaxDiff: %20.16e" % maxDiff)
        print("MaxVal1: %20.16e" % maxVal1)
        print("MaxVal2: %20.16e" % maxVal2)
        print("MaxrowI: %d" % maxRowI)
        print("MaxcolI: %d" % maxColI)
        return maxDiff
    else:
        print("Two matrices are exactly same with tolerance: %e" % diffTol)
        return 0.0, 0.0


if __name__ == "__main__":
    print("\nUsage: python dafoam_matreldiff.py matName1 matName2 mode")
    print("Example python dafoam_matreldiff.py dRdWT1.bin dRdWT2.bin abs\n")
    evalMatDiff(sys.argv[1], sys.argv[2], sys.argv[3])
