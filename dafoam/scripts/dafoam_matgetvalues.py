#!/usr/bin/env python
"""
Read a PETSc matrix and print the value(s) at given row and column(s)
"""
import os, sys
import argparse
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc


def printMatValues(mat, rowI, colI, transposed, diffTol = 1e-30):

    # read the Jac mat
    jacMat = PETSc.Mat().create(PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(mat, comm=PETSc.COMM_WORLD)
    jacMat.load(viewer)

    if transposed == "1":
        print("Getting transposed values")
        jacMat.transpose()
    elif transposed == "0":
        pass
    else:
        print("Error!!!transposed should be either 0 or 1")
        exit()

    rowVals = jacMat.getRow(int(rowI))
    nCols = len(rowVals[0])

    for i in range(nCols):
        if (int(colI) == -1) or (int(rowVals[0][i]) == int(colI)):
            if rowVals[1][i] > diffTol:
                print("%16d %20.16e" % (rowVals[0][i], rowVals[1][i]))


if __name__ == "__main__":
    print("\nUsage: python dafoam_matgetvalues.py matName rowI colI transposed")
    print("Example python dafoam_matgetvalues.py dRdWT.bin 100 25 0")
    print("NOTE: if colI=-1, print all columns\n")
    printMatValues(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
