#!/usr/bin/env python
"""
Read a PETSc vector and print the value(s) at given row(s)
"""
import os, sys
import argparse
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc


def printVecValues(vecName, rowI, diffTol=1e-30):

    # read the vector
    vec1 = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(vecName, comm=PETSc.COMM_WORLD)
    vec1.load(viewer)

    rowI = int(rowI)

    vecSize = vec1.getSize()

    if rowI == -1:
        for i in range(vecSize):
            if abs(vec1.getValue(i)) > diffTol:
                print("%12d %16.14e" % (i, vec1.getValue(i)))
    else:
        print("%12d %16.14e" % (rowI, vec1.getValue(rowI)))


if __name__ == "__main__":
    print("\nUsage: python dafoam_vecgetvalues.py vecName rowI")
    print("Example python dafoam_vecgetvalues.py dFdW.bin 100")
    print("NOTE: if rowI=-1, print all elements\n")
    printVecValues(sys.argv[1], sys.argv[2])
