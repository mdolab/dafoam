#!/usr/bin/env python
'''
Read a PETSc matrix and output the sparsity pattern in tecplot format
Example:
python MatSparsity.py matName
'''
import os,sys
import argparse
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

def outputSparsity(mat):

    # read the Jac mat
    jacMat = PETSc.Mat().create(PETSc.COMM_WORLD)
    viewer = PETSc.Viewer().createBinary(mat,comm=PETSc.COMM_WORLD)
    jacMat.load(viewer)
    Istart, Iend = jacMat.getOwnershipRange()
    m,n=jacMat.getSize()

    f=open(mat+'Sparsity.dat','w')
    for i in range(m):
        rowVals=jacMat.getRow(int(i))
        nCols=len(rowVals[0])
        for j in range(nCols):
            jacVal = rowVals[1][j]
            jacColI = rowVals[0][j]

            if abs(jacVal)>1e-12:
                f.write("GEOMETRY X=%d, Y=%d, T=CIRCLE, C=BLACK, FC=BLACK,CS=GRID\n"%(i,-jacColI))
                f.write("1  #RADIUS\n")

    f.close()
    print ("file written to "+mat+'Sparsity.dat')

if __name__ == '__main__':
    print("\nUsage: python MatSparsity matName")
    outputSparsity(sys.argv[1])
