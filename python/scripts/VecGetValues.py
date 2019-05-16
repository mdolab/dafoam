#!/usr/bin/env python
'''
Read a PETSc vector and print the value(s) at given row(s)
'''
import os,sys
import argparse
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

parser = argparse.ArgumentParser(description='Read a PETSc vector and print the value(s) at given row(s)')
parser.add_argument("--vec", help='path of the vector in binary format', type=str,default='')
parser.add_argument("--vecsize", help="size of the vector", type=int, default=1)
parser.add_argument("--vecindx", help="which element to show, if left default, print the entire vector", type=int, default=-9999)
args = parser.parse_args()

if len(args.vec)==0:
    print("--vec not defined!")
    exit()

# read the vector
vec1 = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
vec1.setSizes( (PETSc.DECIDE,args.vecsize),bsize=1)
vec1.setFromOptions()
viewer = PETSc.Viewer().createBinary(args.vec,comm=PETSc.COMM_WORLD)
vec1.load(viewer)

for i in range(args.vecsize):
    if (args.vecindx==-9999) or ( i == args.vecindx ):
        print('%12d %16.14e'%(i,vec1.getValue(i)))