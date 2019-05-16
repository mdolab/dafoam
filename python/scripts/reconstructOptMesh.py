#!/usr/bin/env python
'''
Reconstruct the decomposed domain for the optimized results.
Note: simply do a reconstruct will use the mesh in constant/polyMesh that is not deformed.
Here we need to use the most-up-to-date point coordinates in processor#/constant/polyMesh
'''

import argparse
import subprocess
import os,re,shutil

parser = argparse.ArgumentParser(description='Reconstruct the decomposed domain for the optimized results.')
parser.add_argument("--prefix", help='prefix of the folder names, e.g., AhmedBody_opt_shape_adjoint_. If left blank, reconstruct current folder', type=str,default='')
parser.add_argument("--index1", help="index from which we reconstruct, not setting this value will reconstruct all indices", type=int, default=-9999)
parser.add_argument("--index2", help="index to which we reconstruct, not setting this value will reconstruct all indices", type=int, default=-9999)
parser.add_argument("--time", help="which time step to reconstruct, not setting this value will reconstruct the latest time", type=int, default=-9999)
args = parser.parse_args()

# main reconstruct function
def reconstructFunc():
    # find nProc
    myDir=os.getcwd()
    nProc=-9999
    for f in os.listdir(myDir):
        if (os.path.isdir(f)) and ("processor" in f):
            idx=re.findall(r'\d+',f)
            if nProc<int(idx[0]):
                nProc=int(idx[0])
    nProc=nProc+1
    print("nProc: %d"%nProc)

    # find which time step to reconstruct
    maxTime=-9999
    if args.time==-9999:
        procDir=os.path.join(myDir,'processor0')
        for f in os.listdir(procDir):
            if f.isdigit():
                if maxTime<int(f):
                    maxTime=int(f)
    else:
        maxTime=int(args.time)
    print('time: %d'%maxTime)
    for i in range(nProc):
        origDir=os.path.join(myDir,'processor%d/constant/polyMesh'%i)
        newDir=os.path.join(myDir,'processor%d/%d/polyMesh'%(i,int(maxTime)))
        if os.path.isdir(newDir):
            shutil.rmtree(newDir)
        shutil.copytree(origDir,newDir)
    
    subprocess.call(['reconstructParMesh','-latestTime'], shell=True)
    subprocess.call(['reconstructPar'], shell=True)

    return


if len(args.prefix)==0:
    reconCurrent=True
else:
    reconCurrent=False

if reconCurrent==True:
    # reconstruct current folder
    reconstructFunc()
            
else:
    # reconstruct specified folders

    currDir=os.getcwd()
    # find index1 and index2
    index1=9999
    if args.index1==-9999:
        for f in os.listdir(currDir):
            if (os.path.isdir(f)) and (args.prefix in f):
                idx=re.findall(r'\d+',f)
                if int(idx[0])<index1:
                    index1=int(idx[0])
    else:
        index1=args.index1
    
    index2=-9999
    if args.index2==-9999:
        for f in os.listdir(currDir):
            if (os.path.isdir(f)) and (args.prefix in f):
                idx=re.findall(r'\d+',f)
                if int(idx[0])>index2:
                    index2=int(idx[0])
    else:
        index2=args.index2
    
    idx1=index1
    idx2=index2+1
    for i in xrange(idx1,idx2):
        os.chdir( os.path.join(currDir,args.prefix+'%03d'%i) )
        reconstructFunc()
        os.chdir('../')
            
