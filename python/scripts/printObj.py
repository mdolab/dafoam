#!/usr/bin/env python
'''
Read objective information from log files and print the mean, std, figure etc.
'''

import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--logName", help="names of log files, e.g., --logName=%s['flowLog','flowLog1']%s "%('"','"'), type=str,default="['flowLog']")
parser.add_argument("--keyword", help="names of objective functions, e.g., --keyword=%s['CD','CL']%s "%('"','"'), type=str, default="['CD']")
parser.add_argument("--readCol", help="which column to read, e.g., --readCol=%s[1,2]%s "%('"','"'), type=str, default="[1]")
parser.add_argument('--xLabel',type=str,help='x label?',default='Steps')
parser.add_argument('--yLabel',type=str,help='y label?',default='Objective')
parser.add_argument("--avgStart", help="start averaging from how much percentage of the time steps?", type=float, default=0.8)
parser.add_argument('--plot',type=int,help='plot the log?',default=1)
parser.add_argument('--logScaleX',type=int,help='plot the x-axis logscale?',default=0)
parser.add_argument('--logScaleY',type=int,help='plot the y-axis logscale?',default=0)
parser.add_argument('--xOffset',type=str,help="offset x values? e.g., --xoffset=%s[0.0,1.0]%s "%('"','"'),default="[0]")
parser.add_argument('--yOffset',type=str,help="offset y values? e.g., --yoffset=%s[0.0,1.0]%s "%('"','"'),default="[0]")
parser.add_argument('--xScale',type=str,help="scale x values? e.g., --xscale=%s[1.0,2.0]%s "%('"','"'),default="[1]")
parser.add_argument('--yScale',type=str,help="scale y values? e.g., --yscale=%s[1.0,2.0]%s "%('"','"'),default="[1]")
parser.add_argument('--legend',type=str,help="legend for the lines e.g., --legend=%s['CD','CL']%s "%('"','"'),default="['CD']")
args = parser.parse_args()

# read lists
exec('args_logName=%s'%args.logName)
nDims=len(args_logName)

exec('args_keyword=%s'%args.keyword)
exec('args_readCol=%s'%args.readCol)
exec('args_xOffset=%s'%args.xOffset)
exec('args_yOffset=%s'%args.yOffset)
exec('args_xScale=%s'%args.xScale)
exec('args_yScale=%s'%args.yScale)
exec('args_legend=%s'%args.legend)

if len(args_keyword)<nDims and len(args_keyword)>0:
    for i in xrange(len(args_keyword),nDims):
        args_keyword.append(args_keyword[0])
if len(args_readCol)<nDims and len(args_readCol)>0:
    for i in xrange(len(args_readCol),nDims):
        args_readCol.append(args_readCol[0])
if len(args_xOffset)<nDims and len(args_xOffset)>0:
    for i in xrange(len(args_xOffset),nDims):
        args_xOffset.append(args_xOffset[0])
if len(args_yOffset)<nDims and len(args_yOffset)>0:
    for i in xrange(len(args_yOffset),nDims):
        args_yOffset.append(args_yOffset[0])
if len(args_xScale)<nDims and len(args_xScale)>0:
    for i in xrange(len(args_xScale),nDims):
        args_xScale.append(args_xScale[0])
if len(args_yScale)<nDims and len(args_yScale)>0:
    for i in xrange(len(args_yScale),nDims):
        args_yScale.append(args_yScale[0])
if len(args_legend)<nDims and len(args_legend)>0:
    for i in xrange(len(args_legend),nDims):
        args_legend.append(args_legend[0])

for i in range(nDims):

    fIn = open(args_logName[i],'r')
    
    lines = fIn.readlines()
    
    objFunc=[]
    iteration=[]
    step = 0
    for line in lines:
        if args_keyword[i] in line:
            cols = line.split()
            xVal = args_xScale[i]*( step+args_xOffset[i] )
            yVal = args_yScale[i]*( float(cols[args_readCol[i]])+args_yOffset[i] )
            objFunc.append(yVal)
            iteration.append(xVal)
            step +=1
    fIn.close()
    
    stepStart = step*args.avgStart # average using the last 80% of the time steps
    
    objFuncMean = 0.0
    nCount = 0
    for j in range(len(objFunc)):
        if j >= stepStart:
            objFuncMean += objFunc[j]
            nCount += 1
    if nCount==0:
        print("keyword %s not found in %s!"%(args_keyword[i],args_logName[i]))
        exit(1)
    objFuncMean /= nCount
    
    objFuncStd = 0.0
    nCount = 0
    for j in range(len(objFunc)):
        if j >= stepStart:
            objFuncStd += (objFunc[j]-objFuncMean)**2.0
            nCount += 1
    objFuncStd /= nCount   
    objFuncStd = objFuncStd**0.5
    
    print('\nIn '+args_logName[i])
    print('Averaging objective function from step: '+str(stepStart))
    print('Mean: '+str(objFuncMean))
    print('Standard deviation: '+str(objFuncStd))

    if args.plot:
        plt.plot(iteration,objFunc,label=args_legend[i])

        plt.legend(frameon=False,loc=0)
    
        plt.xlabel(args.xLabel)
        plt.ylabel(args.yLabel)
    
        if args.logScaleX:
            plt.xscale('log')
        
        if args.logScaleY:
            plt.yscale('log')

if args.plot:
    plt.show()
else:
    plt.savefig('printObj.png',bbox_inches='tight')
    plt.close()

