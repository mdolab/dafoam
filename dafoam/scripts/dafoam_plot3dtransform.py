#!/usr/bin/env python
"""
This script transforms the coordinates in a plot3D file. We support three modes: scale, translate, or rotate

Usage: 

    python dafoam_plot3dscale.py scale plot3d_file_input.xyz plot3d_file_output.xyz 2 2 2
    This will scale the x, y, and z coordinates by a factor of 2
    
    python dafoam_plot3dscale.py translate plot3d_file_input.xyz plot3d_file_output.xyz 1 2 3
    This will translate the x, y, and z coordinates by 1, 2, and 3
    
    python dafoam_plot3dscale.py rotate plot3d_file_input.xyz plot3d_file_output.xyz x 10
    This will rotate the x, y, and z coordinates with respect to the x axis by 10 degree
"""

import sys
from pygeo import *
import numpy as np

mode = sys.argv[1]

inputFileName = sys.argv[2]
outputFileName = sys.argv[3]

if mode == "scale":
    scaleX = float(sys.argv[4])
    scaleY = float(sys.argv[5])
    scaleZ = float(sys.argv[6])
    
    print(
        "Scaling %s to %s with scaleX: %g scaleY: %g scaleZ: %g ...."
        % (inputFileName, outputFileName, scaleX, scaleY, scaleZ)
    )
    
    ffd = pyBlock("plot3d", fileName=inputFileName, FFD=True)
    
    for ivol in range(ffd.nVol):
        vals = ffd.vols[ivol].coef
        vals[:, :, :, 0] = vals[:, :, :, 0] * scaleX
        vals[:, :, :, 1] = vals[:, :, :, 1] * scaleY
        vals[:, :, :, 2] = vals[:, :, :, 2] * scaleZ
    ffd.writePlot3dCoef(outputFileName)
    
    print(
        "Scaling %s to %s with scaleX: %g scaleY: %g scaleZ: %g Done!"
        % (inputFileName, outputFileName, scaleX, scaleY, scaleZ)
    )
elif mode == "translate":
    dX = float(sys.argv[4])
    dY = float(sys.argv[5])
    dZ = float(sys.argv[6])
    
    print(
        "Translating %s to %s with dX: %g dY: %g dZ: %g ...."
        % (inputFileName, outputFileName, dX, dY, dZ)
    )
    
    ffd = pyBlock("plot3d", fileName=inputFileName, FFD=True)
    
    for ivol in range(ffd.nVol):
        vals = ffd.vols[ivol].coef
        vals[:, :, :, 0] = vals[:, :, :, 0] + dX
        vals[:, :, :, 1] = vals[:, :, :, 1] + dY
        vals[:, :, :, 2] = vals[:, :, :, 2] + dZ
    ffd.writePlot3dCoef(outputFileName)
    
    print(
        "Translating %s to %s with dX: %g dT: %g dZ: %g Done!"
        % (inputFileName, outputFileName, dX, dY, dZ)
    )
elif mode == "rotate":
    axis = str(sys.argv[4])
    deg = float(sys.argv[5])

    print(
        "Rotating %s to %s wrt to the %s axis by %g degree...."
        % (inputFileName, outputFileName, axis, deg)
    )

    ffd = pyBlock("plot3d", fileName=inputFileName, FFD=True)
    
    theta = deg * np.pi / 180.0

    for ivol in range(ffd.nVol):
        vals = ffd.vols[ivol].coef
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                for k in range(vals.shape[2]):
                    if axis == "x":
                        yRef = vals[i, j, k, 1]
                        zRef = vals[i, j, k, 2]
                        vals[i, j, k, 1] = np.cos(theta) * yRef - np.sin(theta) * zRef 
                        vals[i, j, k, 2] = np.sin(theta) * yRef + np.cos(theta) * zRef 
                    elif axis == "y":
                        xRef = vals[i, j, k, 0]
                        zRef = vals[i, j, k, 2]
                        vals[i, j, k, 0] = np.cos(theta) * xRef + np.sin(theta) * zRef 
                        vals[i, j, k, 2] = -np.sin(theta) * xRef + np.cos(theta) * zRef 
                    elif axis == "z":
                        xRef = vals[i, j, k, 0]
                        yRef = vals[i, j, k, 1]
                        vals[i, j, k, 0] = np.cos(theta) * xRef - np.sin(theta) * yRef 
                        vals[i, j, k, 1] = np.sin(theta) * xRef + np.cos(theta) * yRef 
                    else:
                        print("Axis %s not supported! Options are: x, y, or z"%s)
    
    ffd.writePlot3dCoef(outputFileName)

    print(
        "Rotating %s to %s wrt to the %s axis by %g degree Done!"
        % (inputFileName, outputFileName, axis, deg)
    )
else:
    print("Mode %s not supported! Options are: scale, translate, or rotate"%s)
