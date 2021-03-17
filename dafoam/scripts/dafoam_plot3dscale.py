#!/usr/bin/env python
"""
This script scale the coordinates in a plot3D file

Usage: python dafoam_plot3dscale.py plot3d_file_input.xyz plot3d_file_output.xyz 2 2 2
This will scale the x, y, and z coordinates by a factor of 2
"""

import sys
from pygeo import *

inputFileName = sys.argv[1]
outputFileName = sys.argv[2]
scaleX = float(sys.argv[3])
scaleY = float(sys.argv[4])
scaleZ = float(sys.argv[5])

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
