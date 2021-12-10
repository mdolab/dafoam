#!/usr/bin/env python
"""
This script converts a Plot3D file to a Tecplot file

Usage: python dafoam_plot3d2tecplot.py plot3d_file_input.xyz tecplot_file_output.dat
"""

import sys
from pygeo import *

inputFileName = sys.argv[1]
outputFileName = sys.argv[2]

print("Converting %s to %s...." % (inputFileName, outputFileName))

DVGeo = DVGeometry(inputFileName)
DVGeo.writeTecplot(outputFileName)

print("Converting %s to %s.... Done!" % (inputFileName, outputFileName))
