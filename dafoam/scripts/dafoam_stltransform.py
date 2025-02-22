#!/usr/bin/env python
"""
This script transforms the coordinates in a stl file. We support three modes: scale, translate, or rotate

Usage: 

    python dafoam_plot3dscale.py scale plot3d_file_input.xyz plot3d_file_output.xyz 2 2 2
    This will scale the x, y, and z coordinates by a factor of 2
    
    python dafoam_plot3dscale.py translate plot3d_file_input.xyz plot3d_file_output.xyz 1 2 3
    This will translate the x, y, and z coordinates by 1, 2, and 3
    
    python dafoam_plot3dscale.py rotate plot3d_file_input.xyz plot3d_file_output.xyz x 10
    This will rotate the x, y, and z coordinates with respect to the x axis by 10 degree
"""

import sys
from stl import mesh
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

    myMesh = mesh.Mesh.from_file(inputFileName)

    myMesh.x *= scaleX
    myMesh.y *= scaleY
    myMesh.z *= scaleZ

    myMesh.save(outputFileName)

    print(
        "Scaling %s to %s with scaleX: %g scaleY: %g scaleZ: %g Done!"
        % (inputFileName, outputFileName, scaleX, scaleY, scaleZ)
    )
elif mode == "translate":
    dX = float(sys.argv[4])
    dY = float(sys.argv[5])
    dZ = float(sys.argv[6])

    print("Translating %s to %s with dX: %g dY: %g dZ: %g ...." % (inputFileName, outputFileName, dX, dY, dZ))

    myMesh = mesh.Mesh.from_file(inputFileName)

    myMesh.x += dX
    myMesh.y += dY
    myMesh.z += dZ

    myMesh.save(outputFileName)

    print("Translating %s to %s with dX: %g dT: %g dZ: %g Done!" % (inputFileName, outputFileName, dX, dY, dZ))
elif mode == "rotate":
    axis = str(sys.argv[4])
    deg = float(sys.argv[5])

    print("Rotating %s to %s wrt to the %s axis by %g degree...." % (inputFileName, outputFileName, axis, deg))

    # ************** NOTE ******************
    # we have to add a minus sign here because numpy-stl is known
    # to have a problem for the direction of the rotation. It somehow rotates
    # clockwise for a positive angle, which is not consistent with the conventional
    # right-hand-side rule of rotation.
    # Check their documentation https://numpy-stl.readthedocs.io/en/latest/stl.html
    theta = -deg * np.pi / 180.0

    myMesh = mesh.Mesh.from_file(inputFileName)

    if axis == "x":
        myMesh.rotate([1.0, 0.0, 0.0], theta)
    elif axis == "y":
        myMesh.rotate([0.0, 1.0, 0.0], theta)
    elif axis == "z":
        myMesh.rotate([0.0, 0.0, 1.0], theta)
    else:
        print("Axis %s not supported! Options are: x, y, or z" % s)

    myMesh.save(outputFileName)

    print("Rotating %s to %s wrt to the %s axis by %g degree Done!" % (inputFileName, outputFileName, axis, deg))
else:
    print("Mode %s not supported! Options are: scale, translate, or rotate" % s)
