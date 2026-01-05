/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    This file is modified from OpenFOAM's source code
    applications/utilities/mesh/manipulation/checkMesh/checkMeshGeometry.C

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "label.H"
#include "HashSet.H"
#include "labelVector.H"

namespace Foam
{
class polyMesh;
class wedgePolyPatch;
class coordSetWriter;
class surfaceWriter;

/// Check 0th vertex on coupled faces
bool checkCoupledPoints(const polyMesh&, const bool report, labelHashSet*);

/// check mesh quality
label checkGeometry(
    const polyMesh& mesh,
    const autoPtr<surfaceWriter>& surfWriter,
    const autoPtr<coordSetWriter>& setWriter,
    const label maxIncorrectlyOrientedFaces);

}
