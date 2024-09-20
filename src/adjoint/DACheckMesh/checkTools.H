/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    applications/utilities/mesh/manipulation/checkMesh/checkTools.H

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

#include "scalar.H"
#include "indirectPrimitivePatch.H"
#include "writer.H"

namespace Foam
{
class polyMesh;
class surfaceWriter;
class pointSet;
class faceSet;
class cellSet;
class fileName;
class polyMesh;

//- Generate merged surface on master and write. Needs input patch
//  to be of mesh faces.
void mergeAndWrite(
    const polyMesh& mesh,
    const surfaceWriter& writer,
    const word& name,
    const indirectPrimitivePatch& setPatch,
    const fileName& outputDir);

//- Write vtk representation of (assembled) faceSet to surface file in
//  postProcessing/ directory
void mergeAndWrite(const surfaceWriter&, const faceSet&);

//- Write vtk representation of (assembled) cellSet to surface file in
//  postProcessing/ directory
void mergeAndWrite(const surfaceWriter&, const cellSet&);

//- Write vtk representation of (assembled) pointSet to 'set' file in
//  postProcessing/ directory
void mergeAndWrite(const writer<scalar>&, const pointSet&);
}

// ************************************************************************* //
